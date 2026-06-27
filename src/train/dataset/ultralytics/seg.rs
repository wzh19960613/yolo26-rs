use super::*;

impl SegmentationDataset {
    /// Creates a segmentation dataset from a detection dataset and target mask size.
    pub fn new(detect: DetectionDataset, mask_size: ImageSize) -> crate::Result<Self> {
        Self::new_with_overlap_mask(detect, mask_size, true)
    }

    /// Creates a segmentation dataset with explicit overlap-mask encoding.
    pub fn new_with_overlap_mask(
        detect: DetectionDataset,
        mask_size: ImageSize,
        overlap_mask: bool,
    ) -> crate::Result<Self> {
        if mask_size.width == 0 || mask_size.height == 0 {
            return Err(crate::Error::InvalidConfig(
                "segmentation mask dimensions must be greater than zero".to_string(),
            ));
        }
        Ok(Self {
            detect,
            mask_size,
            overlap_mask,
        })
    }

    /// Returns whether this dataset emits official overlap instance-index masks.
    pub const fn overlap_mask(&self) -> bool {
        self.overlap_mask
    }

    /// Returns the image paths in deterministic order.
    pub fn image_paths(&self) -> &[PathBuf] {
        self.detect.image_paths()
    }

    /// Uses an Ultralytics-style rectangular validation canvas while retaining
    /// the configured resize size.
    pub fn with_rect_canvas_size(mut self, canvas_size: ImageSize) -> crate::Result<Self> {
        self.detect = self.detect.with_rect_canvas_size(canvas_size)?;
        Ok(self)
    }

    fn sample_segmentation(&self, index: usize) -> crate::Result<Sample> {
        let image_path = self.detect.images.get(index).ok_or_else(|| {
            crate::Error::InvalidConfig(format!(
                "sample index {index} is out of bounds for dataset length {}",
                self.detect.images.len()
            ))
        })?;
        let image = read_rgb_image(image_path)?;
        let source_w = image.width as f32;
        let source_h = image.height as f32;
        let (input, letterbox) = self.detect.letterbox_image(&image)?;
        let labels = read_yolo_segmentation_labels(&label_path_for_image(image_path))?;
        let mut labels = prepare_segmentation_labels(
            labels,
            self.detect.max_objects,
            source_w,
            source_h,
            &letterbox,
            self.mask_size,
        );
        if self.overlap_mask {
            labels.sort_by(|left, right| right.area.total_cmp(&left.area));
        }
        let mut boxes = vec![0f32; self.detect.max_objects * 4];
        let mut class_ids = vec![0u32; self.detect.max_objects];
        let mut valid = vec![0f32; self.detect.max_objects];
        let mask_pixels = self.mask_size.width * self.mask_size.height;
        let mask_channels = if self.overlap_mask {
            1
        } else {
            self.detect.max_objects
        };
        let mut masks = vec![0f32; mask_channels * mask_pixels];

        for (idx, label) in labels.iter().enumerate() {
            class_ids[idx] = label.class_id;
            valid[idx] = 1.0;
            boxes[idx * 4..idx * 4 + 4].copy_from_slice(&label.box_xyxy);
            if self.overlap_mask {
                write_overlap_mask(idx, label, &mut masks);
            } else {
                let dst = idx * mask_pixels..(idx + 1) * mask_pixels;
                masks[dst].copy_from_slice(&label.mask);
            }
        }

        let boxes_xyxy =
            Tensor::from_vec(boxes, (1, self.detect.max_objects, 4), &self.detect.device)?;
        let class_ids =
            Tensor::from_vec(class_ids, (1, self.detect.max_objects), &self.detect.device)?;
        let valid = Tensor::from_vec(valid, (1, self.detect.max_objects), &self.detect.device)?;
        let detection = DetectionTargets::new(boxes_xyxy, class_ids, valid)?;
        let masks = Tensor::from_vec(
            masks,
            (
                1,
                mask_channels,
                self.mask_size.height,
                self.mask_size.width,
            ),
            &self.detect.device,
        )?;
        let targets = if self.overlap_mask {
            SegmentationTargets::new_overlap(detection, masks)?
        } else {
            SegmentationTargets::new(detection, masks)?
        };
        Ok(Sample {
            input,
            target: Target::Segmentation(targets),
        })
    }
}

impl Dataset for SegmentationDataset {
    fn len(&self) -> usize {
        self.detect.len()
    }

    fn sample(&self, index: usize) -> crate::Result<Sample> {
        self.sample_segmentation(index)
    }
}

struct PreparedSegmentationLabel {
    class_id: u32,
    box_xyxy: [f32; 4],
    mask: Vec<f32>,
    area: f32,
}

fn prepare_segmentation_labels(
    labels: Vec<YoloSegmentationLabel>,
    max_objects: usize,
    source_w: f32,
    source_h: f32,
    letterbox: &crate::model::LetterboxInfo,
    mask_size: ImageSize,
) -> Vec<PreparedSegmentationLabel> {
    let mut prepared = Vec::new();
    for label in labels.into_iter().take(max_objects) {
        let model_points = label
            .points
            .iter()
            .map(|&(x, y)| {
                (
                    x * source_w * letterbox.scale + letterbox.pad_x,
                    y * source_h * letterbox.scale + letterbox.pad_y,
                )
            })
            .collect::<Vec<_>>();
        let Some((x1, y1, x2, y2)) = polygon_bounds(&model_points) else {
            continue;
        };
        let box_xyxy = [
            x1.clamp(0.0, letterbox.model_width as f32),
            y1.clamp(0.0, letterbox.model_height as f32),
            x2.clamp(0.0, letterbox.model_width as f32),
            y2.clamp(0.0, letterbox.model_height as f32),
        ];
        if box_xyxy[2] <= box_xyxy[0] || box_xyxy[3] <= box_xyxy[1] {
            continue;
        }
        let mask_points = resample_segment(&model_points);
        let mask = rasterize_official_segmentation_mask(
            &mask_points,
            letterbox.model_width,
            letterbox.model_height,
            mask_size.width,
            mask_size.height,
        );
        let area = mask.iter().sum::<f32>();
        if area > 0.0 {
            prepared.push(PreparedSegmentationLabel {
                class_id: label.class_id,
                box_xyxy,
                mask,
                area,
            });
        }
    }
    prepared
}

fn resample_segment(points: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if points.is_empty() {
        return Vec::new();
    }
    let target = if points.len() > 1000 {
        points.len() + 1
    } else {
        1000
    };
    if points.len() == target {
        return points.to_vec();
    }
    let mut closed = points.to_vec();
    closed.push(points[0]);
    if closed.len() >= target {
        return linspace_interp_segment(&closed, target);
    }
    let base_count = target - closed.len();
    let end = (closed.len() - 1) as f32;
    let mut samples = Vec::with_capacity(target);
    if base_count == 1 {
        samples.push(0.0);
    } else {
        for idx in 0..base_count {
            samples.push(idx as f32 * end / (base_count - 1) as f32);
        }
    }
    samples.extend((0..closed.len()).map(|idx| idx as f32));
    samples.sort_by(|left, right| left.total_cmp(right));
    samples
        .into_iter()
        .map(|x| interp_closed_segment(&closed, x))
        .collect()
}

fn linspace_interp_segment(points: &[(f32, f32)], count: usize) -> Vec<(f32, f32)> {
    if count <= 1 {
        return vec![points[0]];
    }
    let end = (points.len() - 1) as f32;
    (0..count)
        .map(|idx| interp_closed_segment(points, idx as f32 * end / (count - 1) as f32))
        .collect()
}

fn interp_closed_segment(points: &[(f32, f32)], x: f32) -> (f32, f32) {
    let max_idx = points.len().saturating_sub(1);
    let left = x.floor().clamp(0.0, max_idx as f32) as usize;
    let right = (left + 1).min(max_idx);
    let t = (x - left as f32).clamp(0.0, 1.0);
    (
        points[left].0 + (points[right].0 - points[left].0) * t,
        points[left].1 + (points[right].1 - points[left].1) * t,
    )
}

fn rasterize_official_segmentation_mask(
    points: &[(f32, f32)],
    model_width: usize,
    model_height: usize,
    mask_width: usize,
    mask_height: usize,
) -> Vec<f32> {
    if points.len() < 3
        || model_width == 0
        || model_height == 0
        || mask_width == 0
        || mask_height == 0
    {
        return vec![0.0; mask_width * mask_height];
    }
    let int_points = points
        .iter()
        .map(|&(x, y)| (x as i32, y as i32))
        .collect::<Vec<_>>();
    let mut full = vec![0u8; model_width * model_height];
    fill_polygon_i32(&int_points, model_width, model_height, &mut full);
    resize_linear_u8_to_f32(&full, model_width, model_height, mask_width, mask_height)
}

fn fill_polygon_i32(points: &[(i32, i32)], width: usize, height: usize, mask: &mut [u8]) {
    if points.len() < 3 || mask.len() != width * height {
        return;
    }
    let min_x = points
        .iter()
        .map(|(x, _)| *x)
        .min()
        .unwrap_or(0)
        .clamp(0, width.saturating_sub(1) as i32) as usize;
    let min_y = points
        .iter()
        .map(|(_, y)| *y)
        .min()
        .unwrap_or(0)
        .clamp(0, height.saturating_sub(1) as i32) as usize;
    let max_x = points
        .iter()
        .map(|(x, _)| *x)
        .max()
        .unwrap_or(0)
        .clamp(0, width.saturating_sub(1) as i32) as usize;
    let max_y = points
        .iter()
        .map(|(_, y)| *y)
        .max()
        .unwrap_or(0)
        .clamp(0, height.saturating_sub(1) as i32) as usize;
    for y in min_y..=max_y {
        let mut xs = Vec::new();
        let yy = y as f32;
        for idx in 0..points.len() {
            let (x1, y1) = points[idx];
            let (x2, y2) = points[(idx + 1) % points.len()];
            if y1 == y2 {
                continue;
            }
            let y_i = y as i32;
            if (y1 <= y_i && y_i < y2) || (y2 <= y_i && y_i < y1) {
                let t = (yy - y1 as f32) / (y2 - y1) as f32;
                xs.push(x1 as f32 + t * (x2 - x1) as f32);
            }
        }
        xs.sort_by(|left, right| left.total_cmp(right));
        for pair in xs.chunks_exact(2) {
            let start = pair[0].floor().max(min_x as f32) as usize;
            let end = pair[1].ceil().min(max_x as f32) as usize;
            if start <= end {
                let row = y * width;
                mask[row + start..=row + end].fill(1);
            }
        }
    }
    for idx in 0..points.len() {
        let (x1, y1) = points[idx];
        let (x2, y2) = points[(idx + 1) % points.len()];
        draw_line_i32(x1, y1, x2, y2, width, height, mask);
    }
}

fn draw_line_i32(
    mut x0: i32,
    mut y0: i32,
    x1: i32,
    y1: i32,
    width: usize,
    height: usize,
    mask: &mut [u8],
) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        if x0 >= 0 && y0 >= 0 && (x0 as usize) < width && (y0 as usize) < height {
            mask[y0 as usize * width + x0 as usize] = 1;
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn resize_linear_u8_to_f32(
    source: &[u8],
    source_width: usize,
    source_height: usize,
    target_width: usize,
    target_height: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; target_width * target_height];
    let scale_x = source_width as f32 / target_width as f32;
    let scale_y = source_height as f32 / target_height as f32;
    for y in 0..target_height {
        let src_y = (y as f32 + 0.5) * scale_y - 0.5;
        let y0 = src_y.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(source_height - 1);
        let fy = (src_y - y0 as f32).max(0.0);
        for x in 0..target_width {
            let src_x = (x as f32 + 0.5) * scale_x - 0.5;
            let x0 = src_x.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(source_width - 1);
            let fx = (src_x - x0 as f32).max(0.0);
            let v00 = source[y0 * source_width + x0] as f32;
            let v01 = source[y0 * source_width + x1] as f32;
            let v10 = source[y1 * source_width + x0] as f32;
            let v11 = source[y1 * source_width + x1] as f32;
            let value = v00 * (1.0 - fx) * (1.0 - fy)
                + v01 * fx * (1.0 - fy)
                + v10 * (1.0 - fx) * fy
                + v11 * fx * fy;
            out[y * target_width + x] = (value + 0.5).floor().clamp(0.0, 1.0);
        }
    }
    out
}

fn write_overlap_mask(index: usize, label: &PreparedSegmentationLabel, masks: &mut [f32]) {
    let value = (index + 1) as f32;
    for (dst, &src) in masks.iter_mut().zip(&label.mask) {
        if src > 0.0 {
            *dst = value;
        }
    }
}

/// Builds an instance segmentation dataset for one split from an Ultralytics YAML file.
pub fn from_file(
    path: impl AsRef<Path>,
    split: Split,
    image_size: ImageSize,
    mask_size: ImageSize,
    dtype: DType,
    device: Device,
    max_objects: usize,
) -> crate::Result<SegmentationDataset> {
    Yaml::segmentation_dataset_from_file(
        path,
        split,
        image_size,
        mask_size,
        dtype,
        device,
        max_objects,
    )
}
