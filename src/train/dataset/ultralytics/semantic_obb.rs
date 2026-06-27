use super::*;
use crate::train::eval::obb_geometry;

/// Semantic segmentation dataset backed by image directories and class-map masks.
#[derive(Debug, Clone)]
pub struct SemanticDataset {
    pub(crate) dataset: Yaml,
    pub(crate) root: PathBuf,
    pub(crate) split: Split,
    pub(crate) images: Vec<PathBuf>,
    pub(crate) image_size: ImageSize,
    pub(crate) output_size: ImageSize,
    pub(crate) dtype: DType,
    pub(crate) device: Device,
}

/// OBB dataset backed by Ultralytics image directories and four-corner labels.
#[derive(Debug, Clone)]
pub struct ObbDataset {
    pub(crate) detect: DetectionDataset,
}

impl SemanticDataset {
    /// Creates a semantic segmentation dataset from parsed YAML metadata and a resolved root.
    pub fn new(
        dataset: Yaml,
        root: PathBuf,
        split: Split,
        image_size: ImageSize,
        output_size: ImageSize,
        dtype: DType,
        device: Device,
    ) -> crate::Result<Self> {
        if output_size.width == 0 || output_size.height == 0 {
            return Err(crate::Error::InvalidConfig(
                "semantic output dimensions must be greater than zero".to_string(),
            ));
        }
        let split_value = split.value(&dataset).ok_or_else(|| {
            crate::Error::InvalidConfig(format!(
                "Ultralytics dataset YAML does not define {split:?}"
            ))
        })?;
        let split_path = resolve_dataset_path(&root, split_value);
        let images = collect_image_paths(&root, &split_path)?;
        if images.is_empty() {
            return Err(crate::Error::InvalidConfig(format!(
                "Ultralytics {split:?} split contains no supported images at {}",
                split_path.display()
            )));
        }
        Ok(Self {
            dataset,
            root,
            split,
            images,
            image_size,
            output_size,
            dtype,
            device,
        })
    }

    /// Returns parsed dataset metadata.
    pub const fn metadata(&self) -> &Yaml {
        &self.dataset
    }

    /// Returns the resolved dataset root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns the selected split.
    pub const fn split(&self) -> Split {
        self.split
    }

    /// Returns the image paths in deterministic order.
    pub fn image_paths(&self) -> &[PathBuf] {
        &self.images
    }

    /// Returns the target semantic class-map size.
    pub const fn output_size(&self) -> ImageSize {
        self.output_size
    }

    fn sample_semantic(&self, index: usize) -> crate::Result<Sample> {
        let image_path = self.images.get(index).ok_or_else(|| {
            crate::Error::InvalidConfig(format!(
                "sample index {index} is out of bounds for dataset length {}",
                self.images.len()
            ))
        })?;
        let image = read_rgb_image(image_path)?;
        let (input, letterbox) =
            crate::model::letterbox(&image, self.image_size, self.dtype, &self.device)?;
        let mask = read_semantic_mask(&semantic_mask_path_for_image(image_path))?;
        let class_map = semantic_class_map_from_mask(
            &mask,
            (image.width as usize, image.height as usize),
            &letterbox,
            self.output_size,
            &self.device,
        )?;
        Ok(Sample {
            input,
            target: Target::Semantic { class_map },
        })
    }
}

impl Dataset for SemanticDataset {
    fn len(&self) -> usize {
        self.images.len()
    }

    fn sample(&self, index: usize) -> crate::Result<Sample> {
        self.sample_semantic(index)
    }
}

impl ObbDataset {
    fn sample_obb(&self, index: usize) -> crate::Result<Sample> {
        let image_path = self.detect.images.get(index).ok_or_else(|| {
            crate::Error::InvalidConfig(format!(
                "sample index {index} is out of bounds for dataset length {}",
                self.detect.images.len()
            ))
        })?;
        let image = read_rgb_image(image_path)?;
        let source_w = image.width as f32;
        let source_h = image.height as f32;
        let (input, letterbox) = crate::model::letterbox(
            &image,
            self.detect.image_size,
            self.detect.dtype,
            &self.detect.device,
        )?;
        let labels = read_yolo_obb_labels(&label_path_for_image(image_path))?;
        let mut boxes = vec![0f32; self.detect.max_objects * 4];
        let mut class_ids = vec![0u32; self.detect.max_objects];
        let mut valid = vec![0f32; self.detect.max_objects];
        let mut angles = vec![0f32; self.detect.max_objects];
        let mut rboxes = vec![0f32; self.detect.max_objects * 5];

        for (idx, label) in labels.into_iter().take(self.detect.max_objects).enumerate() {
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
            let Some(rbox) = obb_geometry::xywhr_from_points(&model_points) else {
                continue;
            };
            let [x1, y1, x2, y2] = obb_geometry::rbox_xyxy(rbox);
            class_ids[idx] = label.class_id;
            valid[idx] = 1.0;
            boxes[idx * 4] = x1.clamp(0.0, letterbox.model_width as f32);
            boxes[idx * 4 + 1] = y1.clamp(0.0, letterbox.model_height as f32);
            boxes[idx * 4 + 2] = x2.clamp(0.0, letterbox.model_width as f32);
            boxes[idx * 4 + 3] = y2.clamp(0.0, letterbox.model_height as f32);
            if boxes[idx * 4 + 2] <= boxes[idx * 4] || boxes[idx * 4 + 3] <= boxes[idx * 4 + 1] {
                valid[idx] = 0.0;
                continue;
            }
            angles[idx] = rbox[4];
            rboxes[idx * 5..idx * 5 + 5].copy_from_slice(&rbox);
        }

        let boxes_xyxy =
            Tensor::from_vec(boxes, (1, self.detect.max_objects, 4), &self.detect.device)?;
        let class_ids =
            Tensor::from_vec(class_ids, (1, self.detect.max_objects), &self.detect.device)?;
        let valid = Tensor::from_vec(valid, (1, self.detect.max_objects), &self.detect.device)?;
        let detection = DetectionTargets::new(boxes_xyxy, class_ids, valid)?;
        let angles = Tensor::from_vec(angles, (1, self.detect.max_objects), &self.detect.device)?;
        let rboxes_xywhr =
            Tensor::from_vec(rboxes, (1, self.detect.max_objects, 5), &self.detect.device)?;
        Ok(Sample {
            input,
            target: Target::Obb(ObbTargets::new_with_rboxes(
                detection,
                angles,
                rboxes_xywhr,
            )?),
        })
    }
}

impl Dataset for ObbDataset {
    fn len(&self) -> usize {
        self.detect.len()
    }

    fn sample(&self, index: usize) -> crate::Result<Sample> {
        self.sample_obb(index)
    }
}

/// Builds a semantic segmentation dataset for one split from an Ultralytics YAML file.
pub fn semantic_from_file(
    path: impl AsRef<Path>,
    split: Split,
    image_size: ImageSize,
    output_size: ImageSize,
    dtype: DType,
    device: Device,
) -> crate::Result<SemanticDataset> {
    Yaml::semantic_dataset_from_file(path, split, image_size, output_size, dtype, device)
}

/// Builds an oriented bounding-box dataset for one split from an Ultralytics YAML file.
pub fn obb_from_file(
    path: impl AsRef<Path>,
    split: Split,
    image_size: ImageSize,
    dtype: DType,
    device: Device,
    max_objects: usize,
) -> crate::Result<ObbDataset> {
    Yaml::obb_dataset_from_file(path, split, image_size, dtype, device, max_objects)
}
