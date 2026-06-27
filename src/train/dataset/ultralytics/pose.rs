use super::*;

impl PoseDataset {
    /// Creates a pose dataset from a detection dataset and keypoint shape.
    pub fn new(
        detect: DetectionDataset,
        keypoints_count: usize,
        keypoint_dims: usize,
    ) -> crate::Result<Self> {
        if keypoints_count == 0 || !matches!(keypoint_dims, 2 | 3) {
            return Err(crate::Error::InvalidConfig(
                "pose keypoint shape must be [keypoints, 2|3]".to_string(),
            ));
        }
        Ok(Self {
            detect,
            keypoints_count,
            keypoint_dims,
        })
    }

    /// Returns the keypoint shape `(keypoints_count, keypoint_dims)`.
    pub const fn keypoint_shape(&self) -> (usize, usize) {
        (self.keypoints_count, self.keypoint_dims)
    }

    fn sample_pose(&self, index: usize) -> crate::Result<Sample> {
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
        let labels = read_yolo_pose_labels(
            &label_path_for_image(image_path),
            self.keypoints_count,
            self.keypoint_dims,
        )?;
        let mut boxes = vec![0f32; self.detect.max_objects * 4];
        let mut class_ids = vec![0u32; self.detect.max_objects];
        let mut valid = vec![0f32; self.detect.max_objects];
        let mut keypoints = vec![0f32; self.detect.max_objects * self.keypoints_count * 2];
        let mut visibility = vec![0f32; self.detect.max_objects * self.keypoints_count];

        for (idx, label) in labels.into_iter().take(self.detect.max_objects).enumerate() {
            class_ids[idx] = label.class_id;
            valid[idx] = 1.0;
            let x1 = (label.cx - label.w * 0.5) * source_w;
            let y1 = (label.cy - label.h * 0.5) * source_h;
            let x2 = (label.cx + label.w * 0.5) * source_w;
            let y2 = (label.cy + label.h * 0.5) * source_h;
            boxes[idx * 4] =
                (x1 * letterbox.scale + letterbox.pad_x).clamp(0.0, letterbox.model_width as f32);
            boxes[idx * 4 + 1] =
                (y1 * letterbox.scale + letterbox.pad_y).clamp(0.0, letterbox.model_height as f32);
            boxes[idx * 4 + 2] =
                (x2 * letterbox.scale + letterbox.pad_x).clamp(0.0, letterbox.model_width as f32);
            boxes[idx * 4 + 3] =
                (y2 * letterbox.scale + letterbox.pad_y).clamp(0.0, letterbox.model_height as f32);
            if boxes[idx * 4 + 2] <= boxes[idx * 4] || boxes[idx * 4 + 3] <= boxes[idx * 4 + 1] {
                valid[idx] = 0.0;
                continue;
            }
            for (keypoint_idx, keypoint) in label.keypoints.iter().enumerate() {
                let base = (idx * self.keypoints_count + keypoint_idx) * 2;
                keypoints[base] = keypoint.x * source_w * letterbox.scale + letterbox.pad_x;
                keypoints[base + 1] = keypoint.y * source_h * letterbox.scale + letterbox.pad_y;
                visibility[idx * self.keypoints_count + keypoint_idx] = keypoint.visibility;
            }
        }

        let boxes_xyxy =
            Tensor::from_vec(boxes, (1, self.detect.max_objects, 4), &self.detect.device)?;
        let class_ids =
            Tensor::from_vec(class_ids, (1, self.detect.max_objects), &self.detect.device)?;
        let valid = Tensor::from_vec(valid, (1, self.detect.max_objects), &self.detect.device)?;
        let detection = DetectionTargets::new(boxes_xyxy, class_ids, valid)?;
        let keypoints = Tensor::from_vec(
            keypoints,
            (1, self.detect.max_objects, self.keypoints_count, 2),
            &self.detect.device,
        )?;
        let visibility = Tensor::from_vec(
            visibility,
            (1, self.detect.max_objects, self.keypoints_count),
            &self.detect.device,
        )?;
        Ok(Sample {
            input,
            target: Target::Pose(PoseTargets::new(detection, keypoints, visibility)?),
        })
    }
}

impl Dataset for PoseDataset {
    fn len(&self) -> usize {
        self.detect.len()
    }

    fn sample(&self, index: usize) -> crate::Result<Sample> {
        self.sample_pose(index)
    }
}

/// Builds a pose dataset for one split from an Ultralytics YAML file.
pub fn from_file(
    path: impl AsRef<Path>,
    split: Split,
    image_size: ImageSize,
    dtype: DType,
    device: Device,
    max_objects: usize,
) -> crate::Result<PoseDataset> {
    Yaml::pose_dataset_from_file(path, split, image_size, dtype, device, max_objects)
}
