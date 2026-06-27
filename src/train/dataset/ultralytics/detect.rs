use super::*;

impl DetectionDataset {
    /// Creates a detection dataset from parsed YAML metadata and a resolved root.
    pub fn new(
        dataset: Yaml,
        root: PathBuf,
        split: Split,
        image_size: ImageSize,
        dtype: DType,
        device: Device,
        max_objects: usize,
    ) -> crate::Result<Self> {
        if max_objects == 0 {
            return Err(crate::Error::InvalidConfig(
                "max_objects must be greater than zero".to_string(),
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
            rect_canvas_size: None,
            dtype,
            device,
            max_objects,
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

    /// Uses an Ultralytics-style rectangular validation canvas while retaining
    /// the configured resize size.
    pub fn with_rect_canvas_size(mut self, canvas_size: ImageSize) -> crate::Result<Self> {
        if canvas_size.width < self.image_size.width || canvas_size.height < self.image_size.height
        {
            return Err(crate::Error::InvalidConfig(format!(
                "rect validation canvas {}x{} must be at least image size {}x{}",
                canvas_size.width,
                canvas_size.height,
                self.image_size.width,
                self.image_size.height
            )));
        }
        self.rect_canvas_size = Some(canvas_size);
        Ok(self)
    }

    pub(crate) fn letterbox_image(
        &self,
        image: &crate::Image,
    ) -> crate::Result<(Tensor, crate::model::LetterboxInfo)> {
        match self.rect_canvas_size {
            Some(canvas) => crate::model::letterbox_with_canvas(
                image,
                self.image_size,
                canvas,
                self.dtype,
                &self.device,
            ),
            None => crate::model::letterbox(image, self.image_size, self.dtype, &self.device),
        }
    }

    fn sample_detection(&self, index: usize) -> crate::Result<Sample> {
        let image_path = self.images.get(index).ok_or_else(|| {
            crate::Error::InvalidConfig(format!(
                "sample index {index} is out of bounds for dataset length {}",
                self.images.len()
            ))
        })?;
        let image = read_rgb_image(image_path)?;
        let source_w = image.width as f32;
        let source_h = image.height as f32;
        let (input, letterbox) = self.letterbox_image(&image)?;
        let labels = read_yolo_detection_labels(&label_path_for_image(image_path))?;
        let mut boxes = vec![0f32; self.max_objects * 4];
        let mut class_ids = vec![0u32; self.max_objects];
        let mut valid = vec![0f32; self.max_objects];
        for (idx, label) in labels.into_iter().take(self.max_objects).enumerate() {
            class_ids[idx] = label.class_id;
            valid[idx] = 1.0;
            let x1 = (label.cx - label.w * 0.5) * source_w;
            let y1 = (label.cy - label.h * 0.5) * source_h;
            let x2 = (label.cx + label.w * 0.5) * source_w;
            let y2 = (label.cy + label.h * 0.5) * source_h;
            // Match the official `LetterBox` box transform: apply gain (scale)
            // and offset (pad) but do NOT clamp to the letterbox canvas. Boxes
            // may extend past the padded image, and the assigner / CIoU expect
            // the same unclamped coordinates the model predicts against.
            boxes[idx * 4] = x1 * letterbox.scale + letterbox.pad_x;
            boxes[idx * 4 + 1] = y1 * letterbox.scale + letterbox.pad_y;
            boxes[idx * 4 + 2] = x2 * letterbox.scale + letterbox.pad_x;
            boxes[idx * 4 + 3] = y2 * letterbox.scale + letterbox.pad_y;
            if boxes[idx * 4 + 2] <= boxes[idx * 4] || boxes[idx * 4 + 3] <= boxes[idx * 4 + 1] {
                valid[idx] = 0.0;
            }
        }
        let boxes_xyxy = Tensor::from_vec(boxes, (1, self.max_objects, 4), &self.device)?;
        let class_ids = Tensor::from_vec(class_ids, (1, self.max_objects), &self.device)?;
        let valid = Tensor::from_vec(valid, (1, self.max_objects), &self.device)?;
        let target = Target::Detection(DetectionTargets::new(boxes_xyxy, class_ids, valid)?);
        Ok(Sample { input, target })
    }
}

impl Dataset for DetectionDataset {
    fn len(&self) -> usize {
        self.images.len()
    }

    fn sample(&self, index: usize) -> crate::Result<Sample> {
        self.sample_detection(index)
    }
}

/// Builds a detection dataset for one split from an Ultralytics YAML file.
pub fn from_file(
    path: impl AsRef<Path>,
    split: Split,
    image_size: ImageSize,
    dtype: DType,
    device: Device,
    max_objects: usize,
) -> crate::Result<DetectionDataset> {
    Yaml::detection_dataset_from_file(path, split, image_size, dtype, device, max_objects)
}
