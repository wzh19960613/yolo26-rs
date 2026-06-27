use super::*;

impl ClassificationDataset {
    /// Creates a classification dataset from an Ultralytics directory root.
    pub fn from_dir(
        root: impl AsRef<Path>,
        split: Split,
        image_size: ImageSize,
        dtype: DType,
        device: Device,
    ) -> crate::Result<Self> {
        let root = root.as_ref().to_path_buf();
        let split_dir = root.join(split.as_dir_name());
        let classes = classification_class_names_in_split(&split_dir)?;
        let mut samples = Vec::new();
        for (class_id, class_name) in classes.iter().enumerate() {
            let class_dir = split_dir.join(class_name);
            for image_path in collect_images_in_dir(&class_dir)? {
                samples.push((image_path, class_id as u32));
            }
        }
        samples.sort_by(|a, b| a.0.cmp(&b.0));
        if samples.is_empty() {
            return Err(crate::Error::InvalidConfig(format!(
                "classification split {} contains no supported images",
                split_dir.display()
            )));
        }
        Ok(Self {
            root,
            split,
            classes,
            samples,
            image_size,
            dtype,
            device,
        })
    }

    /// Returns class names sorted by class id for a split.
    pub fn class_names(root: impl AsRef<Path>, split: Split) -> crate::Result<Vec<String>> {
        classification_class_names_in_split(&root.as_ref().join(split.as_dir_name()))
    }

    /// Returns class names sorted by class id.
    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    /// Returns the selected split.
    pub const fn split(&self) -> Split {
        self.split
    }

    /// Returns the dataset root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn sample_classification(&self, index: usize) -> crate::Result<Sample> {
        let (image_path, class_id) = self.samples.get(index).ok_or_else(|| {
            crate::Error::InvalidConfig(format!(
                "sample index {index} is out of bounds for dataset length {}",
                self.samples.len()
            ))
        })?;
        let image = read_rgb_image(image_path)?;
        let input = classify_train_preprocess(&image, self.image_size, self.dtype, &self.device)?;
        let class_ids = Tensor::from_vec(vec![*class_id], (1,), &self.device)?;
        Ok(Sample {
            input,
            target: Target::Classification { class_ids },
        })
    }
}

impl Dataset for ClassificationDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn sample(&self, index: usize) -> crate::Result<Sample> {
        self.sample_classification(index)
    }
}
