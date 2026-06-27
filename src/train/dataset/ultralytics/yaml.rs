//! `Yaml` parsing and per-task dataset builders.
//!
//! These entry points parse an Ultralytics-style YAML file and construct the
//! corresponding typed dataset for one split. The `Yaml` struct itself lives
//! in [`super::collate`] (it is shared with the OBB collation path); this
//! module owns the public parsing and builder surface.

use crate::model::ImageSize;

use super::*;

impl Yaml {
    /// Parses a small Ultralytics-style YAML subset without extra dependencies.
    pub fn parse(input: &str) -> crate::Result<Self> {
        let mut dataset = Self {
            path: None,
            train: None,
            val: None,
            test: None,
            names: Vec::new(),
            kpt_shape: None,
        };
        let mut in_names_block = false;
        for raw in input.lines() {
            let line = raw.split('#').next().unwrap_or_default().trim();
            if line.is_empty() {
                continue;
            }
            if in_names_block {
                if let Some(name) = parse_yaml_list_item(line) {
                    dataset.names.push(name);
                    continue;
                }
                in_names_block = false;
            }
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            let key = key.trim();
            let value = value.trim();
            match key {
                "path" => dataset.path = non_empty(value),
                "train" => dataset.train = non_empty(value),
                "val" => dataset.val = non_empty(value),
                "test" => dataset.test = non_empty(value),
                "names" if value.is_empty() => in_names_block = true,
                "names" => dataset.names = parse_inline_names(value),
                "kpt_shape" => dataset.kpt_shape = Some(parse_kpt_shape(value)?),
                _ => {}
            }
        }
        if dataset.names.is_empty() {
            return Err(crate::Error::InvalidConfig(
                "Ultralytics dataset YAML must define names".to_string(),
            ));
        }
        Ok(dataset)
    }

    /// Parses an Ultralytics dataset YAML file from disk.
    pub fn from_file(path: impl AsRef<Path>) -> crate::Result<Self> {
        Self::parse(&std::fs::read_to_string(path)?)
    }

    /// Builds a detection dataset for one split from an Ultralytics YAML file.
    pub fn detection_dataset_from_file(
        path: impl AsRef<Path>,
        split: Split,
        image_size: ImageSize,
        dtype: DType,
        device: Device,
        max_objects: usize,
    ) -> crate::Result<DetectionDataset> {
        let path = path.as_ref();
        let dataset = Self::from_file(path)?;
        let root = dataset_root(path, dataset.path.as_deref());
        DetectionDataset::new(dataset, root, split, image_size, dtype, device, max_objects)
    }

    /// Builds an instance segmentation dataset for one split from an Ultralytics YAML file.
    pub fn segmentation_dataset_from_file(
        path: impl AsRef<Path>,
        split: Split,
        image_size: ImageSize,
        mask_size: ImageSize,
        dtype: DType,
        device: Device,
        max_objects: usize,
    ) -> crate::Result<SegmentationDataset> {
        Self::segmentation_dataset_from_file_with_overlap_mask(
            path,
            split,
            image_size,
            mask_size,
            dtype,
            device,
            max_objects,
            true,
        )
    }

    /// Builds an instance segmentation dataset with explicit overlap-mask encoding.
    #[expect(
        clippy::too_many_arguments,
        reason = "public dataset constructor keeps explicit split, tensor, and mask options"
    )]
    pub fn segmentation_dataset_from_file_with_overlap_mask(
        path: impl AsRef<Path>,
        split: Split,
        image_size: ImageSize,
        mask_size: ImageSize,
        dtype: DType,
        device: Device,
        max_objects: usize,
        overlap_mask: bool,
    ) -> crate::Result<SegmentationDataset> {
        let detect =
            Self::detection_dataset_from_file(path, split, image_size, dtype, device, max_objects)?;
        SegmentationDataset::new_with_overlap_mask(detect, mask_size, overlap_mask)
    }

    /// Builds a pose dataset for one split from an Ultralytics YAML file.
    pub fn pose_dataset_from_file(
        path: impl AsRef<Path>,
        split: Split,
        image_size: ImageSize,
        dtype: DType,
        device: Device,
        max_objects: usize,
    ) -> crate::Result<PoseDataset> {
        let path = path.as_ref();
        let dataset = Self::from_file(path)?;
        let (keypoints_count, keypoint_dims) = dataset.kpt_shape.ok_or_else(|| {
            crate::Error::InvalidConfig(
                "Ultralytics pose dataset YAML must define kpt_shape".to_string(),
            )
        })?;
        let root = dataset_root(path, dataset.path.as_deref());
        let detect =
            DetectionDataset::new(dataset, root, split, image_size, dtype, device, max_objects)?;
        PoseDataset::new(detect, keypoints_count, keypoint_dims)
    }

    /// Builds a semantic segmentation dataset for one split from an Ultralytics YAML file.
    pub fn semantic_dataset_from_file(
        path: impl AsRef<Path>,
        split: Split,
        image_size: ImageSize,
        output_size: ImageSize,
        dtype: DType,
        device: Device,
    ) -> crate::Result<SemanticDataset> {
        let path = path.as_ref();
        let dataset = Self::from_file(path)?;
        let root = dataset_root(path, dataset.path.as_deref());
        SemanticDataset::new(dataset, root, split, image_size, output_size, dtype, device)
    }

    /// Builds an oriented bounding-box dataset for one split from an Ultralytics YAML file.
    pub fn obb_dataset_from_file(
        path: impl AsRef<Path>,
        split: Split,
        image_size: ImageSize,
        dtype: DType,
        device: Device,
        max_objects: usize,
    ) -> crate::Result<ObbDataset> {
        let detect =
            Self::detection_dataset_from_file(path, split, image_size, dtype, device, max_objects)?;
        Ok(ObbDataset { detect })
    }
}
