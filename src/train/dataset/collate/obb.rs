use super::*;

/// Collates single-image OBB samples into one batch.
pub fn collate_obb_samples(samples: &[Sample]) -> crate::Result<Sample> {
    let first = samples
        .first()
        .ok_or_else(|| crate::Error::InvalidConfig("cannot collate an empty batch".to_string()))?;
    let first_input_dims = first.input.dims();
    if first_input_dims.len() != 4 || first_input_dims[0] != 1 {
        return Err(crate::Error::InvalidTensor(format!(
            "obb sample inputs must have shape [1, C, H, W], got {first_input_dims:?}"
        )));
    }

    let mut input_refs = Vec::with_capacity(samples.len());
    let mut box_refs = Vec::with_capacity(samples.len());
    let mut class_refs = Vec::with_capacity(samples.len());
    let mut valid_refs = Vec::with_capacity(samples.len());
    let mut angle_refs = Vec::with_capacity(samples.len());
    let mut rbox_refs = Vec::with_capacity(samples.len());
    let mut target_shape = None;
    for sample in samples {
        if sample.input.dims() != first_input_dims {
            return Err(crate::Error::InvalidTensor(format!(
                "all obb sample inputs must share shape {:?}, got {:?}",
                first_input_dims,
                sample.input.dims()
            )));
        }
        let Target::Obb(targets) = &sample.target else {
            return Err(crate::Error::InvalidTensor(
                "collate_obb_samples only accepts obb targets".to_string(),
            ));
        };
        let shape = targets.detection.boxes_xyxy.dims().to_vec();
        if shape.len() != 3 || shape[0] != 1 || shape[2] != 4 {
            return Err(crate::Error::InvalidTensor(format!(
                "obb detection boxes must have shape [1, objects, 4], got {shape:?}"
            )));
        }
        if let Some(expected) = &target_shape {
            if expected != &shape {
                return Err(crate::Error::InvalidTensor(format!(
                    "all obb detection targets must share shape {expected:?}, got {shape:?}"
                )));
            }
        } else {
            target_shape = Some(shape);
        }
        input_refs.push(&sample.input);
        box_refs.push(&targets.detection.boxes_xyxy);
        class_refs.push(&targets.detection.class_ids);
        valid_refs.push(&targets.detection.valid);
        angle_refs.push(&targets.angles);
        rbox_refs.push(&targets.rboxes_xywhr);
    }

    let input = Tensor::cat(&input_refs, 0)?;
    let boxes_xyxy = Tensor::cat(&box_refs, 0)?;
    let class_ids = Tensor::cat(&class_refs, 0)?;
    let valid = Tensor::cat(&valid_refs, 0)?;
    let angles = Tensor::cat(&angle_refs, 0)?;
    let rboxes_xywhr = Tensor::cat(&rbox_refs, 0)?;
    let detection = DetectionTargets::new(boxes_xyxy, class_ids, valid)?;
    Ok(Sample {
        input,
        target: Target::Obb(ObbTargets::new_with_rboxes(
            detection,
            angles,
            rboxes_xywhr,
        )?),
    })
}

/// Dataset abstraction for Rust-native training.
pub trait Dataset {
    /// Number of samples in the dataset.
    fn len(&self) -> usize;

    /// Returns true when the dataset contains no samples.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Loads one sample by index.
    fn sample(&self, index: usize) -> crate::Result<Sample>;
}

/// Parsed subset of an Ultralytics dataset YAML file.
#[derive(Debug, Clone, PartialEq, Eq)]

pub struct Yaml {
    /// Optional dataset root path.
    pub path: Option<String>,
    /// Training image path or glob.
    pub train: Option<String>,
    /// Validation image path or glob.
    pub val: Option<String>,
    /// Test image path or glob.
    pub test: Option<String>,
    /// Class names.
    pub names: Vec<String>,
    /// Optional pose keypoint shape `(keypoints_count, keypoint_dims)`.
    pub kpt_shape: Option<(usize, usize)>,
}

/// Classification dataset backed by Ultralytics split/class directory layout.
#[derive(Debug, Clone)]

pub struct ClassificationDataset {
    pub(crate) root: PathBuf,
    pub(crate) split: Split,
    pub(crate) classes: Vec<String>,
    pub(crate) samples: Vec<(PathBuf, u32)>,
    pub(crate) image_size: ImageSize,
    pub(crate) dtype: DType,
    pub(crate) device: Device,
}

/// Dataset split selected from an Ultralytics YAML file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]

pub enum Split {
    /// The `train` split.
    Train,
    /// The `val` split.
    Val,
    /// The `test` split.
    Test,
}

impl Split {
    pub(crate) fn value(self, dataset: &Yaml) -> Option<&str> {
        match self {
            Self::Train => dataset.train.as_deref(),
            Self::Val => dataset.val.as_deref(),
            Self::Test => dataset.test.as_deref(),
        }
    }

    pub(crate) fn as_dir_name(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Val => "val",
            Self::Test => "test",
        }
    }
}

/// Detection dataset backed by Ultralytics image directories and YOLO txt labels.
#[derive(Debug, Clone)]

pub struct DetectionDataset {
    pub(crate) dataset: Yaml,
    pub(crate) root: PathBuf,
    pub(crate) split: Split,
    pub(crate) images: Vec<PathBuf>,
    pub(crate) image_size: ImageSize,
    pub(crate) rect_canvas_size: Option<ImageSize>,
    pub(crate) dtype: DType,
    pub(crate) device: Device,
    pub(crate) max_objects: usize,
}

/// Segmentation dataset backed by Ultralytics image directories and polygon labels.
#[derive(Debug, Clone)]

pub struct SegmentationDataset {
    pub(crate) detect: DetectionDataset,
    pub(crate) mask_size: ImageSize,
    pub(crate) overlap_mask: bool,
}

/// Pose dataset backed by Ultralytics image directories and keypoint labels.
#[derive(Debug, Clone)]

pub struct PoseDataset {
    pub(crate) detect: DetectionDataset,
    pub(crate) keypoints_count: usize,
    pub(crate) keypoint_dims: usize,
}
