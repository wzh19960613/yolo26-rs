use candle_core::{DType, Tensor};

use super::{
    DetectionTargets, ObbTargets, PoseTargets, SEMANTIC_IGNORE_CLASS_ID, Sample,
    SegmentationTargets, Target,
};

/// Class selection and single-class remapping for training targets.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ClassFilter {
    /// Whether every retained class id should be remapped to zero.
    pub single_class: bool,
    /// Optional class ids to retain for detection-style and semantic targets.
    pub classes: Option<Vec<u32>>,
}

impl ClassFilter {
    /// Creates a filter from Ultralytics-style `single_cls` and `classes` options.
    pub fn new(single_class: bool, classes: Option<Vec<u32>>) -> crate::Result<Self> {
        Ok(Self {
            single_class,
            classes: classes.map(normalize_classes).transpose()?,
        })
    }

    /// Creates a filter that keeps all classes unchanged.
    pub const fn none() -> Self {
        Self {
            single_class: false,
            classes: None,
        }
    }

    /// Creates a filter that remaps every target class to zero.
    pub const fn single_class() -> Self {
        Self {
            single_class: true,
            classes: None,
        }
    }

    /// Creates a filter that retains the supplied class ids.
    pub fn classes(classes: impl Into<Vec<u32>>) -> crate::Result<Self> {
        let classes = normalize_classes(classes.into())?;
        Ok(Self {
            single_class: false,
            classes: Some(classes),
        })
    }

    /// Returns whether this filter changes samples.
    pub fn is_enabled(&self) -> bool {
        self.single_class || self.classes.is_some()
    }

    /// Filters a training sample, returning `None` when a class-only sample is excluded.
    pub fn filter_sample(&self, sample: Sample) -> crate::Result<Option<Sample>> {
        if !self.is_enabled() {
            return Ok(Some(sample));
        }
        if !self.keep_classification_target(&sample.target)? {
            return Ok(None);
        }
        Ok(Some(Sample {
            input: sample.input,
            target: self.apply_target(sample.target)?,
        }))
    }

    /// Applies the filter to a training target.
    pub fn apply_target(&self, target: Target) -> crate::Result<Target> {
        Ok(match target {
            Target::Classification { class_ids } => Target::Classification {
                class_ids: self.filter_class_ids_1d(class_ids)?,
            },
            Target::Detection(targets) => Target::Detection(self.filter_detection(targets)?),
            Target::Segmentation(targets) => {
                Target::Segmentation(SegmentationTargets::new_with_mask_encoding(
                    self.filter_detection(targets.detection)?,
                    targets.masks,
                    targets.mask_encoding,
                )?)
            }
            Target::Pose(targets) => Target::Pose(PoseTargets::new(
                self.filter_detection(targets.detection)?,
                targets.keypoints,
                targets.visibility,
            )?),
            Target::Obb(targets) => Target::Obb(ObbTargets::new(
                self.filter_detection(targets.detection)?,
                targets.angles,
            )?),
            Target::Semantic { class_map } => Target::Semantic {
                class_map: self.filter_class_map(class_map)?,
            },
            Target::Dense => Target::Dense,
        })
    }

    fn filter_detection(&self, targets: DetectionTargets) -> crate::Result<DetectionTargets> {
        let device = targets.class_ids.device().clone();
        let valid_dtype = targets.valid.dtype();
        let dims = targets.class_ids.dims().to_vec();
        let class_rows = targets.class_ids.to_dtype(DType::U32)?.to_vec2::<u32>()?;
        let valid_rows = targets.valid.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let mut class_ids = Vec::with_capacity(dims.iter().product());
        let mut valid = Vec::with_capacity(class_ids.capacity());
        for (classes, valids) in class_rows.iter().zip(valid_rows.iter()) {
            for (&class_id, &is_valid) in classes.iter().zip(valids.iter()) {
                let keep = is_valid > 0.0 && self.keep_class(class_id);
                class_ids.push(if self.single_class && keep {
                    0
                } else {
                    class_id
                });
                valid.push(if keep { is_valid } else { 0.0 });
            }
        }
        DetectionTargets::new(
            targets.boxes_xyxy,
            Tensor::from_vec(class_ids, dims.clone(), &device)?,
            Tensor::from_vec(valid, dims, &device)?.to_dtype(valid_dtype)?,
        )
    }

    fn filter_class_map(&self, class_map: Tensor) -> crate::Result<Tensor> {
        let device = class_map.device().clone();
        let dims = class_map.dims().to_vec();
        let classes = class_map
            .to_dtype(DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let mapped = classes
            .into_iter()
            .map(|class_id| {
                if class_id == SEMANTIC_IGNORE_CLASS_ID || !self.keep_class(class_id) {
                    SEMANTIC_IGNORE_CLASS_ID
                } else if self.single_class {
                    0
                } else {
                    class_id
                }
            })
            .collect::<Vec<_>>();
        Ok(Tensor::from_vec(mapped, dims, &device)?)
    }

    fn filter_class_ids_1d(&self, class_ids: Tensor) -> crate::Result<Tensor> {
        if self.classes.is_some() && !self.keep_class_ids(&class_ids)? {
            return Err(crate::Error::InvalidConfig(
                "classification classes filtering requires sample-level filtering".to_string(),
            ));
        }
        if !self.single_class {
            return Ok(class_ids);
        }
        Ok(Tensor::zeros(
            class_ids.dims(),
            DType::U32,
            class_ids.device(),
        )?)
    }

    fn keep_classification_target(&self, target: &Target) -> crate::Result<bool> {
        match target {
            Target::Classification { class_ids } if self.classes.is_some() => {
                self.keep_class_ids(class_ids)
            }
            _ => Ok(true),
        }
    }

    fn keep_class_ids(&self, class_ids: &Tensor) -> crate::Result<bool> {
        let ids = class_ids
            .to_dtype(DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        Ok(ids.into_iter().all(|class_id| self.keep_class(class_id)))
    }

    fn keep_class(&self, class_id: u32) -> bool {
        self.classes
            .as_ref()
            .is_none_or(|classes| classes.binary_search(&class_id).is_ok())
    }
}

fn normalize_classes(mut classes: Vec<u32>) -> crate::Result<Vec<u32>> {
    classes.sort_unstable();
    classes.dedup();
    if classes.is_empty() {
        return Err(crate::Error::InvalidConfig(
            "train class filter requires at least one class".to_string(),
        ));
    }
    Ok(classes)
}
