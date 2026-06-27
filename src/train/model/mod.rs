//! Trainable task model: the per-task network wrapped in a Candle `VarMap`.

pub(crate) mod class_names;
pub(crate) mod forward;
pub(crate) mod loading;
pub(crate) mod methods;
pub(crate) mod save;

pub(crate) use crate::train::exports::*;
use candle_core::{DType, Device};
use candle_nn::VarMap;
pub(crate) use class_names::validate_class_names;

use crate::Scale;

/// YOLO task supported by the training facade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]

pub enum Task {
    /// Object detection.
    Detect,
    /// Image classification.
    Classify,
    /// Instance segmentation.
    Segment,
    /// Pose/keypoint estimation.
    Pose,
    /// Semantic segmentation.
    Semantic,
    /// Oriented bounding-box detection.
    Obb,
}

impl Task {
    /// Returns the lowercase task identifier used in `.pt` template names.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Detect => "detect",
            Self::Classify => "classify",
            Self::Segment => "segment",
            Self::Pose => "pose",
            Self::Semantic => "semantic",
            Self::Obb => "obb",
        }
    }
}

/// Model configuration used to instantiate a trainable task network.
#[derive(Debug, Clone)]

pub enum ModelConfig {
    /// Detection config.
    Detect(crate::detect::Config),
    /// Classification config.
    Classify(crate::classify::Config),
    /// Instance segmentation config.
    Segment(crate::segment::Config),
    /// Pose config.
    Pose(crate::pose::Config),
    /// Semantic segmentation config.
    Semantic(crate::semantic::Config),
    /// Oriented bounding-box config.
    Obb(crate::obb::Config),
}

impl ModelConfig {
    /// Returns the task represented by this config.
    pub const fn task(&self) -> Task {
        match self {
            Self::Detect(_) => Task::Detect,
            Self::Classify(_) => Task::Classify,
            Self::Segment(_) => Task::Segment,
            Self::Pose(_) => Task::Pose,
            Self::Semantic(_) => Task::Semantic,
            Self::Obb(_) => Task::Obb,
        }
    }

    pub(crate) fn dtype(&self) -> DType {
        // Training builds the VarMap without a checkpoint in hand, so an `Auto`
        // request resolves to F32 here. Callers that want F16 training must
        // pass `with_dtype(DType::F16)` explicitly; loading a checkpoint later
        // (from_safetensors) keeps its native dtype.
        match self {
            Self::Detect(config)
            | Self::Classify(config)
            | Self::Segment(config)
            | Self::Semantic(config)
            | Self::Obb(config) => config.dtype.resolve_or_f32(),
            Self::Pose(config) => config.base.dtype.resolve_or_f32(),
        }
    }

    pub(crate) fn device(&self) -> Device {
        match self {
            Self::Detect(config)
            | Self::Classify(config)
            | Self::Segment(config)
            | Self::Semantic(config)
            | Self::Obb(config) => config.device.clone(),
            Self::Pose(config) => config.base.device.clone(),
        }
    }

    /// Returns the model scale declared in the config.
    pub(crate) fn scale(&self) -> Scale {
        match self {
            Self::Detect(config)
            | Self::Classify(config)
            | Self::Segment(config)
            | Self::Semantic(config)
            | Self::Obb(config) => config.scale,
            Self::Pose(config) => config.base.scale,
        }
    }

    /// Returns the class count declared in this config.
    pub fn labels_count(&self) -> usize {
        match self {
            Self::Detect(config)
            | Self::Classify(config)
            | Self::Segment(config)
            | Self::Semantic(config)
            | Self::Obb(config) => config.labels_count,
            Self::Pose(config) => config.base.labels_count,
        }
    }

    pub(crate) fn validate(&self) -> crate::Result<()> {
        match self {
            Self::Detect(config)
            | Self::Classify(config)
            | Self::Segment(config)
            | Self::Semantic(config)
            | Self::Obb(config) => config.validate(),
            Self::Pose(config) => {
                if config.keypoints_count == 0 || config.keypoint_dims < 2 {
                    return Err(crate::Error::InvalidConfig(
                        "pose training requires keypoints_count > 0 and keypoint_dims >= 2"
                            .to_string(),
                    ));
                }
                config.base.validate()
            }
        }
    }
}

pub(crate) enum TrainableNetwork {
    Detect(Box<crate::detect::network::Network>),
    Classify(Box<crate::classify::network::Network>),
    Segment(Box<crate::segment::network::Network>),
    Pose(Box<crate::pose::network::Network>),
    Semantic(Box<crate::semantic::network::Network>),
    Obb(Box<crate::obb::network::Network>),
}

/// Trainable YOLO26 model backed by a Candle `VarMap`.
pub struct Model {
    pub(crate) varmap: VarMap,
    pub(crate) network: TrainableNetwork,
    pub(crate) task: Task,
    pub(crate) dtype: DType,
    pub(crate) device: Device,
    pub(crate) scale: Scale,
    pub(crate) labels_count: usize,
    pub(crate) class_names: Option<Vec<String>>,
}
