use candle_core::{DType, Device};

use crate::Scale;

use super::base::Base;
use super::image_size::ImageSize;

/// Configuration used when loading a YOLO26 pose model.
#[derive(Debug, Clone)]
pub struct ForPose {
    /// Base detection config.
    pub base: Base,
    /// Number of keypoints per detected instance.
    pub keypoints_count: usize,
    /// Number of values per keypoint.
    pub keypoint_dims: usize,
}

/// Delegates common builder methods to the inner builder.
macro_rules! delegate_base {
    ($($method:ident($($arg:ident: $ty:ty),*)),* $(,)?) => {
        $(pub fn $method(mut self, $($arg: $ty),*) -> Self {
            self.base = self.base.$method($($arg),*);
            self
        })*
    };
}

/// Builder for [`ForPose`].
pub struct Builder {
    base: super::base::Builder,
    keypoints_count: usize,
    keypoint_dims: usize,
}

impl Builder {
    delegate_base!(
        with_scale(scale: Scale),
        with_device(device: Device),
        with_input_size(input_size: usize),
        with_image_size(width: usize, height: usize),
        with_input_shape(image_size: ImageSize),
        with_max_predictions(max_predictions: usize),
        with_rectangular_padding(rectangular_padding: bool),
        with_labels_count(labels_count: usize),
        with_dtype(dtype: DType)
    );

    pub fn with_keypoints_count(mut self, keypoints_count: usize) -> Self {
        self.keypoints_count = keypoints_count;
        self
    }

    pub fn with_keypoint_dims(mut self, keypoint_dims: usize) -> Self {
        self.keypoint_dims = keypoint_dims;
        self
    }

    pub fn build(self) -> ForPose {
        ForPose {
            base: self.base.build(),
            keypoints_count: self.keypoints_count,
            keypoint_dims: self.keypoint_dims,
        }
    }
}

impl ForPose {
    pub(crate) fn builder() -> Builder {
        Builder {
            base: Base::raw_builder(super::super::MODEL_INPUT_SIZE, 1),
            keypoints_count: 17,
            keypoint_dims: 3,
        }
    }
}

impl Default for ForPose {
    fn default() -> Self {
        Self::builder().build()
    }
}
