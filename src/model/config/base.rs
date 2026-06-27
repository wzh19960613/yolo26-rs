use candle_core::{DType, Device};

use crate::Scale;
use crate::model::DtypeRequest;

use super::image_size::ImageSize;

/// Configuration used when loading a YOLO26 detection / segmentation / OBB / semantic model.
#[derive(Debug, Clone)]
pub struct Base {
    /// Model scale to instantiate.
    pub scale: Scale,
    /// Compute device used by Candle.
    pub device: Device,
    /// Input tensor size used by preprocessing.
    pub image_size: ImageSize,
    /// Maximum number of predictions retained by the model head.
    pub max_predictions: usize,
    /// Whether preprocessing keeps a rectangular input size instead of forcing a square canvas.
    pub rectangular_padding: bool,
    /// Count of class labels.
    pub labels_count: usize,
    /// Compute dtype request for weights and inference. Defaults to
    /// [`DtypeRequest::Auto`], which infers the dtype from the checkpoint; pass
    /// `with_dtype` to force a specific precision.
    pub dtype: DtypeRequest,
}

/// Builder for [`Base`].
pub struct Builder {
    base: Base,
}

impl Builder {
    pub fn with_scale(mut self, scale: Scale) -> Self {
        self.base.scale = scale;
        self
    }

    pub fn with_device(mut self, device: Device) -> Self {
        self.base.device = device;
        self
    }

    pub fn with_input_size(mut self, input_size: usize) -> Self {
        self.base.image_size = ImageSize::square(input_size).snapped();
        self
    }

    pub fn with_image_size(mut self, width: usize, height: usize) -> Self {
        self.base.image_size = ImageSize::new(width, height).snapped();
        self
    }

    pub fn with_input_shape(mut self, image_size: ImageSize) -> Self {
        self.base.image_size = image_size.snapped();
        self
    }

    pub fn with_max_predictions(mut self, max_predictions: usize) -> Self {
        self.base.max_predictions = max_predictions;
        self
    }

    pub fn with_rectangular_padding(mut self, rectangular_padding: bool) -> Self {
        self.base.rectangular_padding = rectangular_padding;
        self
    }

    pub fn with_labels_count(mut self, labels_count: usize) -> Self {
        self.base.labels_count = labels_count;
        self
    }

    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.base.dtype = DtypeRequest::Fixed(dtype);
        self
    }

    pub fn build(self) -> Base {
        self.base
    }
}

impl Base {
    pub(crate) fn validate(&self) -> crate::Result<()> {
        let s = &self.image_size;
        if s.width == 0
            || s.height == 0
            || !s.width.is_multiple_of(32)
            || !s.height.is_multiple_of(32)
        {
            return Err(crate::Error::InvalidConfig(
                "YOLO26 image dimensions must be positive multiples of 32".to_string(),
            ));
        }
        if self.max_predictions == 0 {
            return Err(crate::Error::InvalidConfig(
                "YOLO26 max_predictions must not be 0".to_string(),
            ));
        }
        if self.labels_count == 0 {
            return Err(crate::Error::InvalidConfig(
                "YOLO26 labels_count must not be 0".to_string(),
            ));
        }
        Ok(())
    }

    pub(crate) fn effective_image_size(&self) -> ImageSize {
        if self.rectangular_padding {
            self.image_size
        } else {
            self.image_size.square_from_max()
        }
    }

    fn builder(input_size: usize, labels_count: usize) -> Builder {
        Builder {
            base: Base {
                scale: Scale::N,
                device: Device::Cpu,
                image_size: ImageSize::square(input_size),
                max_predictions: 300,
                rectangular_padding: true,
                labels_count,
                dtype: DtypeRequest::Auto,
            },
        }
    }

    pub(crate) fn detect_builder() -> Builder {
        Self::builder(
            super::super::MODEL_INPUT_SIZE,
            default_labels_count(DefaultLabels::Coco),
        )
    }

    pub(crate) fn segment_builder() -> Builder {
        Self::builder(
            super::super::MODEL_INPUT_SIZE,
            default_labels_count(DefaultLabels::Coco),
        )
    }

    pub(crate) fn obb_builder() -> Builder {
        Self::builder(
            super::super::MODEL_INPUT_SIZE,
            default_labels_count(DefaultLabels::Dota),
        )
    }

    pub(crate) fn semantic_builder() -> Builder {
        Self::builder(
            super::super::MODEL_INPUT_SIZE,
            default_labels_count(DefaultLabels::Cityscapes),
        )
    }

    pub(crate) fn classify_builder() -> Builder {
        Self::builder(224, default_labels_count(DefaultLabels::Imagenet))
    }

    pub(crate) fn raw_builder(input_size: usize, labels_count: usize) -> Builder {
        Self::builder(input_size, labels_count)
    }
}

impl Default for Base {
    fn default() -> Self {
        Self::detect_builder().build()
    }
}

/// Built-in dataset whose default class count a task builder picks.
enum DefaultLabels {
    Coco,
    Dota,
    Cityscapes,
    Imagenet,
}

/// Returns the built-in dataset's class count.
///
/// With the `default_labels` feature the count comes from the matching label
/// table (`crate::default_labels::*`); without it, the dataset's canonical
/// class count is hard-coded so the task builders still compile and produce a
/// usable default `labels_count`.
const fn default_labels_count(dataset: DefaultLabels) -> usize {
    #[cfg(feature = "default_labels")]
    {
        match dataset {
            DefaultLabels::Coco => crate::default_labels::COCO.len(),
            DefaultLabels::Dota => crate::default_labels::DOTA.len(),
            DefaultLabels::Cityscapes => crate::default_labels::CITYSCAPES.len(),
            DefaultLabels::Imagenet => crate::default_labels::IMAGENET.len(),
        }
    }
    #[cfg(not(feature = "default_labels"))]
    {
        match dataset {
            DefaultLabels::Coco => 80,
            DefaultLabels::Dota => 15,
            DefaultLabels::Cityscapes => 19,
            DefaultLabels::Imagenet => 1000,
        }
    }
}
