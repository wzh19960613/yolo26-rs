//! YOLO26 instance segmentation model and post-processing entry points.
//!
//! Raw-output decoding is split across [`postprocess`] (anchor decoding and
//! per-detection mask selection), [`mask_decode`] (prototype matmul and mask
//! tensor assembly) and [`mask_crop`] (bounding-box constrained mask cropping).

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::model::{ImageSize, letterbox};
use crate::{FilterOption, MaskOption, Result, image::Image};

use super::{Config, Prediction, network};

mod mask_crop;
mod mask_decode;
mod postprocess;

#[cfg(any(feature = "yoloe-text", feature = "yoloe-visual", feature = "yoloe-pf"))]
pub(crate) use postprocess::postprocess_segmentation;

/// YOLO26 instance segmentation model.
pub struct Model {
    net: network::Network,
    device: Device,
    dtype: DType,
    image_size: ImageSize,
}

impl Model {
    /// Loads a YOLO26 segmentation model from SafeTensors bytes.
    pub fn from_safetensors(weights: Vec<u8>, config: Config) -> Result<Self> {
        let image_size = config.effective_image_size();
        config.validate()?;
        let dtype = config.dtype.resolve_safetensors(&weights, &config.device)?;
        let vb = VarBuilder::from_buffered_safetensors(weights, dtype, &config.device)?;
        let net = network::load(vb.pp("model"), &config)?;
        Ok(Self {
            net,
            device: config.device,
            dtype,
            image_size,
        })
    }

    /// Loads a segmentation model from a `.safetensors` file.
    pub fn from_safetensors_file(
        path: impl AsRef<std::path::Path>,
        config: Config,
    ) -> Result<Self> {
        Self::from_safetensors(std::fs::read(path.as_ref())?, config)
    }

    /// Loads a YOLO26 segmentation model directly from an official `.pt`
    /// checkpoint.
    #[cfg(feature = "pt")]
    pub fn from_pt_file(path: impl AsRef<std::path::Path>, config: Config) -> Result<Self> {
        let image_size = config.effective_image_size();
        config.validate()?;
        let dtype = config.dtype.resolve_pt(&path, &config.device)?;
        let vb = crate::pt_loader::var_builder_from_pt_file(path, dtype, &config.device)?;
        let net = network::load(vb.pp("model"), &config)?;
        Ok(Self {
            net,
            device: config.device,
            dtype,
            image_size,
        })
    }

    /// Loads a segmentation model from a `.pt` or `.safetensors` checkpoint,
    /// inferring `scale`, `labels_count`, `device` and `dtype` automatically.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_file_with(path, Config::default())
    }

    /// Loads a segmentation model from a `.pt` or `.safetensors` checkpoint,
    /// inferring `scale`/`labels_count` while honoring config overrides.
    pub fn from_file_with(path: impl AsRef<std::path::Path>, mut config: Config) -> Result<Self> {
        let path = path.as_ref();
        let shapes = crate::model::checkpoint_shapes(path)?;
        config.scale = crate::model::infer_scale_from_shapes(&shapes)?;
        config.labels_count = crate::model::infer_labels_count_from_shapes(
            crate::model::InferredTask::Detect,
            &shapes,
        )?;
        if crate::model::is_pt_path(path) {
            #[cfg(feature = "pt")]
            {
                Self::from_pt_file(path, config)
            }
            #[cfg(not(feature = "pt"))]
            {
                Err(crate::Error::InvalidConfig(
                    "loading .pt requires the 'pt' feature".to_string(),
                ))
            }
        } else {
            Self::from_safetensors_file(path, config)
        }
    }

    /// Loads a segmentation model from an in-memory `.pt` or `.safetensors` byte
    /// buffer, inferring `scale`/`labels_count` automatically.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes_with(bytes, Config::default())
    }

    /// Loads a segmentation model from an in-memory byte buffer, inferring
    /// `scale`/`labels_count` while honoring config overrides.
    pub fn from_bytes_with(bytes: &[u8], mut config: Config) -> Result<Self> {
        let shapes = crate::model::checkpoint_shapes_from_bytes(bytes)?;
        config.scale = crate::model::infer_scale_from_shapes(&shapes)?;
        config.labels_count = crate::model::infer_labels_count_from_shapes(
            crate::model::InferredTask::Detect,
            &shapes,
        )?;
        if crate::model::is_pt_bytes(bytes) {
            #[cfg(feature = "pt")]
            {
                Self::from_pt_bytes(bytes, config)
            }
            #[cfg(not(feature = "pt"))]
            {
                Err(crate::Error::InvalidConfig(
                    "loading .pt requires the 'pt' feature".to_string(),
                ))
            }
        } else {
            Self::from_safetensors(bytes.to_vec(), config)
        }
    }

    /// Loads a YOLO26 segmentation model directly from an in-memory `.pt`
    /// checkpoint (no filesystem). The `pt` feature is required.
    #[cfg(feature = "pt")]
    pub fn from_pt_bytes(bytes: &[u8], config: Config) -> Result<Self> {
        let image_size = config.effective_image_size();
        config.validate()?;
        let dtype = config.dtype.resolve_pt_bytes(bytes, &config.device)?;
        let vb =
            crate::pt_loader::var_builder_from_pt_bytes(bytes, dtype, &config.device)?.pp("model");
        let net = network::load(vb, &config)?;
        Ok(Self {
            net,
            device: config.device,
            dtype,
            image_size,
        })
    }

    /// Returns the resolved compute dtype used by this model.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Runs the raw network forward pass for a preprocessed tensor.
    pub fn forward_tensor(&self, input: &Tensor) -> Result<Tensor> {
        Ok(self.net.forward(input)?.0)
    }

    /// Runs instance segmentation for one image.
    pub fn predict(
        &self,
        image: &Image,
        filter: &FilterOption,
        mask: &MaskOption,
    ) -> Result<Vec<Prediction>> {
        let (input, letterbox_info) = letterbox(image, self.image_size, self.dtype, &self.device)?;
        let (detections, proto) = self.net.forward(&input)?;
        postprocess::postprocess_segmentation(
            &detections,
            &proto,
            &letterbox_info,
            (image.width, image.height),
            filter,
            mask,
        )
    }
}
