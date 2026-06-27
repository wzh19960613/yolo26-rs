use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::MaskOption;
use crate::Result;
use crate::image::Image;
use crate::model::{ImageSize, LetterboxInfo, letterbox};

use super::{Config, Prediction, network};

/// YOLO26 semantic segmentation model.
pub struct Model {
    net: network::Network,
    device: Device,
    dtype: DType,
    image_size: ImageSize,
}

impl Model {
    /// Loads a YOLO26 semantic segmentation model from SafeTensors bytes.
    pub fn from_safetensors(weights: Vec<u8>, config: Config) -> Result<Self> {
        let image_size = config.effective_image_size();
        config.validate()?;
        let dtype = config.dtype.resolve_safetensors(&weights, &config.device)?;
        let vb = VarBuilder::from_buffered_safetensors(weights, dtype, &config.device)?;
        let net = network::Network::load(vb.pp("model"), &config)?;
        Ok(Self {
            net,
            device: config.device,
            dtype,
            image_size,
        })
    }

    /// Loads a semantic segmentation model from a `.safetensors` file.
    pub fn from_safetensors_file(
        path: impl AsRef<std::path::Path>,
        config: Config,
    ) -> Result<Self> {
        Self::from_safetensors(std::fs::read(path.as_ref())?, config)
    }

    /// Loads a YOLO26 semantic segmentation model directly from an official
    /// `.pt` checkpoint.
    #[cfg(feature = "pt")]
    pub fn from_pt_file(path: impl AsRef<std::path::Path>, config: Config) -> Result<Self> {
        let image_size = config.effective_image_size();
        config.validate()?;
        let dtype = config.dtype.resolve_pt(&path, &config.device)?;
        let vb = crate::pt_loader::var_builder_from_pt_file(path, dtype, &config.device)?;
        let net = network::Network::load(vb.pp("model"), &config)?;
        Ok(Self {
            net,
            device: config.device,
            dtype,
            image_size,
        })
    }

    /// Loads a semantic segmentation model from a `.pt` or `.safetensors`
    /// checkpoint, inferring `scale`, `labels_count`, `device`, `dtype` auto.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_file_with(path, Config::default())
    }

    /// Loads a semantic segmentation model from a `.pt` or `.safetensors`
    /// checkpoint, inferring `scale`/`labels_count` while honoring config overrides.
    pub fn from_file_with(path: impl AsRef<std::path::Path>, mut config: Config) -> Result<Self> {
        let path = path.as_ref();
        let shapes = crate::model::checkpoint_shapes(path)?;
        config.scale = crate::model::infer_scale_from_shapes(&shapes)?;
        config.labels_count = crate::model::infer_labels_count_from_shapes(
            crate::model::InferredTask::Semantic,
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

    /// Loads a semantic model from an in-memory `.pt` or `.safetensors` byte
    /// buffer, inferring `scale`/`labels_count` automatically.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes_with(bytes, Config::default())
    }

    /// Loads a semantic model from an in-memory byte buffer, inferring
    /// `scale`/`labels_count` while honoring config overrides.
    pub fn from_bytes_with(bytes: &[u8], mut config: Config) -> Result<Self> {
        let shapes = crate::model::checkpoint_shapes_from_bytes(bytes)?;
        config.scale = crate::model::infer_scale_from_shapes(&shapes)?;
        config.labels_count = crate::model::infer_labels_count_from_shapes(
            crate::model::InferredTask::Semantic,
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

    /// Loads a YOLO26 semantic model directly from an in-memory `.pt` checkpoint
    /// (no filesystem). The `pt` feature is required.
    #[cfg(feature = "pt")]
    pub fn from_pt_bytes(bytes: &[u8], config: Config) -> Result<Self> {
        let image_size = config.effective_image_size();
        config.validate()?;
        let dtype = config.dtype.resolve_pt_bytes(bytes, &config.device)?;
        let vb =
            crate::pt_loader::var_builder_from_pt_bytes(bytes, dtype, &config.device)?.pp("model");
        let net = network::Network::load(vb, &config)?;
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
        Ok(self.net.forward(input)?)
    }

    /// Runs semantic segmentation for one image.
    pub fn predict(&self, image: &Image, mask: &MaskOption) -> Result<Prediction> {
        let (input, letterbox_info) = letterbox(image, self.image_size, self.dtype, &self.device)?;
        let output = self.net.forward(&input)?;
        postprocess_semantic(
            &output,
            &letterbox_info,
            (image.width, image.height),
            mask.high_resolution,
        )
    }
}

fn postprocess_semantic(
    output: &Tensor,
    letterbox: &LetterboxInfo,
    (image_width, image_height): (u32, u32),
    high_resolution: bool,
) -> Result<Prediction> {
    let (classes, out_h, out_w) = match output.dims() {
        [1, c, h, w] => (*c, *h, *w),
        [c, h, w] => (*c, *h, *w),
        dims => {
            return Err(crate::Error::InvalidTensor(format!(
                "expected [1, C, H, W] or [C, H, W] semantic logits, got {dims:?}"
            )));
        }
    };

    let crop_x = letterbox.feature_pad_x(out_w).round() as usize;
    let crop_y = letterbox.feature_pad_y(out_h).round() as usize;
    let content_w = (out_w - 2 * crop_x).min(out_w);
    let content_h = (out_h - 2 * crop_y).min(out_h);

    let logits = match output.dims() {
        [1, _, _, _] => output.clone(),
        [_, _, _] => output.reshape((1, classes, out_h, out_w))?,
        dims => {
            return Err(crate::Error::InvalidTensor(format!(
                "expected [1, C, H, W] or [C, H, W] semantic logits, got {dims:?}"
            )));
        }
    };
    let logits = logits
        .narrow(3, crop_x, content_w)?
        .narrow(2, crop_y, content_h)?;

    let logits = if high_resolution {
        upscale_to_source(
            logits,
            (content_w, content_h),
            (image_width as usize, image_height as usize),
            letterbox,
        )?
    } else {
        logits
    };

    let (final_w, final_h) = if high_resolution {
        (image_width as u16, image_height as u16)
    } else {
        (content_w as u16, content_h as u16)
    };

    let logits = logits
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;

    Prediction::new(final_w, final_h, classes, logits)
}

fn upscale_to_source(
    logits: Tensor,
    (content_w, content_h): (usize, usize),
    (image_width, image_height): (usize, usize),
    letterbox: &LetterboxInfo,
) -> Result<Tensor> {
    let scale = content_w as f32 / (letterbox.model_width as f32 - 2.0 * letterbox.pad_x).max(1.0);
    let src_w = ((image_width as f32) * scale).round() as usize;
    let src_h = ((image_height as f32) * scale).round() as usize;
    let src_w = src_w.min(content_w);
    let src_h = src_h.min(content_h);

    let logits = logits.narrow(3, 0, src_w)?.narrow(2, 0, src_h)?;
    Ok(logits.upsample_bilinear2d(image_height, image_width, false)?)
}
