use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::model::ImageSize;
use crate::{Result, image::Image};

use super::{Config, Prediction, network};

/// YOLO26 image classification model.
pub struct Model {
    net: network::Network,
    device: Device,
    dtype: DType,
    image_size: ImageSize,
}

impl Model {
    /// Loads a YOLO26 classification model from SafeTensors bytes.
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

    /// Loads a classification model from a `.safetensors` file.
    pub fn from_safetensors_file(
        path: impl AsRef<std::path::Path>,
        config: Config,
    ) -> Result<Self> {
        Self::from_safetensors(std::fs::read(path.as_ref())?, config)
    }

    /// Loads a YOLO26 classification model directly from an official `.pt`
    /// checkpoint.
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

    /// Loads a classification model from a `.pt` or `.safetensors` checkpoint,
    /// inferring `scale`, `labels_count`, `device` and `dtype` automatically.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_file_with(path, Config::default())
    }

    /// Loads a classification model from a `.pt` or `.safetensors` checkpoint,
    /// inferring `scale` and `labels_count` from the checkpoint while honoring
    /// the caller's `config` for `device`/`dtype`/`image_size` overrides.
    pub fn from_file_with(path: impl AsRef<std::path::Path>, mut config: Config) -> Result<Self> {
        let path = path.as_ref();
        let shapes = crate::model::checkpoint_shapes(path)?;
        config.scale = crate::model::infer_scale_from_shapes(&shapes)?;
        config.labels_count = crate::model::infer_labels_count_from_shapes(
            crate::model::InferredTask::Classify,
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

    /// Loads a classification model from an in-memory `.pt` or `.safetensors`
    /// byte buffer, inferring `scale`/`labels_count` automatically.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes_with(bytes, Config::default())
    }

    /// Loads a classification model from an in-memory byte buffer, inferring
    /// `scale`/`labels_count` while honoring config overrides.
    pub fn from_bytes_with(bytes: &[u8], mut config: Config) -> Result<Self> {
        let shapes = crate::model::checkpoint_shapes_from_bytes(bytes)?;
        config.scale = crate::model::infer_scale_from_shapes(&shapes)?;
        config.labels_count = crate::model::infer_labels_count_from_shapes(
            crate::model::InferredTask::Classify,
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

    /// Loads a YOLO26 classification model directly from an in-memory `.pt`
    /// checkpoint (no filesystem). The `pt` feature is required.
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

    /// Runs image classification for one image.
    pub fn predict(&self, image: &Image) -> Result<Vec<Prediction>> {
        let input = classify_preprocess(image, self.image_size, self.dtype, &self.device)?;
        let output = self.net.forward(&input)?;
        postprocess_classification(&output)
    }
}

/// Resizes and center-crops an image into a normalized classification tensor.
fn classify_preprocess(
    image: &Image,
    target: ImageSize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    if target.width == 0 || target.height == 0 {
        return Err(crate::Error::InvalidConfig(
            "classification target dimensions must be greater than zero".to_string(),
        ));
    }

    let src_w = image.width as usize;
    let src_h = image.height as usize;
    let scale = f32::max(
        target.width as f32 / src_w as f32,
        target.height as f32 / src_h as f32,
    );
    let resized_w = (src_w as f32 * scale).round().max(target.width as f32) as usize;
    let resized_h = (src_h as f32 * scale).round().max(target.height as f32) as usize;
    let crop_x = (resized_w - target.width) as f32 * 0.5;
    let crop_y = (resized_h - target.height) as f32 * 0.5;

    let plane = target.width * target.height;
    let mut chw = vec![0.0f32; 3 * plane];

    for dst_y in 0..target.height {
        let resized_y = dst_y as f32 + crop_y;
        let src_y = (resized_y + 0.5) * src_h as f32 / resized_h as f32 - 0.5;
        let y0 = src_y.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let fy = (src_y - y0 as f32).max(0.0);
        let row0 = y0 * src_w * 3;
        let row1 = y1 * src_w * 3;
        let data = &image.data;

        for dst_x in 0..target.width {
            let resized_x = dst_x as f32 + crop_x;
            let src_x = (resized_x + 0.5) * src_w as f32 / resized_w as f32 - 0.5;
            let x0 = src_x.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let fx = (src_x - x0 as f32).max(0.0);

            let i00 = row0 + x0 * 3;
            let i01 = row0 + x1 * 3;
            let i10 = row1 + x0 * 3;
            let i11 = row1 + x1 * 3;
            let w00 = (1.0 - fx) * (1.0 - fy);
            let w01 = fx * (1.0 - fy);
            let w10 = (1.0 - fx) * fy;
            let w11 = fx * fy;

            let out = dst_y * target.width + dst_x;
            for c in 0..3 {
                let value = data[i00 + c] as f32 * w00
                    + data[i01 + c] as f32 * w01
                    + data[i10 + c] as f32 * w10
                    + data[i11 + c] as f32 * w11;
                chw[c * plane + out] = value / 255.0;
            }
        }
    }

    Ok(Tensor::from_vec(chw, (1, 3, target.height, target.width), device)?.to_dtype(dtype)?)
}

fn postprocess_classification(output: &Tensor) -> Result<Vec<Prediction>> {
    let flattened = match output.dims() {
        [1, _classes] => output.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?,
        [_classes] => output.to_dtype(DType::F32)?.to_vec1::<f32>()?,
        dims => {
            return Err(crate::Error::InvalidTensor(format!(
                "expected [1, C] or [C], got {dims:?}"
            )));
        }
    };

    let mut scores: Vec<Prediction> = flattened
        .into_iter()
        .enumerate()
        .map(|(class_id, confidence)| Prediction {
            class_id: class_id as u32,
            confidence,
        })
        .collect();
    scores.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(scores)
}
