use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use crate::model::{ImageSize, LetterboxInfo, OutputViewer, letterbox};
use crate::{BBox, FilterOption, Result, image::Image};

use super::{Config, Prediction, network, sahi::sliced_predict};

/// YOLO26 object detection model.
pub struct Model {
    net: network::Network,
    device: Device,
    dtype: candle_core::DType,
    image_size: ImageSize,
}

impl Model {
    /// Loads a YOLO26 detection model from SafeTensors bytes.
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

    /// Loads a detection model from a `.safetensors` file.
    pub fn from_safetensors_file(
        path: impl AsRef<std::path::Path>,
        config: Config,
    ) -> Result<Self> {
        Self::from_safetensors(std::fs::read(path.as_ref())?, config)
    }

    /// Loads a YOLO26 detection model directly from an official `.pt` checkpoint
    /// (no manual safetensors conversion required).
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

    /// Loads a detection model from a `.pt` or `.safetensors` checkpoint,
    /// inferring `scale`, `labels_count`, `device` and `dtype` automatically.
    /// Use [`Self::from_file_with`] to override any of those explicitly.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_file_with(path, Config::default())
    }

    /// Loads a detection model from a `.pt` or `.safetensors` checkpoint,
    /// inferring `scale` and `labels_count` from the checkpoint while honoring
    /// the caller's `config` for `device`/`dtype`/`image_size` overrides.
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

    /// Loads a detection model from an in-memory `.pt` or `.safetensors` byte
    /// buffer, inferring `scale`, `labels_count`, `device` and `dtype`
    /// automatically. Use [`Self::from_bytes_with`] to override any of those.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes_with(bytes, Config::default())
    }

    /// Loads a detection model from an in-memory `.pt` or `.safetensors` byte
    /// buffer, inferring `scale` and `labels_count` from the checkpoint while
    /// honoring the caller's `config` for `device`/`dtype`/`image_size` overrides.
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

    /// Loads a YOLO26 detection model directly from an official `.pt` checkpoint
    /// held in memory (no filesystem). The `pt` feature is required.
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
    pub fn dtype(&self) -> candle_core::DType {
        self.dtype
    }

    /// Runs the raw network forward pass for a preprocessed tensor.
    pub fn forward_tensor(&self, input: &Tensor) -> Result<Tensor> {
        Ok(self.net.forward(input)?)
    }

    /// Runs object detection for one image.
    pub fn predict(&self, image: &Image, filter: &FilterOption) -> Result<Vec<Prediction>> {
        let (input, letterbox_info) = letterbox(image, self.image_size, self.dtype, &self.device)?;
        let output = self.net.forward(&input)?;
        postprocess(&output, &letterbox_info, image.width, image.height, filter)
    }

    /// Runs SAHI-style sliced prediction with this model.
    pub fn predict_sahi(
        &self,
        image: &Image,
        inference: &FilterOption,
        sahi_options: &crate::detect::sahi::Options,
    ) -> Result<Vec<Prediction>> {
        sliced_predict(self, image, inference, sahi_options)
    }
}

/// Decodes YOLO26 end-to-end head output into source-image detections.
pub fn postprocess(
    output: &Tensor,
    letterbox: &LetterboxInfo,
    image_width: u32,
    image_height: u32,
    filter: &FilterOption,
) -> Result<Vec<Prediction>> {
    const COLS: usize = 6;
    let (rows, flattened) = match output.dims() {
        [1, rows, COLS] => (
            *rows,
            output
                .squeeze(0)?
                .flatten_all()?
                .to_dtype(candle_core::DType::F32)?
                .to_vec1::<f32>()?,
        ),
        [rows, COLS] => (
            *rows,
            output
                .flatten_all()?
                .to_dtype(candle_core::DType::F32)?
                .to_vec1::<f32>()?,
        ),
        err_dims => {
            return Err(crate::Error::InvalidTensor(format!(
                "expected [1, N, {COLS}] or [N, {COLS}], got {err_dims:?}"
            )));
        }
    };

    let mut detections = Vec::new();
    for row in 0..rows {
        let r = OutputViewer::for_detect(&flattened, row).ok_or_else(|| {
            crate::Error::InvalidTensor(format!("detect output row {row} out of range"))
        })?;
        let (confidence, class_id) = match r.check(filter) {
            Some(pair) => pair,
            None => continue,
        };

        let bbox = BBox::from_xyxy(
            letterbox.to_source_x(r.x1()),
            letterbox.to_source_y(r.y1()),
            letterbox.to_source_x(r.x2()),
            letterbox.to_source_y(r.y2()),
        )
        .clamp(image_width, image_height);

        if bbox.area() <= 0.0 {
            continue;
        }

        detections.push(Prediction {
            bbox,
            confidence,
            class_id,
        });
    }

    Ok(detections)
}
