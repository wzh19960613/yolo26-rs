use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use crate::model::{ImageSize, LetterboxInfo, OutputViewer, flattened_rows, letterbox};
use crate::{FilterOption, Result, image::Image};

use super::{Config, Keypoint, Prediction, network};

/// YOLO26 pose/keypoint estimation model.
pub struct Model {
    net: network::Network,
    device: Device,
    dtype: candle_core::DType,
    image_size: ImageSize,
    keypoints_count: usize,
    keypoint_dims: usize,
}

impl Model {
    /// Loads a YOLO26 pose model from SafeTensors bytes.
    pub fn from_safetensors(weights: Vec<u8>, config: Config) -> Result<Self> {
        let image_size = config.base.effective_image_size();
        if config.keypoints_count == 0 || config.keypoint_dims < 2 {
            return Err(crate::Error::InvalidConfig(
                "YOLO26 pose models require keypoints_count > 0 and keypoint_dims >= 2".to_string(),
            ));
        }
        config.base.validate()?;
        let dtype = config
            .base
            .dtype
            .resolve_safetensors(&weights, &config.base.device)?;
        let vb = VarBuilder::from_buffered_safetensors(weights, dtype, &config.base.device)?;
        let net = network::load(vb.pp("model"), &config)?;
        Ok(Self {
            net,
            device: config.base.device,
            dtype,
            image_size,
            keypoints_count: config.keypoints_count,
            keypoint_dims: config.keypoint_dims,
        })
    }

    /// Loads a pose model from a `.safetensors` file.
    pub fn from_safetensors_file(
        path: impl AsRef<std::path::Path>,
        config: Config,
    ) -> Result<Self> {
        Self::from_safetensors(std::fs::read(path.as_ref())?, config)
    }

    /// Loads a YOLO26 pose model directly from an official `.pt` checkpoint.
    #[cfg(feature = "pt")]
    pub fn from_pt_file(path: impl AsRef<std::path::Path>, config: Config) -> Result<Self> {
        let image_size = config.base.effective_image_size();
        if config.keypoints_count == 0 || config.keypoint_dims < 2 {
            return Err(crate::Error::InvalidConfig(
                "YOLO26 pose models require keypoints_count > 0 and keypoint_dims >= 2".to_string(),
            ));
        }
        config.base.validate()?;
        let dtype = config.base.dtype.resolve_pt(&path, &config.base.device)?;
        let vb = crate::pt_loader::var_builder_from_pt_file(path, dtype, &config.base.device)?;
        let net = network::load(vb.pp("model"), &config)?;
        Ok(Self {
            net,
            device: config.base.device,
            dtype,
            image_size,
            keypoints_count: config.keypoints_count,
            keypoint_dims: config.keypoint_dims,
        })
    }

    /// Loads a pose model from a `.pt` or `.safetensors` checkpoint, inferring
    /// `scale`, `labels_count`, `keypoints_count`, `device`, `dtype` auto.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_file_with(path, Config::default())
    }

    /// Loads a pose model from a `.pt` or `.safetensors` checkpoint, inferring
    /// `scale`/`labels_count`/`keypoints_count` while honoring config overrides.
    pub fn from_file_with(path: impl AsRef<std::path::Path>, mut config: Config) -> Result<Self> {
        let path = path.as_ref();
        let shapes = crate::model::checkpoint_shapes(path)?;
        config.base.scale = crate::model::infer_scale_from_shapes(&shapes)?;
        config.base.labels_count = crate::model::infer_labels_count_from_shapes(
            crate::model::InferredTask::Pose,
            &shapes,
        )?;
        config.keypoints_count = crate::model::infer_keypoints_count(&shapes)?;
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

    /// Loads a pose model from an in-memory `.pt` or `.safetensors` byte
    /// buffer, inferring `scale`/`labels_count`/`keypoints_count` automatically.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes_with(bytes, Config::default())
    }

    /// Loads a pose model from an in-memory byte buffer, inferring
    /// `scale`/`labels_count`/`keypoints_count` while honoring config overrides.
    pub fn from_bytes_with(bytes: &[u8], mut config: Config) -> Result<Self> {
        let shapes = crate::model::checkpoint_shapes_from_bytes(bytes)?;
        config.base.scale = crate::model::infer_scale_from_shapes(&shapes)?;
        config.base.labels_count = crate::model::infer_labels_count_from_shapes(
            crate::model::InferredTask::Pose,
            &shapes,
        )?;
        config.keypoints_count = crate::model::infer_keypoints_count(&shapes)?;
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

    /// Loads a YOLO26 pose model directly from an in-memory `.pt` checkpoint
    /// (no filesystem). The `pt` feature is required.
    #[cfg(feature = "pt")]
    pub fn from_pt_bytes(bytes: &[u8], config: Config) -> Result<Self> {
        let image_size = config.base.effective_image_size();
        if config.keypoints_count == 0 || config.keypoint_dims < 2 {
            return Err(crate::Error::InvalidConfig(
                "YOLO26 pose models require keypoints_count > 0 and keypoint_dims >= 2".to_string(),
            ));
        }
        config.base.validate()?;
        let dtype = config
            .base
            .dtype
            .resolve_pt_bytes(bytes, &config.base.device)?;
        let vb = crate::pt_loader::var_builder_from_pt_bytes(bytes, dtype, &config.base.device)?
            .pp("model");
        let net = network::load(vb, &config)?;
        Ok(Self {
            net,
            device: config.base.device,
            dtype,
            image_size,
            keypoints_count: config.keypoints_count,
            keypoint_dims: config.keypoint_dims,
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

    /// Runs pose/keypoint prediction for one image.
    pub fn predict(&self, image: &Image, filter: &FilterOption) -> Result<Vec<Prediction>> {
        let (input, letterbox_info) = letterbox(image, self.image_size, self.dtype, &self.device)?;
        let output = self.net.forward(&input)?;
        postprocess_pose(
            &output,
            &letterbox_info,
            (image.width, image.height),
            filter,
            (self.keypoints_count, self.keypoint_dims),
        )
    }
}

fn postprocess_pose(
    output: &Tensor,
    letterbox: &LetterboxInfo,
    (image_width, image_height): (u32, u32),
    filter: &FilterOption,
    (keypoints_count, keypoint_dims): (usize, usize),
) -> Result<Vec<Prediction>> {
    let cols = 6 + keypoints_count * keypoint_dims;
    let (rows, flattened) = flattened_rows(output, cols)?;
    let mut detections = Vec::new();

    for row in 0..rows {
        let r = OutputViewer::for_dynamic(&flattened, row, cols).ok_or_else(|| {
            crate::Error::InvalidTensor(format!("pose output row {row} out of range"))
        })?;
        let (confidence, class_id) = match r.check(filter) {
            Some(pair) => pair,
            None => continue,
        };

        let bbox = letterbox.xyxy_to_source_bbox(&r.as_slice()[..4], image_width, image_height);
        if bbox.area() <= 0.0 {
            continue;
        }

        let mut keypoints = Vec::with_capacity(keypoints_count);
        for keypoint_idx in 0..keypoints_count {
            let base = 6 + keypoint_idx * keypoint_dims;
            let x = letterbox.to_source_x(r[base]);
            let y = letterbox.to_source_y(r[base + 1]);
            let visibility = (keypoint_dims >= 3).then_some(r[base + 2]);
            keypoints.push(Keypoint { x, y, visibility });
        }

        detections.push(Prediction {
            bbox,
            confidence,
            class_id,
            keypoints,
        });
    }

    Ok(detections)
}
