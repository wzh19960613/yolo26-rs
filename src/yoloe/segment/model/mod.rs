pub(crate) mod config;
pub(crate) mod config_infer;
pub(crate) mod network;
pub(crate) mod predict;
pub(crate) mod shape;
pub mod train_config;
pub(crate) mod visual;
pub(crate) mod visual_encode;

use candle_core::DType;
use candle_nn::VarBuilder;

use crate::model::ImageSize;

use crate::yoloe::config::Config;
use crate::yoloe::reprta::RepRta;
use crate::yoloe::segment::model::config::Config as ModelConfig;
use crate::yoloe::segment::model::network::Network;

/// YOLOE open-vocabulary segmentation model backed by SafeTensors weights.
///
/// The model holds the loaded network and resolved compute dtype but not the
/// prompt state — pass a [`Session`](super::Session) to each
/// `forward_*` / `predict*` call. This mirrors the task-root `Model` shape
/// while keeping YOLOE's per-call prompt flexibility.
pub struct Model {
    pub(crate) network: Network,
    pub(crate) config: ModelConfig,
    pub(crate) image_size: ImageSize,
    /// Concrete compute dtype resolved from `config.dtype` at load time.
    pub(crate) dtype: DType,
    /// RepRTA text-alignment adapter loaded from `model.23.reprta` at load
    /// time; `None` when the checkpoint has no RepRTA (e.g. prompt-free only).
    /// Borrowed by [`Session::text`](super::Session::text) to align text-prompt
    /// embeddings through the official CLIP → RepRTA → score path.
    pub(crate) reprta: Option<RepRta>,
}

impl Model {
    /// Loads a YOLOE open-vocabulary segmentation model from SafeTensors bytes,
    /// inferring the scale and head layout from the checkpoint and taking
    /// `device`/`dtype`/`max_predictions` from `config`.
    pub fn from_safetensors(weights: Vec<u8>, config: Config) -> crate::Result<Self> {
        let model_config =
            ModelConfig::infer_from_safetensors_bytes_with_config(&weights, &config)?;
        Self::load(weights, model_config, config.image_size)
    }

    /// Loads a YOLOE segmentation model from a `.pt` or `.safetensors`
    /// checkpoint, inferring `scale`, `device`, `dtype` and the head layout
    /// automatically — the caller only supplies the path.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> crate::Result<Self> {
        Self::from_file_with(path, Config::default())
    }

    /// Like [`Self::from_file`], but honors the caller's `config` for
    /// `device`/`dtype`/`image_size`/`max_predictions`/prompt-module overrides.
    pub fn from_file_with(
        path: impl AsRef<std::path::Path>,
        config: Config,
    ) -> crate::Result<Self> {
        let path_ref = path.as_ref();
        if crate::model::is_pt_path(path_ref) {
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
            let weights = std::fs::read(path_ref)?;
            Self::from_safetensors(weights, config)
        }
    }

    /// Loads a YOLOE model from an in-memory `.pt` or `.safetensors` byte
    /// buffer, inferring `scale`, `device`, `dtype` and the head layout
    /// automatically.
    pub fn from_bytes(bytes: &[u8]) -> crate::Result<Self> {
        Self::from_bytes_with(bytes, Config::default())
    }

    /// Like [`Self::from_bytes`], but honors the caller's `config` for
    /// `device`/`dtype`/`image_size`/`max_predictions`/prompt-module overrides.
    pub fn from_bytes_with(bytes: &[u8], config: Config) -> crate::Result<Self> {
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

    /// Loads a YOLOE open-vocabulary segmentation model directly from an
    /// official `.pt` checkpoint, inferring the head layout from it.
    #[cfg(feature = "pt")]
    pub fn from_pt_file(path: impl AsRef<std::path::Path>, config: Config) -> crate::Result<Self> {
        let path_ref = path.as_ref();
        let model_config = ModelConfig::infer_from_pt_file_with_config(path_ref, &config)?;
        let dtype = model_config
            .dtype
            .resolve_pt(path_ref, &model_config.device)?;
        let vb = crate::pt_loader::var_builder_from_pt_file(path, dtype, &model_config.device)?
            .pp("model");
        let network = Network::load(vb.clone(), &model_config)?;
        let reprta = RepRta::load_optional(vb.pp("23").pp("reprta"))?;
        Ok(Self {
            network,
            config: model_config,
            image_size: config.image_size,
            dtype,
            reprta,
        })
    }

    /// Loads a YOLOE segmentation model directly from an in-memory `.pt` byte
    /// buffer (no filesystem), mirroring [`Self::from_pt_file`]. The `pt`
    /// feature is required.
    #[cfg(feature = "pt")]
    pub fn from_pt_bytes(bytes: &[u8], config: Config) -> crate::Result<Self> {
        let model_config = ModelConfig::infer_from_pt_bytes_with_config(bytes, &config)?;
        let dtype = model_config
            .dtype
            .resolve_pt_bytes(bytes, &model_config.device)?;
        let vb = crate::pt_loader::var_builder_from_pt_bytes(bytes, dtype, &model_config.device)?
            .pp("model");
        let network = Network::load(vb.clone(), &model_config)?;
        let reprta = RepRta::load_optional(vb.pp("23").pp("reprta"))?;
        Ok(Self {
            network,
            config: model_config,
            image_size: config.image_size,
            dtype,
            reprta,
        })
    }

    fn load(
        weights: Vec<u8>,
        model_config: ModelConfig,
        image_size: ImageSize,
    ) -> crate::Result<Self> {
        model_config.validate()?;
        let dtype = model_config
            .dtype
            .resolve_safetensors(&weights, &model_config.device)?;
        let vb = VarBuilder::from_buffered_safetensors(weights, dtype, &model_config.device)?
            .pp("model");
        let network = Network::load(vb.clone(), &model_config)?;
        let reprta = RepRta::load_optional(vb.pp("23").pp("reprta"))?;
        Ok(Self {
            network,
            config: model_config,
            image_size,
            dtype,
            reprta,
        })
    }

    /// Returns this model's inferred internal config.
    pub const fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Returns the resolved compute dtype.
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the class count required by the prompt-free LRPC vocabulary, if present.
    pub fn prompt_free_class_count(&self) -> Option<usize> {
        self.network.prompt_free_class_count()
    }

    /// Returns the RepRTA text-alignment adapter loaded from this checkpoint,
    /// if present. Borrowed by [`Session::text`](super::Session::text) to align
    /// text-prompt embeddings through the official CLIP → RepRTA → score path.
    pub fn reprta(&self) -> Option<&RepRta> {
        self.reprta.as_ref()
    }
}
