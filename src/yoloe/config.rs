use candle_core::Device;

use crate::Scale;
use crate::model::DtypeRequest;
use crate::model::ImageSize;

use crate::yoloe::usage::{BaseTask, CheckpointKind, RepRtaConfig, SavpeConfig, Usage};

/// YOLOE model configuration.
///
/// This is the user-facing config for the YOLOE task root, mirroring the shape
/// of the stable task roots' [`crate::model::config::Base`]: it carries the
/// model `scale`, compute `device`/`dtype`, input `image_size`, and
/// `max_predictions`, plus the YOLOE-specific prompt-module flags. The head's
/// inferred dimensions (`embed_dim`, mask/prototype channels, hidden widths) are
/// not exposed here — they are read from the checkpoint by the model loader.
#[derive(Debug, Clone)]
pub struct Config {
    /// YOLOE model scale.
    pub scale: Scale,
    /// Compute device used by Candle.
    pub device: Device,
    /// Compute dtype request for weights and inference. Defaults to
    /// [`DtypeRequest::Auto`], which infers the dtype from the checkpoint.
    pub dtype: DtypeRequest,
    /// Input tensor size used by preprocessing.
    pub image_size: ImageSize,
    /// Maximum predictions retained by top-k postprocessing.
    pub max_predictions: usize,
    /// Identity family.
    pub checkpoint: CheckpointKind,
    /// Number of prompt embedding dimensions.
    pub prompt_dim: usize,
    /// Whether visual prompt encoding is enabled.
    pub visual_prompts: bool,
    /// Whether prompt-free LRPC vocabulary is enabled.
    pub prompt_free: bool,
    /// RepRTA text-alignment module configuration.
    pub rep_rta: RepRtaConfig,
    /// SAVPE visual-prompt encoder configuration.
    pub savpe: SavpeConfig,
    /// Whether LRPC prompt-free vocabulary lookup is available.
    pub lrpc: bool,
}

impl Default for Config {
    fn default() -> Self {
        let prompt_dim = 512;
        Self {
            scale: Scale::N,
            device: crate::device::auto(),
            dtype: DtypeRequest::Auto,
            image_size: ImageSize::square(crate::model::MODEL_INPUT_SIZE),
            max_predictions: 300,
            checkpoint: CheckpointKind::Prompted,
            prompt_dim,
            visual_prompts: true,
            prompt_free: false,
            rep_rta: RepRtaConfig::default(),
            savpe: SavpeConfig {
                prompt_dim,
                ..SavpeConfig::default()
            },
            lrpc: false,
        }
    }
}

impl Config {
    /// Returns a builder for a [`Config`], starting from [`Config::default`].
    pub fn builder() -> Builder {
        Builder {
            config: Config::default(),
        }
    }

    /// Returns a prompt-free configuration for the chosen scale.
    pub fn prompt_free(scale: Scale) -> Self {
        Self {
            scale,
            checkpoint: CheckpointKind::PromptFree,
            prompt_free: true,
            visual_prompts: false,
            lrpc: true,
            ..Self::default()
        }
    }

    /// Returns a segmentation-first configuration for the chosen scale.
    pub fn segmentation(scale: Scale) -> Self {
        Self {
            scale,
            checkpoint: CheckpointKind::Segmentation,
            ..Self::default()
        }
    }

    /// Returns the base task implied by this YOLOE configuration.
    pub const fn base_task(&self) -> BaseTask {
        match self.checkpoint {
            CheckpointKind::Prompted => BaseTask::Detect,
            CheckpointKind::PromptFree | CheckpointKind::Segmentation => BaseTask::Segment,
        }
    }

    /// Validates that an operating mode is compatible with this configuration.
    pub fn validate_usage(&self, usage: Usage) -> crate::Result<()> {
        match usage {
            Usage::Visual if !self.visual_prompts || !self.savpe.enabled => {
                Err(crate::Error::Unsupported(
                    "YOLOE visual prompts require SAVPE-enabled text/visual checkpoints"
                        .to_string(),
                ))
            }
            Usage::PromptFree if !self.prompt_free || !self.lrpc => Err(crate::Error::Unsupported(
                "YOLOE prompt-free inference requires an LRPC prompt-free checkpoint".to_string(),
            )),
            Usage::TextPrompt if !self.rep_rta.enabled => Err(crate::Error::Unsupported(
                "YOLOE text prompts require RepRTA-enabled checkpoints".to_string(),
            )),
            _ => Ok(()),
        }
    }
}

/// Builder for [`Config`], mirroring the `config_builder().with_*(...).build()`
/// shape used by the task roots. Starts from [`Config::default`].
#[derive(Debug, Clone)]
pub struct Builder {
    config: Config,
}

impl Builder {
    /// Sets the YOLOE model scale.
    pub fn with_scale(mut self, scale: Scale) -> Self {
        self.config.scale = scale;
        self
    }

    /// Sets the compute device used by Candle.
    pub fn with_device(mut self, device: Device) -> Self {
        self.config.device = device;
        self
    }

    /// Forces a specific compute dtype at load time.
    pub fn with_dtype(mut self, dtype: candle_core::DType) -> Self {
        self.config.dtype = DtypeRequest::Fixed(dtype);
        self
    }

    /// Sets a square input size (snapped to a multiple of 32).
    pub fn with_input_size(mut self, input_size: usize) -> Self {
        self.config.image_size = ImageSize::square(input_size).snapped();
        self
    }

    /// Sets the maximum predictions retained by top-k postprocessing.
    pub fn with_max_predictions(mut self, max_predictions: usize) -> Self {
        self.config.max_predictions = max_predictions;
        self
    }

    /// Sets the prompt embedding dimension (shared by RepRTA/SAVPE paths).
    pub fn with_prompt_dim(mut self, prompt_dim: usize) -> Self {
        self.config.prompt_dim = prompt_dim;
        self
    }

    /// Enables or disables visual prompt encoding.
    pub fn with_visual_prompts(mut self, visual_prompts: bool) -> Self {
        self.config.visual_prompts = visual_prompts;
        self
    }

    /// Enables or disables the prompt-free LRPC vocabulary path.
    pub fn with_prompt_free(mut self, prompt_free: bool) -> Self {
        self.config.prompt_free = prompt_free;
        self
    }

    /// Enables or disables LRPC prompt-free vocabulary lookup.
    pub fn with_lrpc(mut self, lrpc: bool) -> Self {
        self.config.lrpc = lrpc;
        self
    }

    /// Replaces the full RepRTA configuration.
    pub fn with_rep_rta(mut self, rep_rta: RepRtaConfig) -> Self {
        self.config.rep_rta = rep_rta;
        self
    }

    /// Convenience toggle for RepRTA's `enabled` flag.
    pub fn with_rep_rta_enabled(mut self, enabled: bool) -> Self {
        self.config.rep_rta.enabled = enabled;
        self
    }

    /// Replaces the full SAVPE configuration.
    pub fn with_savpe(mut self, savpe: SavpeConfig) -> Self {
        self.config.savpe = savpe;
        self
    }

    /// Convenience toggle for SAVPE's `enabled` flag.
    pub fn with_savpe_enabled(mut self, enabled: bool) -> Self {
        self.config.savpe.enabled = enabled;
        self
    }

    /// Replaces the checkpoint family tag.
    pub fn with_checkpoint(mut self, checkpoint: CheckpointKind) -> Self {
        self.config.checkpoint = checkpoint;
        self
    }

    /// Finalizes the configuration.
    pub fn build(self) -> Config {
        self.config
    }
}
