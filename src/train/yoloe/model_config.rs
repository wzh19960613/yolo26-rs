//! Configuration for the trainable YOLOE segmentation network.

use candle_core::{DType, Device};

use crate::Scale;
use crate::yoloe::segment::model::train_config::PromptMode;

/// Default YOLOE text/image embedding dimensionality.
pub const EMBED_DIM: usize = 512;
/// Default mask-coefficient channels (`nm`).
pub const MASK_CHANNELS: usize = 32;
/// Official RepRTA inner hidden dim for every YOLOE-26 scale (`reprta.m.w12`
/// has `[2*hidden, embed_dim]`); matches `yoloe-26{scale}-seg.pt`.
pub const REPRTA_HIDDEN: usize = 1024;
/// Official prompt-free LRPC open vocabulary size (`lrpc.{i}.vocab.weight`
/// rows in `yoloe-26{scale}-seg-pf.pt`).
pub const PROMPT_FREE_VOCAB: usize = 4585;

/// Configuration for constructing a trainable YOLOE segmentation model.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model scale (n/s/m/l/x).
    pub scale: Scale,
    /// Compute device.
    pub device: Device,
    /// Tensor dtype for trainable weights.
    pub dtype: DType,
    /// Square input image size.
    pub image_size: usize,
    /// Maximum predictions retained by postprocessing.
    pub max_predictions: usize,
    /// Region/image embedding dimensionality (`embed_dim`).
    pub embed_dim: usize,
    /// Mask-coefficient channels (`nm`).
    pub mask_channels: usize,
    /// Prototype hidden channels (`npr`), scale-derived by default.
    pub proto_channels: usize,
    /// Prompt-free LRPC vocabulary size.
    pub lrpc_vocab: usize,
    /// SAVPE attention channels (official default 16).
    pub savpe_attention_channels: usize,
    /// RepRTA hidden/inner dimension.
    pub reprta_hidden: usize,
    /// Official `cv3`/`one2one_cv3` hidden width per scale (n=80, s=128, m=256,
    /// l=256, x=384). Set by [`Self::new`] to match `yoloe-26{scale}-seg.pt`.
    pub cls_hidden: usize,
    /// Official `cv2`/`one2one_cv2` box-hidden width (16 for every scale), also
    /// the prompt-free LRPC `loc` feature dim. Matches `yoloe-26{scale}-seg*.pt`.
    pub box_hidden: usize,
    /// Prompt-free LRPC open vocabulary size (matches `yoloe-26{scale}-seg-pf.pt`,
    /// `lrpc.{i}.vocab.weight` rows). Used only in [`PromptMode::PromptFree`].
    pub prompt_free_vocab: usize,
    /// Prompt-free LRPC `pf` proposal channel count (n-scale 512, else 1),
    /// matching `yoloe-26{scale}-seg-pf.pt`. Used only in [`PromptMode::PromptFree`].
    pub prompt_free_proposal_channels: usize,
    /// Whether ConvBlock layers are stored as fused conv+bias without BatchNorm.
    ///
    /// Defaults to `false` for official train-time `*-seg` checkpoints and
    /// from-scratch training; [`super::Model::from_safetensors`]
    /// infers and enables it for deploy-style prompt-free checkpoints.
    pub fused_conv_blocks: bool,
    /// Active prompt-alignment mode.
    pub mode: PromptMode,
}

impl ModelConfig {
    /// Creates a config for `yolo26{scale}-seg` with official YOLOE defaults.
    pub fn new(scale: Scale, device: Device, dtype: DType, mode: PromptMode) -> Self {
        Self {
            scale,
            device,
            dtype,
            image_size: crate::model::MODEL_INPUT_SIZE,
            max_predictions: 300,
            embed_dim: EMBED_DIM,
            mask_channels: MASK_CHANNELS,
            proto_channels: scale.channel(256),
            lrpc_vocab: 80,
            savpe_attention_channels: 16,
            reprta_hidden: REPRTA_HIDDEN,
            cls_hidden: official_cls_hidden(scale),
            box_hidden: official_box_hidden(scale),
            prompt_free_vocab: PROMPT_FREE_VOCAB,
            prompt_free_proposal_channels: official_prompt_free_proposal_channels(scale),
            fused_conv_blocks: false,
            mode,
        }
    }

    /// Validates the configuration.
    pub fn validate(&self) -> crate::Result<()> {
        if self.image_size == 0 || !self.image_size.is_multiple_of(32) {
            return Err(crate::Error::InvalidConfig(format!(
                "YOLOE seg image_size {} must be a positive multiple of 32",
                self.image_size
            )));
        }
        if self.max_predictions == 0 {
            return Err(crate::Error::InvalidConfig(
                "YOLOE seg max_predictions must be greater than zero".to_string(),
            ));
        }
        if self.embed_dim == 0 || !self.embed_dim.is_multiple_of(self.savpe_attention_channels) {
            return Err(crate::Error::InvalidConfig(format!(
                "YOLOE embed_dim {} must be divisible by savpe_attention_channels {}",
                self.embed_dim, self.savpe_attention_channels
            )));
        }
        if self.mask_channels == 0 || self.proto_channels == 0 || self.lrpc_vocab == 0 {
            return Err(crate::Error::InvalidConfig(
                "YOLOE seg mask/proto/lrpc channels must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }

    /// Mask-size for instance-mask supervision (`image_size / mask_ratio`).
    pub fn mask_size(&self) -> usize {
        self.image_size / 4
    }
}

/// Returns the official `cv3`/`one2one_cv3` hidden width for a scale, matching
/// `yoloe-26{scale}-seg.pt` (n=80, s=128, m=256, l=256, x=384). These are fixed
/// official sizes; the historical formula `input_channels[0].max(100)` agrees
/// for s/m/l/x but yields 100 for n where the official size is 80.
pub fn official_cls_hidden(scale: Scale) -> usize {
    match scale {
        Scale::N => 80,
        Scale::S => 128,
        Scale::M => 256,
        Scale::L => 256,
        Scale::X => 384,
    }
}

/// Returns the official `cv2`/`one2one_cv2` box-hidden width for a scale,
/// matching `yoloe-26{scale}-seg*.pt` (n=16, s=32, m=64, l=64, x=96). This also
/// equals the prompt-free LRPC `loc` feature dim. Equals
/// `head_input_channels()[0] / 4` for every scale.
pub fn official_box_hidden(scale: Scale) -> usize {
    scale.head_input_channels()[0] / 4
}

/// Returns the official prompt-free LRPC `pf` proposal channel count for a
/// scale, matching `yoloe-26{scale}-seg-pf.pt`. Only n-scale uses a 512-channel
/// proposal filter; the larger scales use a single channel.
pub fn official_prompt_free_proposal_channels(scale: Scale) -> usize {
    match scale {
        Scale::N => 512,
        _ => 1,
    }
}
