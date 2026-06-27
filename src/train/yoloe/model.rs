//! Trainable YOLOE segmentation model backed by a Candle `VarMap`.
//!
//! Wraps the shared backbone+neck, the open-vocabulary segmentation head
//! (boxes + image embeddings + BNContrastiveHead + mask coefficients + proto),
//! the prompt-free LRPC head, the visual-prompt SAVPE encoder, and the text
//! RepRTA adapter. All modules share one trainable `VarMap` whose key layout
//! matches the official YOLOE-26 safetensors, so a saved checkpoint loads
//! directly into [`Model`](super::Model).

use std::collections::HashMap;

use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_nn::VarMap;

use crate::network::backbone::Base as BackboneBase;
use crate::network::blocks::with_fused_conv_layout;
use crate::network::neck::Base as NeckBase;

use super::head::TrainableSegHead;
use super::model_config::ModelConfig;
use super::output::Output;
use super::prompt_free_head::TrainablePromptFreeHead;
use crate::yoloe::head::contrastive::Contrastive;
use crate::yoloe::head::lrpc::pyramid::Pyramid;
use crate::yoloe::reprta::RepRta;
use crate::yoloe::savpe::encoder::Encoder;
use crate::yoloe::segment::head::Head;
use crate::yoloe::segment::model::train_config::PromptMode;
use crate::yoloe::usage::EmbeddingTable;

/// Trainable YOLOE segmentation network with all prompt heads.
pub struct Model {
    /// Shared trainable variable map; save() writes the YOLOE checkpoint.
    pub varmap: VarMap,
    pub(crate) backbone: BackboneBase,
    pub(crate) neck: NeckBase,
    /// Task head variant: prompted `-seg` or prompt-free `-seg-pf`.
    pub(crate) head: TrainableSegHead,
    pub(crate) savpe: Encoder,
    pub(crate) reprta: RepRta,
    pub(crate) config: ModelConfig,
}

impl Model {
    /// Builds a fresh trainable YOLOE segment model from a config.
    ///
    /// For [`PromptMode::PromptFree`] the layout is forced to the
    /// official fused conv+bias (no BatchNorm) form, matching
    /// `yoloe-26*-seg-pf.pt`; the prompted `-seg` layout keeps the config's
    /// `fused_conv_blocks` (train-time conv+BN by default).
    pub fn new(mut config: ModelConfig) -> crate::Result<Self> {
        if config.mode == PromptMode::PromptFree {
            config.fused_conv_blocks = true;
        }
        with_fused_conv_layout(config.fused_conv_blocks, || Self::new_inner(config))
    }

    fn new_inner(config: ModelConfig) -> crate::Result<Self> {
        config.validate()?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, config.dtype, &config.device).pp("model");
        let input_channels = config.scale.head_input_channels();
        let backbone = BackboneBase::load(vb.clone(), config.scale)?;
        let neck = NeckBase::load(vb.clone(), config.scale)?;
        // Head layout follows the prompt mode:
        // - TextPrompt/Visual: full symmetric `-seg` head (one-to-many +
        //   one-to-one + BNContrastive), no LRPC.
        // - PromptFree: `-seg-pf` head (2-layer box/embed stems + cv5 mask +
        //   one2one_cv5 mask + 4585-class LRPC), no BNContrastive, no one-to-many.
        let head = if config.mode == PromptMode::PromptFree {
            TrainableSegHead::PromptFree(TrainablePromptFreeHead::load(vb.pp("23"), &config)?)
        } else {
            let prompted = Head::load_full(
                vb.pp("23"),
                &input_channels,
                config.embed_dim,
                config.max_predictions,
                config.mask_channels,
                config.proto_channels,
                Contrastive::default(),
                "one2one_cv5",
                Some(config.cls_hidden),
                None,
                None,
                true,
                true,
            )?;
            TrainableSegHead::Prompted(prompted)
        };
        let savpe = Encoder::load_with_class_count(
            vb.pp("23").pp("savpe"),
            &input_channels,
            config.embed_dim,
            config.lrpc_vocab,
        )?;
        let reprta = RepRta::load_with_hidden(
            vb.pp("23").pp("reprta"),
            config.embed_dim,
            config.reprta_hidden,
        )?;
        Ok(Self {
            varmap,
            backbone,
            neck,
            head,
            savpe,
            reprta,
            config,
        })
    }

    /// Builds a trainable YOLOE segment model and shape-loads weights from a
    /// safetensors checkpoint.
    ///
    /// Official text/visual and prompt-free YOLOE checkpoints do not expose the
    /// same head variables, so tensors that are absent or shape-mismatched are
    /// left at their initialized values. Conv-BN vs fused-conv layout is inferred
    /// from the checkpoint key set before the trainable variables are created.
    pub fn from_safetensors<P: AsRef<std::path::Path>>(
        mut config: ModelConfig,
        path: P,
    ) -> crate::Result<Self> {
        let path_ref = path.as_ref();
        let tensors = candle_core::safetensors::load(path_ref, &config.device)?;
        config.fused_conv_blocks = infer_fused_conv_blocks(&tensors);
        let mut model = Self::new(config)?;
        model.load_tensor_map(tensors)?;
        Ok(model)
    }

    /// Loads a trainable YOLOE model from an official `.pt` checkpoint,
    /// mirroring [`Self::from_safetensors`].
    #[cfg(feature = "pt")]
    pub fn from_pt_file<P: AsRef<std::path::Path>>(
        mut config: ModelConfig,
        path: P,
    ) -> crate::Result<Self> {
        let tensors = crate::pt_loader::load_pt_to_tensors(path.as_ref(), &config.device)?;
        config.fused_conv_blocks = infer_fused_conv_blocks(&tensors);
        let mut model = Self::new(config)?;
        model.load_tensor_map(tensors)?;
        Ok(model)
    }

    /// Returns the configured dtype.
    pub const fn dtype(&self) -> DType {
        self.config.dtype
    }

    /// Returns the configured device.
    pub fn device(&self) -> &candle_core::Device {
        &self.config.device
    }

    /// Returns the underlying config.
    pub const fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Applies RepRTA to a text prompt table, matching inference alignment.
    pub fn align_text_prompts(&self, table: &EmbeddingTable) -> crate::Result<EmbeddingTable> {
        self.reprta.forward_table(table)
    }

    /// Runs the shared backbone+neck+head in training mode (text/visual prompts).
    pub fn forward_dense(
        &self,
        input: &candle_core::Tensor,
        prompts: &EmbeddingTable,
    ) -> crate::Result<Output> {
        let head = match &self.head {
            TrainableSegHead::Prompted(head) => head,
            TrainableSegHead::PromptFree(_) => {
                return Err(crate::Error::InvalidConfig(
                    "YOLOE text/visual forward requires a prompted model".to_string(),
                ));
            }
        };
        let features = self.backbone.forward(input)?;
        let pyramid = self.neck.forward(&features)?;
        let head_features = [&pyramid.small, &pyramid.medium, &pyramid.large];
        head.forward_dense(&head_features, prompts)
    }

    /// Encodes batched visual prompt masks into a prompt embedding table via SAVPE.
    ///
    /// SAVPE returns `[batch, prompts, embed_dim]`; for single-image visual
    /// prompting the batch axis is squeezed to yield `[classes, embed_dim]`.
    /// `visuals.tensor` is the `[batch, prompts, h, w]` prompt mask built by the
    /// caller (typically a dataset loader).
    pub fn encode_visual_prompts(
        &self,
        input: &candle_core::Tensor,
        visuals: &crate::yoloe::visuals::BatchVisuals,
        classes: Vec<String>,
    ) -> crate::Result<EmbeddingTable> {
        let features = self.backbone.forward(input)?;
        let pyramid = self.neck.forward(&features)?;
        let head_features = [&pyramid.small, &pyramid.medium, &pyramid.large];
        let embeddings = self.savpe.forward(&head_features, &visuals.tensor)?;
        let dims = embeddings.dims();
        let table = if dims.len() == 3 && dims[0] == 1 {
            embeddings.squeeze(0)?
        } else {
            embeddings
        };
        EmbeddingTable::new(table, classes)
    }

    /// Returns the prompt-free LRPC head for loss computation, present only in
    /// [`PromptMode::PromptFree`].
    pub fn lrpc(&self) -> Option<&Pyramid> {
        self.head.prompt_free_lrpc()
    }

    /// Returns the SAVPE encoder for loss computation.
    pub const fn savpe(&self) -> &Encoder {
        &self.savpe
    }

    /// Returns the RepRTA adapter.
    pub const fn reprta(&self) -> &RepRta {
        &self.reprta
    }

    /// Returns a sorted `(name, var)` iterator over all trainable variables.
    pub fn variables(&self) -> crate::Result<Vec<(String, candle_core::Var)>> {
        let data = self.varmap.data().lock().map_err(|_| {
            crate::Error::InvalidConfig("YOLOE variable map lock was poisoned".to_string())
        })?;
        Ok(data.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
    }

    fn load_tensor_map(&mut self, tensors: HashMap<String, Tensor>) -> crate::Result<()> {
        let mut loaded = 0usize;
        let data = self.varmap.data().lock().map_err(|_| {
            crate::Error::InvalidConfig("YOLOE variable map lock was poisoned".to_string())
        })?;
        for (name, var) in data.iter() {
            if let Some(tensor) = tensors
                .get(name)
                .filter(|tensor| tensor.shape() == var.shape())
            {
                var.set(&tensor.to_dtype(var.dtype())?)?;
                loaded += 1;
            }
        }
        if loaded == 0 {
            return Err(crate::Error::InvalidConfig(
                "YOLOE checkpoint did not contain any matching trainable tensor".to_string(),
            ));
        }
        Ok(())
    }
}

fn infer_fused_conv_blocks(tensors: &HashMap<String, Tensor>) -> bool {
    let has_bn = tensors.keys().any(|name| name.ends_with(".bn.weight"));
    let has_conv_bias = tensors.keys().any(|name| name.ends_with(".conv.bias"));
    has_conv_bias && !has_bn
}
