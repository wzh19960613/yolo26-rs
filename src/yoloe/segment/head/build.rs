//! Shared build body for [`Head`] loaders, split out so
//! the public entry-point file stays under the per-file line cap.

use candle_nn::{Conv2dConfig, VarBuilder};

use crate::network::head::{dense_branch::DenseBranch, proto::Proto26};

use crate::yoloe::detect::head::branch_set::OneToManySpec;
use crate::yoloe::head::contrastive::Contrastive;
use crate::yoloe::segment::head::Head;

/// Internal YOLOE segmentation head loader configuration.
#[derive(Clone, Copy)]
pub(crate) struct LoadInnerSpec<'a> {
    /// Prompt embedding dimension.
    pub(crate) embed_dim: usize,
    /// Maximum number of retained detections.
    pub(crate) max_det: usize,
    /// Mask coefficient channel count.
    pub(crate) nm: usize,
    /// Prototype hidden channel count.
    pub(crate) npr: usize,
    /// Prompt scoring configuration.
    pub(crate) contrastive: Contrastive,
    /// Mask branch key prefix.
    pub(crate) mask_branch: &'a str,
    /// Optional checkpoint-inferred class hidden width.
    pub(crate) cls_hidden: Option<usize>,
    /// Optional checkpoint-inferred box hidden width.
    pub(crate) box_hidden: Option<usize>,
    /// Optional checkpoint-inferred mask hidden width.
    pub(crate) mask_hidden: Option<usize>,
    /// Whether BN contrastive heads are required.
    pub(crate) require_bn_contrastive: bool,
    /// Whether to build official one-to-many branches.
    pub(crate) build_one_to_many: bool,
}

impl Head {
    /// Shared loader body invoked by every public entry point. Optionally also
    /// builds the one-to-many branches so a saved checkpoint matches the
    /// official symmetric `-seg` layout.
    pub(crate) fn load_inner(
        vb: VarBuilder,
        input_channels: &[usize],
        spec: LoadInnerSpec<'_>,
    ) -> crate::Result<Self> {
        if spec.nm == 0 {
            return Err(crate::Error::InvalidConfig(
                "YOLOE segmentation mask channels must be greater than zero".to_string(),
            ));
        }
        if spec.mask_branch.is_empty() {
            return Err(crate::Error::InvalidConfig(
                "YOLOE segmentation mask branch prefix must not be empty".to_string(),
            ));
        }
        let detect: crate::yoloe::detect::head::Head = if spec.require_bn_contrastive {
            crate::yoloe::detect::head::Head::load_with_bn_contrastive_hidden(
                vb.clone(),
                input_channels,
                spec.embed_dim,
                spec.max_det,
                spec.contrastive,
                spec.cls_hidden,
                spec.box_hidden,
            )?
        } else {
            crate::yoloe::detect::head::Head::load_with_hidden(
                vb.clone(),
                input_channels,
                spec.embed_dim,
                spec.max_det,
                spec.contrastive,
                spec.cls_hidden,
                spec.box_hidden,
            )?
        };
        let c5 = spec
            .mask_hidden
            .unwrap_or_else(|| (input_channels[0] / 4).max(spec.nm));
        let cfg = Conv2dConfig::default();
        let mut mask_branches = Vec::with_capacity(input_channels.len());
        for (i, &channels) in input_channels.iter().enumerate() {
            mask_branches.push(DenseBranch::load(
                vb.pp(spec.mask_branch).pp(i.to_string()),
                channels,
                c5,
                spec.nm,
                cfg,
            )?);
        }
        let proto = Proto26::load(vb.pp("proto"), input_channels, spec.npr, spec.nm)?;
        let one_to_many = if spec.build_one_to_many {
            Some(
                crate::yoloe::detect::head::branch_set::build_one_to_many_branches(
                    vb.clone(),
                    input_channels,
                    OneToManySpec {
                        embed_dim: spec.embed_dim,
                        cls_hidden: spec.cls_hidden,
                        box_hidden: spec.box_hidden,
                        mask_hidden: c5,
                        mask_channels: spec.nm,
                        use_bn_contrastive: spec.require_bn_contrastive,
                        cfg,
                    },
                )?,
            )
        } else {
            None
        };
        Ok(Self {
            detect,
            mask_branches,
            proto,
            nm: spec.nm,
            one_to_many,
        })
    }
}
