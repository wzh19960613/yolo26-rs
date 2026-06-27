//! One-to-many branch set for the YOLOE open-vocabulary head.
//!
//! The official `yoloe-26*-seg.pt` ships a symmetric head: one-to-many
//! (`cv2`/`cv3`/`cv4`/`cv5`) and one-to-one (`one2one_cv2/3/4/5`) branches side
//! by side. The trainable model builds the one-to-one set via
//! [`super::Head`] and the one-to-many set via this module, so the
//! saved checkpoint matches the official layout for `.pt` template writing.
//!
//! Mirrors the `load_branch_set(box_name, cls_name, ...)` pattern from the
//! standard YOLO26 detect head (`src/detect/head.rs`): a single generic loader
//! invoked twice with different key prefixes, sharing hidden widths.

use candle_nn::{Conv2dConfig, VarBuilder};

use crate::network::head::dense_branch::DenseBranch;
use crate::network::head::{box_branch::BoxBranch, cls_branch::ClsBranch};

use crate::yoloe::head::contrastive::BnContrastive;
use crate::yoloe::segment::head::OneToManyBranches;

/// A set of detection branches (box + embedding + BN-contrastive) loaded under a
/// shared key prefix. Used for the one-to-many `cv2/cv3/cv4` family.
///
/// Fields are read only indirectly: constructing the branches registers their
/// weights into the shared `VarMap` so a saved checkpoint matches the official
/// `-seg` layout for `.pt` template writing.
#[allow(dead_code)]
pub(crate) struct BranchSet {
    /// Box-distance branches (`cv2.{i}` / `one2one_cv2.{i}`).
    pub(crate) box_branches: Vec<BoxBranch>,
    /// Embedding branches (`cv3.{i}` / `one2one_cv3.{i}`).
    pub(crate) embedding_branches: Vec<ClsBranch>,
    /// Optional BN-contrastive heads (`cv4.{i}` / `one2one_cv4.{i}`).
    pub(crate) bn_contrastive_heads: Option<Vec<BnContrastive>>,
}

/// Loads a `BranchSet` under the given key prefixes, mirroring how the
/// one-to-one head is built. `c2`/`c3` are the shared box/cls hidden widths.
struct BranchSetSpec<'a> {
    box_name: &'a str,
    cls_name: &'a str,
    contrastive_name: &'a str,
    c2: usize,
    c3: usize,
    embed_dim: usize,
    use_bn_contrastive: bool,
    cfg: Conv2dConfig,
}

/// Configuration for building official one-to-many YOLOE segmentation branches.
#[derive(Clone, Copy)]
pub(crate) struct OneToManySpec {
    /// Prompt embedding dimension.
    pub(crate) embed_dim: usize,
    /// Optional checkpoint-inferred class hidden width.
    pub(crate) cls_hidden: Option<usize>,
    /// Optional checkpoint-inferred box hidden width.
    pub(crate) box_hidden: Option<usize>,
    /// Mask coefficient hidden width.
    pub(crate) mask_hidden: usize,
    /// Mask coefficient channel count.
    pub(crate) mask_channels: usize,
    /// Whether BN contrastive heads are required.
    pub(crate) use_bn_contrastive: bool,
    /// Shared convolution configuration.
    pub(crate) cfg: Conv2dConfig,
}

fn load_branch_set(
    vb: VarBuilder,
    input_channels: &[usize],
    spec: BranchSetSpec<'_>,
) -> crate::Result<BranchSet> {
    let reg_max = 1;
    let mut box_branches = Vec::with_capacity(input_channels.len());
    let mut embedding_branches = Vec::with_capacity(input_channels.len());
    for (i, &channels) in input_channels.iter().enumerate() {
        box_branches.push(BoxBranch::load(
            vb.pp(spec.box_name).pp(i.to_string()),
            channels,
            spec.c2,
            reg_max,
            spec.cfg,
        )?);
        embedding_branches.push(ClsBranch::load(
            vb.pp(spec.cls_name).pp(i.to_string()),
            channels,
            spec.c3,
            spec.embed_dim,
            spec.cfg,
        )?);
    }
    let bn_contrastive_heads = if spec.use_bn_contrastive {
        let mut heads = Vec::with_capacity(input_channels.len());
        for i in 0..input_channels.len() {
            heads.push(BnContrastive::load(
                vb.pp(spec.contrastive_name).pp(i.to_string()),
                spec.embed_dim,
            )?);
        }
        Some(heads)
    } else {
        None
    };
    Ok(BranchSet {
        box_branches,
        embedding_branches,
        bn_contrastive_heads,
    })
}

/// Builds the one-to-many branch set (`cv2`/`cv3`/`cv4` detection + `cv5` mask
/// coefficients) under the official `-seg` prefixes. Shares the same hidden
/// widths as the one-to-one head so both branches stay symmetric.
pub(crate) fn build_one_to_many_branches(
    vb: VarBuilder,
    input_channels: &[usize],
    spec: OneToManySpec,
) -> crate::Result<OneToManyBranches> {
    let c2 = spec
        .box_hidden
        .unwrap_or_else(|| 16_usize.max(input_channels[0] / 4).max(4));
    let c3 = spec
        .cls_hidden
        .unwrap_or_else(|| input_channels[0].max(spec.embed_dim.min(100)));
    let detect = load_branch_set(
        vb.clone(),
        input_channels,
        BranchSetSpec {
            box_name: "cv2",
            cls_name: "cv3",
            contrastive_name: "cv4",
            c2,
            c3,
            embed_dim: spec.embed_dim,
            use_bn_contrastive: spec.use_bn_contrastive,
            cfg: spec.cfg,
        },
    )?;
    let mut mask_branches = Vec::with_capacity(input_channels.len());
    for (i, &channels) in input_channels.iter().enumerate() {
        mask_branches.push(DenseBranch::load(
            vb.pp("cv5").pp(i.to_string()),
            channels,
            spec.mask_hidden,
            spec.mask_channels,
            spec.cfg,
        )?);
    }
    Ok(OneToManyBranches {
        detect,
        mask_branches,
    })
}
