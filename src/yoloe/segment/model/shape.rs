//! Identity shape inference for the YOLOE segmentation head config.
//!
//! Extracted from [`crate::yoloe::segment::model::config`]: reads the official
//! SafeTensors/`.pt` tensor shapes to derive the segment head's embed/mask/
//! proto channels, hidden widths, branch selection, and availability flags,
//! returning the inferred head fields.

use std::collections::BTreeMap;

use crate::yoloe::checkpoint::layout::Layout;
use crate::yoloe::first_dim::first_dim;
use crate::yoloe::segment::model::config::MaskBranch;

/// Fields inferred from checkpoint tensor shapes, used to construct a
/// [`super::Config`].
pub(crate) struct InferredSegmentShapes {
    /// Mask coefficient branch selected for the head.
    pub(crate) mask_branch: MaskBranch,
    /// Region/prompt embedding dimension.
    pub(crate) embed_dim: usize,
    /// Mask coefficient channels.
    pub(crate) mask_channels: usize,
    /// Prototype hidden channels.
    pub(crate) proto_channels: usize,
    /// Whether official prompt-free LRPC heads are available.
    pub(crate) official_lrpc: bool,
    /// Whether official SAVPE visual-prompt encoder weights are available.
    pub(crate) official_savpe: bool,
    /// Whether the regular text/visual prompt head final projections exist.
    pub(crate) prompt_head: bool,
    /// Intermediate width of the classification/embedding branch `cv3`.
    pub(crate) cls_hidden: usize,
    /// Intermediate width of the box-regression branch `cv2`.
    pub(crate) box_hidden: usize,
    /// Intermediate width of the mask-coefficient branch `cv5`.
    pub(crate) mask_hidden: usize,
    /// Intermediate width of the official SAVPE encoder.
    pub(crate) savpe_hidden: usize,
}

/// Infers the segment head layout from checkpoint tensor shapes.
///
/// `scale`/`device`/`dtype`/`max_predictions` are passed through verbatim; only
/// the head-specific fields are read from `shapes`.
pub(crate) fn infer_segment_shapes(
    shapes: &BTreeMap<String, Vec<usize>>,
) -> crate::Result<InferredSegmentShapes> {
    let layout = Layout::from_tensor_names(shapes.keys().cloned().collect::<Vec<_>>());
    let plan = layout.compatible_segment_head_plan()?;
    let mask_branch = if plan.uses_official_segment_mask_branch {
        MaskBranch::OfficialOne2OneCv5
    } else {
        MaskBranch::CompatibleOne2OneCv4
    };
    // embed_dim: the text/visual checkpoint exposes it as the projection
    // `one2one_cv3.0.2.weight`; the prompt-free (`-pf`) checkpoint has no such
    // projection, so fall back to the LRPC prompt-free conv `lrpc.0.pf`.
    let embed_dim = first_dim(
        shapes,
        &format!("{}.one2one_cv3.0.2.weight", layout.head_prefix),
    )
    .or_else(|_| first_dim(shapes, &format!("{}.lrpc.0.pf.weight", layout.head_prefix)))?;
    let mask_channels = first_dim(
        shapes,
        &format!("{}.{}.0.2.weight", layout.head_prefix, mask_branch.as_str()),
    )?;
    let proto_channels = first_dim(
        shapes,
        &format!("{}.proto.feat_fuse.conv.weight", layout.head_prefix),
    )?;
    // Intermediate (hidden) widths: read from the checkpoint so the built head
    // matches the official layout exactly, instead of recomputing from scale
    // formulas (e.g. for Scale::N official cv3 hidden is 80, not the 100 a
    // `min(100)` formula yields).
    let cls_hidden = first_dim(
        shapes,
        &format!("{}.one2one_cv3.0.0.1.conv.weight", layout.head_prefix),
    )
    .unwrap_or(0);
    let box_hidden = first_dim(
        shapes,
        &format!("{}.one2one_cv2.0.0.conv.weight", layout.head_prefix),
    )
    .unwrap_or(0);
    let mask_hidden = first_dim(
        shapes,
        &format!(
            "{}.{}.0.0.conv.weight",
            layout.head_prefix,
            mask_branch.as_str()
        ),
    )
    .unwrap_or(0);
    let prompt_head = shapes
        .contains_key(&format!("{}.one2one_cv2.0.2.weight", layout.head_prefix))
        && shapes.contains_key(&format!("{}.one2one_cv3.0.2.weight", layout.head_prefix));
    let savpe_hidden = first_dim(
        shapes,
        &format!("{}.savpe.cv1.0.0.conv.weight", layout.head_prefix),
    )
    .unwrap_or(0);
    Ok(InferredSegmentShapes {
        mask_branch,
        embed_dim,
        mask_channels,
        proto_channels,
        official_lrpc: layout.has_official_lrpc,
        official_savpe: layout.has_official_savpe,
        prompt_head,
        cls_hidden,
        box_hidden,
        mask_hidden,
        savpe_hidden,
    })
}
