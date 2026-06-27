//! Identity shape inference for the YOLOE detection-only head config.
//!
//! Mirrors [`crate::yoloe::segment::model::shape`]: reads the official
//! SafeTensors/`.pt` tensor shapes to derive the detect head's embed dimension,
//! LRPC/SAVPE availability flags, and SAVPE hidden width. Model checkpoints
//! omit mask/prototype tensors, so only the box/embedding/SAVPE families are
//! inspected.

use std::collections::BTreeMap;

use crate::yoloe::checkpoint::layout::Layout;
use crate::yoloe::first_dim::first_dim;

/// Fields inferred from checkpoint tensor shapes, used to construct a
/// [`super::Config`].
pub(crate) struct InferredDetectShapes {
    /// Region/prompt embedding dimension.
    pub(crate) embed_dim: usize,
    /// Whether official prompt-free LRPC heads are available.
    pub(crate) official_lrpc: bool,
    /// Whether official SAVPE visual-prompt encoder weights are available.
    pub(crate) official_savpe: bool,
    /// Intermediate width of the official SAVPE encoder.
    pub(crate) savpe_hidden: usize,
    /// Whether the regular text/visual prompt head final projections exist.
    pub(crate) prompt_head: bool,
    /// Intermediate width of the classification/embedding branch `cv3`.
    pub(crate) cls_hidden: usize,
    /// Intermediate width of the box-regression branch `cv2`.
    pub(crate) box_hidden: usize,
}

/// Infers the detect head layout from checkpoint tensor shapes.
pub(crate) fn infer_detect_shapes(
    shapes: &BTreeMap<String, Vec<usize>>,
    layout: &Layout,
) -> crate::Result<InferredDetectShapes> {
    // embed_dim: the text/visual checkpoint exposes it as the projection
    // `one2one_cv3.0.2.weight`; the prompt-free (`-pf`) checkpoint has no such
    // projection, so fall back to the LRPC prompt-free conv `lrpc.0.pf`.
    let embed_dim = first_dim(
        shapes,
        &format!("{}.one2one_cv3.0.2.weight", layout.head_prefix),
    )
    .or_else(|_| first_dim(shapes, &format!("{}.lrpc.0.pf.weight", layout.head_prefix)))?;
    let savpe_hidden = first_dim(
        shapes,
        &format!("{}.savpe.cv1.0.0.conv.weight", layout.head_prefix),
    )
    .unwrap_or(0);
    let prompt_head = shapes
        .contains_key(&format!("{}.one2one_cv2.0.2.weight", layout.head_prefix))
        && shapes.contains_key(&format!("{}.one2one_cv3.0.2.weight", layout.head_prefix));
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
    Ok(InferredDetectShapes {
        embed_dim,
        official_lrpc: layout.has_official_lrpc,
        official_savpe: layout.has_official_savpe,
        savpe_hidden,
        prompt_head,
        cls_hidden,
        box_hidden,
    })
}
