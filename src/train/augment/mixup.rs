//! Two-image mixup blend for box-bearing tasks.
//!
//! The image becomes `lam * primary + (1 - lam) * other` with `lam` drawn from
//! `rng`; the boxes/classes and task-specific extras (pose keypoints+visibility,
//! OBB angles, per-instance masks) of both samples are merged at full canvas
//! (identity affine, canvas-clamped) and truncated to the per-sample
//! `max_objects` slot count. Overlap-encoded segmentation and non-box targets
//! are skipped (returned unchanged) since their targets cannot be concatenated.

use super::affine::AffinePlan;
use super::mosaic::build_detection;
use super::mosaic::mosaic_max_objects;
use super::mosaic_target::MosaicExtras;
use super::{Sample, SeededRng};

/// Identity affine plan: no scale/translate (boxes keep canvas coordinates).
const IDENTITY: AffinePlan = AffinePlan {
    s_w: 1.0,
    s_h: 1.0,
    dx: 0.0,
    dy: 0.0,
};

/// Blends `primary` with `other` via mixup, merging both targets.
///
/// `rng` draws the blend weight from the official `Beta(32, 32)` distribution
/// (concentrated around 0.5), matching `np.random.beta` used by Ultralytics.
pub(crate) fn compose_mixup(
    mut primary: Sample,
    other: &Sample,
    rng: &mut SeededRng,
) -> crate::Result<Sample> {
    let mut extras = match MosaicExtras::new(&primary.target) {
        Ok(extras) => extras,
        // Non-box or overlap-encoded targets cannot be merged: skip mixup.
        Err(_) => return Ok(primary),
    };
    let max_objects = mosaic_max_objects(&primary.target)?;
    let width = primary.input.dim(3)? as f32;
    let height = primary.input.dim(2)? as f32;
    let device = primary.input.device().clone();

    let mut all_boxes: Vec<[f32; 4]> = Vec::new();
    let mut all_classes: Vec<u32> = Vec::new();
    extras.collect_sample(
        &primary,
        IDENTITY,
        width,
        height,
        &mut all_boxes,
        &mut all_classes,
    )?;
    extras.collect_sample(
        other,
        IDENTITY,
        width,
        height,
        &mut all_boxes,
        &mut all_classes,
    )?;

    // Official MixUp samples the blend weight from Beta(32, 32) (concentrated
    // around 0.5) and concatenates the labels.
    let lam = rng.beta(32, 32);
    let blended = ((&primary.input * lam)? + (&other.input * (1.0 - lam)))?;
    let take = all_boxes.len().min(max_objects);
    let detection = build_detection(&all_boxes, &all_classes, take, max_objects, &device)?;
    primary.input = blended;
    primary.target = extras.finish(detection, take, &device)?;
    Ok(primary)
}
