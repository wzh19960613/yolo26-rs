//! Pose keypoint left/right pairing for horizontal flips.
//!
//! Mirroring x alone would put a "left shoulder" keypoint on the right side of
//! the image while keeping its semantic label. The official Ultralytics pose
//! flip additionally permutes the keypoint channel with `flip_indices` so that
//! each symmetric pair swaps label along with position, then mirrors x. This
//! module owns that permutation table and the keypoint flip operator.

use candle_core::Tensor;

/// Official COCO 17-keypoint horizontal-flip permutation.
///
/// Index 0 (nose) is self-paired; the remaining sixteen keypoints form eight
/// left/right pairs `(1,2),(3,4),...,(15,16)` that swap. After this permutation
/// position `i` holds the data that was at position `flip_indices[i]`.
const COCO_17: [usize; 17] = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15];

/// Returns the flip permutation for `num_keypoints`.
///
/// Matches the official COCO 17-keypoint `flip_indices` when the keypoint count
/// is 17; otherwise returns the identity permutation (x-only mirror) so other
/// keypoint layouts are not silently corrupted.
pub(crate) fn flip_indices(num_keypoints: usize) -> Vec<usize> {
    if num_keypoints == COCO_17.len() {
        COCO_17.to_vec()
    } else {
        (0..num_keypoints).collect()
    }
}

/// Permutes the keypoint channel by `perm` and mirrors x (`do_lr`) / y
/// (`do_ud`) in place, matching the official pose flip order.
///
/// `keypoints` is shaped `[..., num_keypoints, 2]` with stride-2 `(x, y)` pairs.
/// The permutation is applied per leading group (batch/object) so each object's
/// keypoints are reordered independently. When `perm.len()` does not match the
/// keypoint count the permutation is skipped (x/y mirror still applies).
pub(crate) fn flip_keypoints(
    keypoints: &Tensor,
    do_lr: bool,
    do_ud: bool,
    width: f32,
    height: f32,
    perm: &[usize],
) -> crate::Result<Tensor> {
    let shape = keypoints.dims();
    let mut flat = keypoints
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let num_kp = shape[shape.len() - 2];
    let pairs_per_group = num_kp * 2;
    let num_groups = flat.len() / pairs_per_group;
    let use_perm = perm.len() == num_kp;
    for g in 0..num_groups {
        let base = g * pairs_per_group;
        let mut row = flat[base..base + pairs_per_group].to_vec();
        if use_perm {
            let mut reordered = vec![0.0f32; pairs_per_group];
            for k in 0..num_kp {
                let src = perm[k];
                reordered[k * 2] = row[src * 2];
                reordered[k * 2 + 1] = row[src * 2 + 1];
            }
            row = reordered;
        }
        for k in 0..num_kp {
            if do_lr {
                row[k * 2] = width - row[k * 2];
            }
            if do_ud {
                row[k * 2 + 1] = height - row[k * 2 + 1];
            }
        }
        flat[base..base + pairs_per_group].copy_from_slice(&row);
    }
    Ok(Tensor::from_vec(flat, shape, keypoints.device())?.to_dtype(keypoints.dtype())?)
}
