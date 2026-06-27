//! Official Ultralytics average-precision interpolation (pure math).

/// Computes the average precision for one precision/recall curve using the
/// official Ultralytics 101-point interpolation.
///
/// `recall` and `precision` are the per-detection cumulative curves for one
/// class at one IoU threshold (both ordered by descending detection score).
/// Returns the area under the precision envelope integrated over `x` sampled at
/// 101 points in `[0, 1]`, matching `ultralytics/utils/metrics.py::compute_ap`.
pub(crate) fn compute_ap(recall: &[f32], precision: &[f32]) -> f32 {
    // mrec = [0.0] + recall + [recall[-1] or 1.0] + [1.0]
    // mpre = [1.0] + precision + [0.0, 0.0]
    let last_rec = recall.last().copied().unwrap_or(1.0) as f64;
    let mut mrec: Vec<f64> = Vec::with_capacity(recall.len() + 3);
    mrec.push(0.0);
    mrec.extend(recall.iter().map(|v| *v as f64));
    mrec.push(last_rec);
    mrec.push(1.0);

    let mut mpre: Vec<f64> = Vec::with_capacity(precision.len() + 3);
    mpre.push(1.0);
    mpre.extend(precision.iter().map(|v| *v as f64));
    mpre.push(0.0);
    mpre.push(0.0);

    // Precision envelope: right-to-left cumulative maximum.
    let n = mpre.len();
    let mut env = mpre.clone();
    for i in (0..n.saturating_sub(1)).rev() {
        if env[i] < env[i + 1] {
            env[i] = env[i + 1];
        }
    }

    // Integrate the interpolated envelope over 101 uniform points in [0, 1].
    let mut area = 0.0f64;
    let mut prev_x = 0.0f64;
    let mut prev_y = interp_at(&mrec, &env, 0.0);
    for k in 1..=100u32 {
        let x = f64::from(k) / 100.0;
        let y = interp_at(&mrec, &env, x);
        area += 0.5 * (x - prev_x) * (y + prev_y);
        prev_x = x;
        prev_y = y;
    }
    area as f32
}

/// One-dimensional linear interpolation matching `numpy.interp`.
///
/// `xp` must be non-decreasing. The interior interval is `[j-1, j]` where `j`
/// is the right-side insertion index (count of `xp` elements `<= x`), clamped
/// to `[1, n-1]`. For a degenerate vertical segment (`xp[j] == xp[j-1]`) the
/// lower value `fp[j-1]` is returned, replicating `numpy.interp`.
fn interp_at(xp: &[f64], fp: &[f64], x: f64) -> f64 {
    let n = xp.len();
    if n == 0 {
        return 0.0;
    }
    if x < xp[0] {
        return fp[0];
    }
    if x > xp[n - 1] {
        return fp[n - 1];
    }
    let mut count_le = 0usize;
    for (idx, value) in xp.iter().enumerate().take(n) {
        if *value <= x {
            count_le = idx + 1;
        } else {
            break;
        }
    }
    let j = count_le.clamp(1, n - 1);
    let dx = xp[j] - xp[j - 1];
    if dx == 0.0 {
        return fp[j - 1];
    }
    fp[j - 1] + (fp[j] - fp[j - 1]) * (x - xp[j - 1]) / dx
}
