//! Nearest-neighbor resampling for segmentation masks.
//!
//! `candle` 0.10 only exposes bilinear `interpolate2d`, which blurs discrete
//! mask labels. Segmentation masks (per-instance, or overlap instance-index
//! maps) must be resampled with nearest-neighbor sampling so labels survive
//! mosaic/affine geometric transforms intact.

use candle_core::Tensor;

/// Resamples a `[N, C, H, W]` mask to `[N, C, out_h, out_w]` using nearest
/// neighbor sampling.
///
/// Output pixel `(oi, oj)` copies source pixel `(oi * H / out_h, oj * W / out_w)`
/// (clamped to the source bounds), matching `cv2.INTER_NEAREST` semantics for
/// label-preserving up/downsampling. The dtype is round-tripped through `f32`.
pub(crate) fn nearest_resample(mask: &Tensor, out_h: usize, out_w: usize) -> crate::Result<Tensor> {
    let dims = mask.dims();
    if dims.len() != 4 || out_h == 0 || out_w == 0 {
        return Err(crate::Error::InvalidTensor(format!(
            "nearest_resample expects a 4D mask and non-zero target size, got dims={dims:?} out=({out_h},{out_w})"
        )));
    }
    let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let device = mask.device();
    let dtype = mask.dtype();
    let src = mask
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let plane = h * w;
    let out_plane = out_h * out_w;
    let mut out = vec![0.0f32; n * c * out_plane];
    let row_idx = (0..out_h)
        .map(|oi| (oi * h / out_h).min(h.saturating_sub(1)))
        .collect::<Vec<_>>();
    let col_idx = (0..out_w)
        .map(|oj| (oj * w / out_w).min(w.saturating_sub(1)))
        .collect::<Vec<_>>();
    for g in 0..n * c {
        let src_base = g * plane;
        let dst_base = g * out_plane;
        for (oi, &src_row) in row_idx.iter().enumerate().take(out_h) {
            let s_row = src_row * w;
            let d_row = oi * out_w;
            for (oj, &src_col) in col_idx.iter().enumerate().take(out_w) {
                out[dst_base + d_row + oj] = src[src_base + s_row + src_col];
            }
        }
    }
    Ok(Tensor::from_vec(out, (n, c, out_h, out_w), device)?.to_dtype(dtype)?)
}
