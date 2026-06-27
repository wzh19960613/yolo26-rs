//! Official-aligned box regression losses for YOLO26 detection.
//!
//! YOLO26's `Model` head uses `reg_max = 1` (direct l/t/r/b regression, no
//! DFL distribution), which is exactly what the official `v8DetectionLoss` /
//! `BboxLoss` runs when `reg_max > 1` is false: `use_dfl = False` and
//! `bbox_loss.dfl_loss = None`. In that regime the official `BboxLoss.forward`
//! returns `(loss_iou, loss_dfl)` where:
//!
//! - `loss_iou = ((1 - CIoU(pred, target)) * weight).sum() / target_scores_sum`
//!   using `bbox_iou(..., xywh=False, CIoU=True)`.
//! - `loss_dfl = (F.l1_loss(pred_dist*stride/imgsz, target_ltrb*stride/imgsz,
//!   reduction="none").mean(-1, keepdim=True) * weight).sum() /
//!   target_scores_sum` (the `else` branch of `BboxLoss.forward`, taken when
//!   `dfl_loss is None`).
//!
//! Both are normalized by `target_scores_sum`, matching the official
//! `BboxLoss`/`v8DetectionLoss` semantics.

use super::*;

/// Computes the CIoU between xyxy boxes for every foreground anchor.
///
/// Mirrors `ultralytics.utils.metrics.bbox_iou(..., xywh=False, CIoU=True)`.
/// The returned tensor has the same shape as the per-anchor IoU broadcast.
pub(crate) fn xyxy_ciou(pred: &Tensor, target: &Tensor) -> crate::Result<Tensor> {
    let device = pred.device();
    let dtype = pred.dtype();
    let eps = Tensor::new(1e-7f32, device)?.to_dtype(dtype)?;
    let one = Tensor::new(1f32, device)?.to_dtype(dtype)?;

    let px1 = pred.narrow(1, 0, 1)?;
    let py1 = pred.narrow(1, 1, 1)?;
    let px2 = pred.narrow(1, 2, 1)?;
    let py2 = pred.narrow(1, 3, 1)?;
    let tx1 = target.narrow(1, 0, 1)?;
    let ty1 = target.narrow(1, 1, 1)?;
    let tx2 = target.narrow(1, 2, 1)?;
    let ty2 = target.narrow(1, 3, 1)?;

    // Width/height of both boxes, matching Ultralytics' xyxy branch exactly:
    // width is raw `x2 - x1`, while height gets the epsilon.
    let pw = px2.broadcast_sub(&px1)?;
    let ph = py2.broadcast_sub(&py1)?.broadcast_add(&eps)?;
    let tw = tx2.broadcast_sub(&tx1)?;
    let th = ty2.broadcast_sub(&ty1)?.broadcast_add(&eps)?;

    // Intersection.
    let inter_w = px2
        .minimum(&tx2)?
        .broadcast_sub(&px1.maximum(&tx1)?)?
        .maximum(0f32)?;
    let inter_h = py2
        .minimum(&ty2)?
        .broadcast_sub(&py1.maximum(&ty1)?)?
        .maximum(0f32)?;
    let inter = inter_w.broadcast_mul(&inter_h)?;
    let union = pw
        .broadcast_mul(&ph)?
        .broadcast_add(&tw.broadcast_mul(&th)?)?
        .broadcast_sub(&inter)?
        .broadcast_add(&eps)?;
    let iou = inter.broadcast_div(&union)?;

    // Convex enclosing box (smallest enclosing rectangle).
    let cw = px2.maximum(&tx2)?.broadcast_sub(&px1.minimum(&tx1)?)?;
    let ch = py2.maximum(&ty2)?.broadcast_sub(&py1.minimum(&ty1)?)?;
    let c2 = cw
        .broadcast_mul(&cw)?
        .broadcast_add(&ch.broadcast_mul(&ch)?)?
        .broadcast_add(&eps)?;

    // Center-point distance squared.
    let dx = tx1
        .broadcast_add(&tx2)?
        .broadcast_sub(&px1.broadcast_add(&px2)?)?;
    let dy = ty1
        .broadcast_add(&ty2)?
        .broadcast_sub(&py1.broadcast_add(&py2)?)?;
    let rho2 = dx
        .broadcast_mul(&dx)?
        .broadcast_add(&dy.broadcast_mul(&dy)?)?
        .affine(0.25, 0.0)?;

    // Aspect-ratio consistency term. `atan` is provided by a custom candle op
    // (`crate::train::atan_op::atan`), since candle 0.10.2 has no built-in atan.
    let pi = Tensor::new(std::f32::consts::PI, device)?.to_dtype(dtype)?;
    let four_over_pi2 = (Tensor::new(4f32, device)?.to_dtype(dtype)? / pi.broadcast_mul(&pi)?)?;
    let atan_w1 = crate::train::atan_op::atan(&pw.broadcast_div(&ph)?)?;
    let atan_w2 = crate::train::atan_op::atan(&tw.broadcast_div(&th)?)?;
    let v = four_over_pi2.broadcast_mul(&atan_w2.broadcast_sub(&atan_w1)?.sqr()?)?;
    // alpha is detached (no gradient), matching official `with torch.no_grad()`.
    let v_det = v.detach();
    let alpha =
        v_det.broadcast_div(&(v_det.broadcast_sub(&iou)?.broadcast_add(&(&one + &eps)?)?))?;
    let penalty = rho2
        .broadcast_div(&c2)?
        .broadcast_add(&v.broadcast_mul(&alpha)?)?;
    iou.broadcast_sub(&penalty).map_err(Into::into)
}

/// Computes the official `loss_iou` = `sum((1 - CIoU) * weight) / target_scores_sum`.
///
/// `weight` is the per-anchor target-score sum (foreground only), matching the
/// official `weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)`.
pub(crate) fn ciou_loss(
    pred_xyxy: &Tensor,
    target_xyxy: &Tensor,
    weight: &Tensor,
    target_scores_sum: f64,
) -> crate::Result<Tensor> {
    let device = pred_xyxy.device();
    let one = Tensor::new(1f32, device)?.to_dtype(pred_xyxy.dtype())?;
    let ciou = xyxy_ciou(pred_xyxy, target_xyxy)?;
    let loss_term = one.broadcast_sub(&ciou)?;
    let loss = loss_term.broadcast_mul(weight)?.sum_all()?;
    Ok(loss.broadcast_div(&Tensor::new(target_scores_sum.max(1.0) as f32, device)?)?)
}

/// Computes the official `loss_dfl` for the `reg_max == 1` (no DFL) regime.
///
/// Mirrors the `else` branch of `BboxLoss.forward`: both predicted and target
/// l/t/r/b (in stride units) are scaled to pixels by `stride` then normalized
/// by the image size, the elementwise L1 is averaged over the 4 coords, and the
/// per-anchor mean is weighted and summed / `target_scores_sum`.
pub(crate) fn normalized_l1_dfl_loss(
    pred_stride_units: &Tensor,
    target_stride_units: &Tensor,
    stride_tensor: &Tensor,
    image_width: f32,
    image_height: f32,
    weight: &Tensor,
    target_scores_sum: f64,
) -> crate::Result<Tensor> {
    let device = pred_stride_units.device();
    let dtype = pred_stride_units.dtype();
    let pred_pixels = pred_stride_units.broadcast_mul(stride_tensor)?;
    let target_pixels = target_stride_units.broadcast_mul(stride_tensor)?;
    let scale = scale_xyxy_per_coord(image_width, image_height, device, dtype)?;
    let pred_norm = pred_pixels.broadcast_mul(&scale)?;
    let target_norm = target_pixels.broadcast_mul(&scale)?;
    // elementwise L1, mean over the 4 coordinate channels (dim 1), keepdim so
    // the per-anchor result broadcasts against `weight` ([batch, 1, anchors]).
    let l1 = pred_norm
        .broadcast_sub(&target_norm)?
        .abs()?
        .mean_keepdim(1)?;
    let loss = l1.broadcast_mul(weight)?.sum_all()?;
    Ok(loss.broadcast_div(&Tensor::new(target_scores_sum.max(1.0) as f32, device)?)?)
}

/// Builds the per-coordinate normalization scale `[1/w, 1/h, 1/w, 1/h]` as a
/// `[1, 4, 1]` tensor so it broadcasts over batch and anchors exactly like the
/// official in-place `pred_dist[..., 0::2] /= imgsz[1]; pred_dist[..., 1::2] /= imgsz[0]`.
fn scale_xyxy_per_coord(
    image_width: f32,
    image_height: f32,
    device: &Device,
    dtype: DType,
) -> crate::Result<Tensor> {
    let w = image_width.max(1.0);
    let h = image_height.max(1.0);
    let scale = vec![1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h];
    Ok(Tensor::from_vec(scale, (1, 4, 1), device)?.to_dtype(dtype)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ciou_loss_is_not_negative_for_identical_boxes() -> crate::Result<()> {
        let device = Device::Cpu;
        let pred = Tensor::from_vec(vec![0.0f32, 0.0, 10.0, 10.0], (1, 4, 1), &device)?;
        let target = Tensor::from_vec(vec![0.0f32, 0.0, 10.0, 10.0], (1, 4, 1), &device)?;
        let weight = Tensor::from_vec(vec![1.0f32], (1, 1, 1), &device)?;

        let loss = ciou_loss(&pred, &target, &weight, 1.0)?.to_scalar::<f32>()?;

        assert!(loss >= 0.0, "CIoU loss must be non-negative, got {loss}");
        Ok(())
    }
}
