use candle_core::{DType, Tensor};

use super::{EvalPrediction, SegmentationMaskEncoding};

pub(crate) fn prediction_masks(
    preds: &[EvalPrediction],
    coeffs: &[Vec<f32>],
    proto: &[Vec<Vec<f32>>],
    input_hw: (usize, usize),
) -> crate::Result<Vec<Vec<f32>>> {
    let (mask_dim, mask_h, mask_w) = proto_dims(proto)?;
    let mut out = Vec::with_capacity(preds.len());
    for pred in preds {
        let mut mask = vec![0f32; mask_h * mask_w];
        for c in 0..mask_dim {
            let coeff = coeffs[c][pred.anchor_idx];
            for y in 0..mask_h {
                for x in 0..mask_w {
                    mask[y * mask_w + x] += coeff * proto[c][y][x];
                }
            }
        }
        crop_and_threshold(&mut mask, mask_h, mask_w, input_hw, pred.xyxy);
        out.push(mask);
    }
    Ok(out)
}

pub(crate) fn prediction_masks_from_tensors(
    preds: &[EvalPrediction],
    mask_coefficients: &Tensor,
    proto: &Tensor,
    batch_idx: usize,
    input_hw: (usize, usize),
) -> crate::Result<Vec<Vec<f32>>> {
    if preds.is_empty() {
        return Ok(Vec::new());
    }
    let (_, proto_dim, mask_h, mask_w) = proto.dims4()?;
    let pixels = mask_h * mask_w;
    let anchor_indices = preds
        .iter()
        .map(|pred| pred.anchor_idx as u32)
        .collect::<Vec<_>>();
    let index_tensor = Tensor::new(anchor_indices, mask_coefficients.device())?;
    let selected_coeffs =
        selected_anchor_coefficients(mask_coefficients, batch_idx, proto_dim, &index_tensor)?
            .to_dtype(DType::F32)?
            .contiguous()?;
    let proto_b = proto
        .narrow(0, batch_idx, 1)?
        .squeeze(0)?
        .to_dtype(DType::F32)?
        .reshape((proto_dim, pixels))?
        .contiguous()?;
    if selected_coeffs.dims() != [preds.len(), proto_dim] {
        return Err(crate::Error::InvalidTensor(format!(
            "selected segmentation coefficients must have shape [{}, {proto_dim}], got {:?}",
            preds.len(),
            selected_coeffs.dims()
        )));
    }
    let decoded = selected_coeffs.matmul(&proto_b)?;
    let mut out = decoded.to_vec2::<f32>()?;
    for (mask, pred) in out.iter_mut().zip(preds) {
        crop_and_threshold(mask, mask_h, mask_w, input_hw, pred.xyxy);
    }
    Ok(out)
}

fn selected_anchor_coefficients(
    mask_coefficients: &Tensor,
    batch_idx: usize,
    proto_dim: usize,
    index_tensor: &Tensor,
) -> crate::Result<Tensor> {
    let (_, dim_1, dim_2) = mask_coefficients.dims3()?;
    let coeffs = mask_coefficients.narrow(0, batch_idx, 1)?.squeeze(0)?;
    if dim_1 == proto_dim {
        coeffs.index_select(index_tensor, 1)?.transpose(0, 1)
    } else if dim_2 == proto_dim {
        coeffs.index_select(index_tensor, 0)
    } else {
        Err(candle_core::Error::Msg(format!(
            "segmentation mask coefficients must be [batch, channels, anchors] or [batch, anchors, channels], got second dim {dim_1}, third dim {dim_2}, proto channels {proto_dim}"
        )))
    }
    .map_err(Into::into)
}

pub(crate) fn target_masks_for_image(
    masks: &[Vec<Vec<f32>>],
    encoding: SegmentationMaskEncoding,
    objects: usize,
    (_mask_dim, mask_h, mask_w): (usize, usize, usize),
) -> crate::Result<Vec<Vec<f32>>> {
    let pixels = mask_h * mask_w;
    let mut out = Vec::with_capacity(objects);
    match encoding {
        SegmentationMaskEncoding::PerInstance => {
            for obj in 0..objects {
                let mut plane = vec![0f32; pixels];
                if let Some(src) = masks.get(obj) {
                    flatten_binary(src, &mut plane);
                }
                out.push(plane);
            }
        }
        SegmentationMaskEncoding::Overlap => {
            let src = masks.first().ok_or_else(|| {
                crate::Error::InvalidTensor("overlap mask tensor has no channel".to_string())
            })?;
            for obj in 0..objects {
                out.push(overlap_plane(src, (obj + 1) as f32, mask_h, mask_w));
            }
        }
    }
    Ok(out)
}

pub(crate) fn proto_dims(proto: &[Vec<Vec<f32>>]) -> crate::Result<(usize, usize, usize)> {
    let mask_dim = proto.len();
    let mask_h = proto.first().map(Vec::len).unwrap_or(0);
    let mask_w = proto
        .first()
        .and_then(|channel| channel.first())
        .map(Vec::len)
        .unwrap_or(0);
    if mask_dim == 0 || mask_h == 0 || mask_w == 0 {
        return Err(crate::Error::InvalidTensor(
            "segmentation proto must have non-empty [channels, height, width]".to_string(),
        ));
    }
    Ok((mask_dim, mask_h, mask_w))
}

fn crop_and_threshold(
    mask: &mut [f32],
    mask_h: usize,
    mask_w: usize,
    (input_h, input_w): (usize, usize),
    xyxy: [f32; 4],
) {
    let width_ratio = mask_w as f32 / input_w.max(1) as f32;
    let height_ratio = mask_h as f32 / input_h.max(1) as f32;
    let x1 = crop_coord(xyxy[0] * width_ratio, mask_w);
    let y1 = crop_coord(xyxy[1] * height_ratio, mask_h);
    let x2 = crop_coord(xyxy[2] * width_ratio, mask_w);
    let y2 = crop_coord(xyxy[3] * height_ratio, mask_h);
    for y in 0..mask_h {
        for x in 0..mask_w {
            let keep = x >= x1 && x < x2 && y >= y1 && y < y2 && mask[y * mask_w + x] > 0.0;
            mask[y * mask_w + x] = keep as u8 as f32;
        }
    }
}

fn crop_coord(value: f32, limit: usize) -> usize {
    value.max(0.0).round().clamp(0.0, limit as f32) as usize
}

fn flatten_binary(src: &[Vec<f32>], dst: &mut [f32]) {
    let width = src.first().map(Vec::len).unwrap_or(0);
    for (y, row) in src.iter().enumerate() {
        for (x, &value) in row.iter().enumerate() {
            let idx = y * width + x;
            if let Some(slot) = dst.get_mut(idx) {
                *slot = (value > 0.5) as u8 as f32;
            }
        }
    }
}

fn overlap_plane(src: &[Vec<f32>], expected: f32, mask_h: usize, mask_w: usize) -> Vec<f32> {
    let mut out = vec![0f32; mask_h * mask_w];
    for y in 0..mask_h {
        for x in 0..mask_w {
            out[y * mask_w + x] = (src[y][x] == expected) as u8 as f32;
        }
    }
    out
}
