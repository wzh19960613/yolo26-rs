use candle_core::{DType, Device, Result, Tensor};

pub fn make_anchors(
    feat_sizes: &[(usize, usize)],
    strides: &[f32; 3],
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let anchor_count = feat_sizes.iter().map(|(h, w)| h * w).sum::<usize>();
    let mut anchor_data = Vec::with_capacity(anchor_count * 2);
    let mut stride_data = Vec::with_capacity(anchor_count);

    for (i, &(h, w)) in feat_sizes.iter().enumerate() {
        for y in 0..h {
            for x in 0..w {
                anchor_data.push(x as f32 + 0.5);
                anchor_data.push(y as f32 + 0.5);
                stride_data.push(strides[i]);
            }
        }
    }

    let n = stride_data.len();
    let anchors = Tensor::from_vec(anchor_data, (n, 2), device)?
        .to_dtype(dtype)?
        .transpose(0, 1)?
        .unsqueeze(0)?;
    let stride_tensor = Tensor::from_vec(stride_data, (1, 1, n), device)?.to_dtype(dtype)?;
    Ok((anchors, stride_tensor))
}

pub fn dist2bbox_xyxy(distance: &Tensor, anchors: &Tensor) -> Result<Tensor> {
    let lt = distance.narrow(1, 0, 2)?;
    let rb = distance.narrow(1, 2, 2)?;
    let x1y1 = anchors.broadcast_sub(&lt)?;
    let x2y2 = anchors.broadcast_add(&rb)?;
    Tensor::cat(&[&x1y1, &x2y2], 1)
}

pub fn dist2rbox_xywh(distance: &Tensor, angle: &Tensor, anchors: &Tensor) -> Result<Tensor> {
    let lt = distance.narrow(1, 0, 2)?;
    let rb = distance.narrow(1, 2, 2)?;
    let offset = rb
        .broadcast_sub(&lt)?
        .broadcast_div(&Tensor::new(2f32, distance.device())?.to_dtype(distance.dtype())?)?;
    let xf = offset.narrow(1, 0, 1)?;
    let yf = offset.narrow(1, 1, 1)?;
    let cos = angle.cos()?;
    let sin = angle.sin()?;
    let x = xf
        .broadcast_mul(&cos)?
        .broadcast_sub(&yf.broadcast_mul(&sin)?)?;
    let y = xf
        .broadcast_mul(&sin)?
        .broadcast_add(&yf.broadcast_mul(&cos)?)?;
    let xy = Tensor::cat(&[&x, &y], 1)?.broadcast_add(anchors)?;
    let wh = lt.broadcast_add(&rb)?;
    Tensor::cat(&[&xy, &wh], 1)
}
