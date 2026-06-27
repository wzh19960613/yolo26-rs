use super::*;

pub(crate) fn instance_mask_loss(
    mask_coefficients: &Tensor,
    proto: &Tensor,
    targets: &SegmentationTargets,
    built: &BuiltDetectionTargets,
) -> crate::Result<Tensor> {
    let (batch, coeff_dim_1, coeff_dim_2) = mask_coefficients.dims3()?;
    let (proto_batch, proto_dim, proto_h, proto_w) = proto.dims4()?;
    if proto_batch != batch {
        return Err(crate::Error::InvalidTensor(
            "segmentation mask coefficients and proto batch dimensions do not match".to_string(),
        ));
    }
    let detection_anchors_len = built
        .target_gt_idx
        .len()
        .checked_div(batch)
        .ok_or_else(|| {
            crate::Error::InvalidTensor(
                "segmentation target assignments have invalid batch size".to_string(),
            )
        })?;
    let coeff_layout = MaskCoefficientLayout::from_dims(
        coeff_dim_1,
        coeff_dim_2,
        proto_dim,
        detection_anchors_len,
    )?;
    let anchors_len = detection_anchors_len;
    let (mask_h, mask_w) = validate_target_mask_shape(targets, batch)?;
    let proto = if (proto_h, proto_w) == (mask_h, mask_w) {
        proto.clone()
    } else {
        proto.upsample_bilinear2d(mask_h, mask_w, false)?
    };
    let pixels = mask_h * mask_w;
    let positives = positive_mask_assignments(built, batch, anchors_len);
    if positives.is_empty() {
        return Tensor::new(0f32, proto.device()).map_err(Into::into);
    }

    let target_masks =
        TargetMaskData::from_targets(&targets.masks, targets.mask_encoding, mask_h, mask_w)?;
    let boxes = built.target_xyxy.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let image_w = built.image_width.max(1.0);
    let image_h = built.image_height.max(1.0);
    let device = proto.device();
    let mut positives_per_image = vec![0usize; batch];
    for positive in &positives {
        positives_per_image[positive.batch_idx] += 1;
    }
    let mut total: Option<Tensor> = None;

    for (batch_idx, _) in positives_per_image.iter().enumerate().take(batch) {
        let image_positives = positives
            .iter()
            .copied()
            .filter(|positive| positive.batch_idx == batch_idx)
            .collect::<Vec<_>>();
        if image_positives.is_empty() {
            continue;
        }
        let round_crop = image_positives.len() < 50 && !device.is_cuda();
        let proto_b = proto
            .narrow(0, batch_idx, 1)?
            .squeeze(0)?
            .reshape((proto_dim, pixels))?
            .contiguous()?;
        if proto_b.dims() != [proto_dim, pixels] {
            return Err(crate::Error::InvalidTensor(format!(
                "segmentation mask matmul expects proto [{proto_dim}, {pixels}], got {:?}",
                proto_b.dims()
            )));
        }

        let mut coeffs = Vec::with_capacity(image_positives.len());
        let mut target_flat = Vec::with_capacity(image_positives.len() * pixels);
        let mut crop_flat = Vec::with_capacity(image_positives.len() * pixels);
        let mut areas = Vec::with_capacity(image_positives.len());

        for positive in &image_positives {
            coeffs.push(
                coeff_layout
                    .anchor_coefficients(mask_coefficients, batch_idx, positive.anchor_idx)?
                    .contiguous()?,
            );
            target_flat.extend(target_masks.mask_for(batch_idx, positive.object_idx)?);
            let xyxy = [
                boxes[batch_idx][0][positive.anchor_idx],
                boxes[batch_idx][1][positive.anchor_idx],
                boxes[batch_idx][2][positive.anchor_idx],
                boxes[batch_idx][3][positive.anchor_idx],
            ];
            let (crop_mask, area_pixels) =
                positive_crop_mask_and_area(xyxy, image_w, image_h, mask_h, mask_w, round_crop);
            crop_flat.extend(crop_mask);
            areas.push(area_pixels);
        }

        let coeff_refs = coeffs.iter().collect::<Vec<_>>();
        let coeff = Tensor::cat(&coeff_refs, 0)?;
        let positives_count = image_positives.len();
        if coeff.dims() != [positives_count, proto_dim] {
            return Err(crate::Error::InvalidTensor(format!(
                "segmentation mask matmul expects coeff [{positives_count}, {proto_dim}], got {:?}",
                coeff.dims()
            )));
        }
        let pred_mask = coeff.matmul(&proto_b)?;
        let target_mask = Tensor::from_vec(target_flat, (positives_count, pixels), device)?
            .to_dtype(pred_mask.dtype())?;
        let crop_mask = Tensor::from_vec(crop_flat, (positives_count, pixels), device)?
            .to_dtype(pred_mask.dtype())?;
        let area_pixels =
            Tensor::from_vec(areas, positives_count, device)?.to_dtype(pred_mask.dtype())?;
        let loss = bce_with_logits_elementwise(&pred_mask, &target_mask)?
            .broadcast_mul(&crop_mask)?
            .sum(1)?
            .broadcast_div(&area_pixels)?
            .sum_all()?;
        total = Some(match total {
            Some(total) => (total + loss)?,
            None => loss,
        });
    }

    let Some(total) = total else {
        return Tensor::new(0f32, device).map_err(Into::into);
    };
    let denom = Tensor::new(built.foreground_count.max(1.0) as f32, device)?;
    total.broadcast_div(&denom).map_err(Into::into)
}

fn validate_target_mask_shape(
    targets: &SegmentationTargets,
    batch: usize,
) -> crate::Result<(usize, usize)> {
    let dims = targets.masks.dims();
    if dims.len() != 4 || dims[0] != batch {
        return Err(crate::Error::InvalidTensor(format!(
            "segmentation target masks must have shape [batch, masks, H, W], got {dims:?}"
        )));
    }
    match targets.mask_encoding {
        SegmentationMaskEncoding::PerInstance => {
            if dims[1] != targets.detection.boxes_xyxy.dim(1)? {
                return Err(crate::Error::InvalidTensor(format!(
                    "segmentation target masks must have one mask per object, got {dims:?}"
                )));
            }
        }
        SegmentationMaskEncoding::Overlap => {
            if dims[1] != 1 {
                return Err(crate::Error::InvalidTensor(format!(
                    "overlap segmentation masks must have one map channel, got {dims:?}"
                )));
            }
        }
    }
    Ok((dims[2], dims[3]))
}

#[derive(Clone, Copy)]
enum MaskCoefficientLayout {
    ChannelAnchor { anchors_len: usize },
    AnchorChannel { anchors_len: usize },
}

impl MaskCoefficientLayout {
    fn from_dims(
        dim_1: usize,
        dim_2: usize,
        proto_dim: usize,
        anchors_len: usize,
    ) -> crate::Result<Self> {
        if dim_1 == proto_dim && dim_2 == anchors_len {
            Ok(Self::ChannelAnchor { anchors_len: dim_2 })
        } else if dim_1 == anchors_len && dim_2 == proto_dim {
            Ok(Self::AnchorChannel { anchors_len: dim_1 })
        } else {
            Err(crate::Error::InvalidTensor(format!(
                "segmentation mask coefficients must be [batch, channels, anchors] or [batch, anchors, channels], got second dim {dim_1}, third dim {dim_2}, proto channels {proto_dim}, detection anchors {anchors_len}"
            )))
        }
    }

    fn anchors_len(self) -> usize {
        match self {
            Self::ChannelAnchor { anchors_len } | Self::AnchorChannel { anchors_len } => {
                anchors_len
            }
        }
    }

    fn anchor_coefficients(
        self,
        mask_coefficients: &Tensor,
        batch_idx: usize,
        anchor_idx: usize,
    ) -> crate::Result<Tensor> {
        let coeff = mask_coefficients.narrow(0, batch_idx, 1)?.squeeze(0)?;
        match self {
            Self::ChannelAnchor { .. } => coeff.narrow(1, anchor_idx, 1)?.transpose(0, 1),
            Self::AnchorChannel { .. } => coeff.narrow(0, anchor_idx, 1),
        }
        .map_err(Into::into)
    }
}

#[derive(Clone, Copy)]
struct PositiveMaskAssignment {
    batch_idx: usize,
    anchor_idx: usize,
    object_idx: usize,
}

fn positive_mask_assignments(
    built: &BuiltDetectionTargets,
    batch: usize,
    anchors_len: usize,
) -> Vec<PositiveMaskAssignment> {
    let mut positives = Vec::with_capacity(built.foreground_count as usize);
    for b in 0..batch {
        for anchor_idx in 0..anchors_len {
            let object_idx = built.target_gt_idx[b * anchors_len + anchor_idx];
            if object_idx == usize::MAX {
                continue;
            }
            positives.push(PositiveMaskAssignment {
                batch_idx: b,
                anchor_idx,
                object_idx,
            });
        }
    }
    positives
}

fn positive_crop_mask_and_area(
    xyxy: [f32; 4],
    image_w: f32,
    image_h: f32,
    mask_h: usize,
    mask_w: usize,
    round_crop: bool,
) -> (Vec<f32>, f32) {
    let pixels = mask_h * mask_w;
    let mut crop = vec![0f32; pixels];

    let x1 = xyxy[0];
    let y1 = xyxy[1];
    let x2 = xyxy[2];
    let y2 = xyxy[3];
    let area = ((x2 - x1).max(0.0) / image_w) * ((y2 - y1).max(0.0) / image_h);
    let area_pixels = area.max(1e-7) * pixels as f32;

    let mx1 = (x1 / image_w * mask_w as f32).clamp(0.0, mask_w as f32);
    let my1 = (y1 / image_h * mask_h as f32).clamp(0.0, mask_h as f32);
    let mx2 = (x2 / image_w * mask_w as f32).clamp(0.0, mask_w as f32);
    let my2 = (y2 / image_h * mask_h as f32).clamp(0.0, mask_h as f32);
    if round_crop {
        let (mx1, my1, mx2, my2) = (
            mx1.round() as usize,
            my1.round() as usize,
            mx2.round() as usize,
            my2.round() as usize,
        );
        for y in my1..my2 {
            let row = y * mask_w;
            crop[row + mx1..row + mx2].fill(1.0);
        }
    } else {
        for y in 0..mask_h {
            let yf = y as f32;
            if yf < my1 || yf >= my2 {
                continue;
            }
            let row = y * mask_w;
            for x in 0..mask_w {
                let xf = x as f32;
                if xf >= mx1 && xf < mx2 {
                    crop[row + x] = 1.0;
                }
            }
        }
    }

    (crop, area_pixels)
}

struct TargetMaskData {
    data: Vec<f32>,
    batch: usize,
    objects: usize,
    pixels: usize,
    encoding: SegmentationMaskEncoding,
}

impl TargetMaskData {
    fn from_targets(
        masks: &Tensor,
        encoding: SegmentationMaskEncoding,
        mask_h: usize,
        mask_w: usize,
    ) -> crate::Result<Self> {
        let dims = masks.dims();
        let data = masks
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok(Self {
            data,
            batch: dims[0],
            objects: dims[1],
            pixels: mask_h * mask_w,
            encoding,
        })
    }

    fn mask_for(&self, batch_idx: usize, object_idx: usize) -> crate::Result<Vec<f32>> {
        if batch_idx >= self.batch {
            return Err(crate::Error::InvalidTensor(format!(
                "segmentation target batch index {batch_idx} is outside batch {}",
                self.batch
            )));
        }
        match self.encoding {
            SegmentationMaskEncoding::PerInstance => {
                if object_idx >= self.objects {
                    return Err(crate::Error::InvalidTensor(format!(
                        "segmentation target object index {object_idx} is outside objects {}",
                        self.objects
                    )));
                }
                let start = (batch_idx * self.objects + object_idx) * self.pixels;
                Ok(self.data[start..start + self.pixels].to_vec())
            }
            SegmentationMaskEncoding::Overlap => {
                if self.objects != 1 {
                    return Err(crate::Error::InvalidTensor(format!(
                        "overlap segmentation masks must have one map channel, got {}",
                        self.objects
                    )));
                }
                let start = batch_idx * self.pixels;
                let expected = (object_idx + 1) as f32;
                Ok(self.data[start..start + self.pixels]
                    .iter()
                    .map(|value| (*value == expected) as u8 as f32)
                    .collect())
            }
        }
    }
}
