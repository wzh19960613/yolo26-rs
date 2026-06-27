//! Anchor decoding and per-detection mask selection for segmentation output.

use candle_core::Tensor;

use crate::model::{LetterboxInfo, OutputViewer, flattened_rows};
use crate::{FilterOption, MaskOption, Result, detect};

use crate::segment::Prediction;

use super::mask_decode::{
    CroppedMaskSpec, MaskDecodeBase, NativeMaskSpec, instance_mask_tensor,
    instance_mask_tensor_native,
};

pub(crate) fn postprocess_segmentation(
    output: &Tensor,
    proto: &Tensor,
    letterbox: &LetterboxInfo,
    (image_width, image_height): (u32, u32),
    filter: &FilterOption,
    mask: &MaskOption,
) -> Result<Vec<Prediction>> {
    let proto_3d = if proto.rank() == 4 {
        proto.squeeze(0)?
    } else {
        proto.clone()
    };
    let (mask_channels, proto_h, proto_w) = proto_3d.dims3()?;
    let cols = 6 + mask_channels;
    let (rows, flattened) = flattened_rows(output, cols)?;

    let proto_2d = proto_3d.reshape((mask_channels, proto_h * proto_w))?;

    let content_w = letterbox.source_to_feature_w(image_width, proto_w);
    let content_h = letterbox.source_to_feature_h(image_height, proto_h);
    let crop_x = letterbox.feature_pad_x(proto_w).round() as usize;
    let crop_y = letterbox.feature_pad_y(proto_h).round() as usize;

    let mut segmentations = Vec::new();

    for row in 0..rows {
        let r = OutputViewer::for_dynamic(&flattened, row, cols).ok_or_else(|| {
            crate::Error::InvalidTensor(format!(
                "segmentation row {row} out of bounds for {cols}-column flattened output"
            ))
        })?;
        let (confidence, class_id) = match r.check(filter) {
            Some(pair) => pair,
            None => continue,
        };

        let bbox = letterbox.xyxy_to_source_bbox(&r.as_slice()[..4], image_width, image_height);
        if bbox.area() <= 0.0 {
            continue;
        }

        let mask_coeffs = &r.as_slice()[6..cols];
        let mask_base = MaskDecodeBase {
            proto_shape: (proto_h, proto_w),
            channels: mask_channels,
            coefficients: mask_coeffs,
            bbox,
            letterbox,
        };
        let mask = if mask.high_resolution {
            instance_mask_tensor_native(
                &proto_2d,
                NativeMaskSpec {
                    base: mask_base,
                    image_size: (image_width, image_height),
                },
            )?
        } else {
            instance_mask_tensor(
                &proto_2d,
                CroppedMaskSpec {
                    base: mask_base,
                    crop_origin: (crop_x, crop_y),
                    content_size: (content_w, content_h),
                },
            )?
        };

        segmentations.push(Prediction {
            detection: detect::Prediction {
                bbox,
                confidence,
                class_id,
            },
            mask,
        });
    }

    Ok(segmentations)
}
