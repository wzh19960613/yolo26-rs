use super::*;

/// Report returned after partial checkpoint loading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadReport {
    /// Variables loaded from the checkpoint.
    pub loaded: usize,
    /// Model variables missing from the checkpoint.
    pub missing: usize,
    /// Identity variables skipped due to shape mismatch.
    pub skipped: usize,
    /// Model variable names missing from the checkpoint.
    pub missing_names: Vec<String>,
    /// Model variable names skipped due to shape mismatch.
    pub skipped_names: Vec<String>,
}

/// Dense detection raw output before top-k/NMS.
pub struct DenseDetectionOutput {
    /// Box branch output.
    pub boxes: Tensor,
    /// Class branch logits.
    pub scores: Tensor,
    /// Anchor centers.
    pub anchors: Tensor,
    /// Per-anchor stride tensor.
    pub stride_tensor: Tensor,
}

impl DenseDetectionOutput {
    pub(crate) fn from_parts(parts: crate::detect::head::HeadParts) -> Self {
        Self {
            boxes: parts.boxes,
            scores: parts.scores,
            anchors: parts.anchors,
            stride_tensor: parts.stride_tensor,
        }
    }
}

/// Raw train-time output for each task.
pub enum Output {
    /// Detection raw outputs.
    Detect(DenseDetectionOutput),
    /// End-to-end detection raw outputs with one-to-many and one-to-one heads.
    DetectE2e {
        /// One-to-many training head output.
        one_to_many: DenseDetectionOutput,
        /// One-to-one inference-aligned training head output.
        one_to_one: DenseDetectionOutput,
    },
    /// Classification logits.
    Classify {
        /// Raw class logits.
        logits: Tensor,
    },
    /// Instance segmentation raw outputs.
    Segment {
        /// Detection branch raw outputs.
        detect: DenseDetectionOutput,
        /// Mask coefficients.
        masks: Tensor,
        /// Prototype masks.
        proto: Tensor,
        /// Optional YOLO26 semantic logits used only during training.
        semantic: Option<Tensor>,
    },
    /// End-to-end instance segmentation raw outputs.
    SegmentE2e {
        /// One-to-many detection branch raw output.
        one_to_many_detect: DenseDetectionOutput,
        /// One-to-many mask coefficients.
        one_to_many_masks: Tensor,
        /// One-to-one detection branch raw output.
        one_to_one_detect: DenseDetectionOutput,
        /// One-to-one mask coefficients.
        one_to_one_masks: Tensor,
        /// Shared prototype masks.
        proto: Tensor,
        /// Optional YOLO26 semantic logits from the one-to-many proto branch.
        semantic: Option<Tensor>,
    },
    /// Pose raw outputs.
    Pose {
        /// Detection branch raw outputs.
        detect: DenseDetectionOutput,
        /// Keypoint branch output.
        keypoints: Tensor,
    },
    /// End-to-end pose raw outputs.
    PoseE2e {
        /// One-to-many detection branch raw output.
        one_to_many_detect: DenseDetectionOutput,
        /// One-to-many keypoint branch output.
        one_to_many_keypoints: Tensor,
        /// One-to-one detection branch raw output.
        one_to_one_detect: DenseDetectionOutput,
        /// One-to-one keypoint branch output.
        one_to_one_keypoints: Tensor,
    },
    /// Semantic segmentation logits.
    Semantic {
        /// Per-class segmentation logits.
        logits: Tensor,
    },
    /// Oriented bounding-box raw outputs.
    Obb {
        /// Detection branch raw outputs.
        detect: DenseDetectionOutput,
        /// Angle branch output.
        angles: Tensor,
    },
    /// End-to-end oriented bounding-box raw outputs.
    ObbE2e {
        /// One-to-many detection branch raw output.
        one_to_many_detect: DenseDetectionOutput,
        /// One-to-many angle branch output.
        one_to_many_angles: Tensor,
        /// One-to-one detection branch raw output.
        one_to_one_detect: DenseDetectionOutput,
        /// One-to-one angle branch output.
        one_to_one_angles: Tensor,
    },
}

/// Inference-style top-k output used by end-to-end validation.
pub(crate) enum EvalPostprocessOutput {
    /// Detection predictions shaped `[batch, detections, 6]`.
    Detect { predictions: Tensor },
    /// Segmentation predictions shaped `[batch, detections, 6 + nm]` plus prototypes.
    Segment { predictions: Tensor, proto: Tensor },
}

/// Training target supplied to a task loss.
pub enum Target {
    /// Classification class ids shaped `[batch]`.
    Classification {
        /// Class id tensor.
        class_ids: Tensor,
    },
    /// Detection targets in model-image xyxy coordinates.
    Detection(DetectionTargets),
    /// Instance segmentation targets.
    Segmentation(SegmentationTargets),
    /// Pose/keypoint targets.
    Pose(PoseTargets),
    /// Oriented bounding-box targets.
    Obb(ObbTargets),
    /// Semantic segmentation class map shaped `[batch, height, width]`.
    Semantic {
        /// Per-pixel class ids.
        class_map: Tensor,
    },
    /// Dense task targets for task-specific future losses.
    Dense,
}

/// Detection targets in model input coordinates.
#[derive(Debug, Clone)]

pub struct DetectionTargets {
    /// Target boxes shaped `[batch, max_objects, 4]` in xyxy model pixels.
    pub boxes_xyxy: Tensor,
    /// Target class ids shaped `[batch, max_objects]`.
    pub class_ids: Tensor,
    /// Valid-object mask shaped `[batch, max_objects]`.
    pub valid: Tensor,
}

impl DetectionTargets {
    /// Creates validated detection targets.
    pub fn new(boxes_xyxy: Tensor, class_ids: Tensor, valid: Tensor) -> crate::Result<Self> {
        if boxes_xyxy.rank() != 3 || boxes_xyxy.dim(2)? != 4 {
            return Err(crate::Error::InvalidTensor(
                "detection boxes must have shape [batch, max_objects, 4]".to_string(),
            ));
        }
        if class_ids.dims() != &boxes_xyxy.dims()[..2] {
            return Err(crate::Error::InvalidTensor(
                "detection class_ids must have shape [batch, max_objects]".to_string(),
            ));
        }
        if valid.dims() != &boxes_xyxy.dims()[..2] {
            return Err(crate::Error::InvalidTensor(
                "detection valid mask must have shape [batch, max_objects]".to_string(),
            ));
        }
        Ok(Self {
            boxes_xyxy,
            class_ids,
            valid,
        })
    }
}

/// Instance segmentation targets in model/prototype coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentationMaskEncoding {
    /// Masks are stored per instance as `[batch, max_objects, mask_h, mask_w]`.
    PerInstance,
    /// Masks are stored as official overlap instance-index maps `[batch, 1, mask_h, mask_w]`.
    Overlap,
}

/// Instance segmentation targets in model/prototype coordinates.
#[derive(Debug, Clone)]
pub struct SegmentationTargets {
    /// Detection targets shared with the box/class branches.
    pub detection: DetectionTargets,
    /// Instance masks in the layout described by `mask_encoding`.
    pub masks: Tensor,
    /// Per-pixel class ids derived from the instance masks, shaped `[batch, H, W]`.
    pub semantic_masks: Tensor,
    /// Encoding used by the segmentation mask tensor.
    pub mask_encoding: SegmentationMaskEncoding,
}

impl SegmentationTargets {
    /// Creates validated per-instance segmentation targets.
    pub fn new(detection: DetectionTargets, masks: Tensor) -> crate::Result<Self> {
        Self::new_with_mask_encoding(detection, masks, SegmentationMaskEncoding::PerInstance)
    }

    /// Creates validated overlap-encoded segmentation targets.
    pub fn new_overlap(detection: DetectionTargets, masks: Tensor) -> crate::Result<Self> {
        Self::new_with_mask_encoding(detection, masks, SegmentationMaskEncoding::Overlap)
    }

    /// Creates validated segmentation targets with an explicit mask encoding.
    pub fn new_with_mask_encoding(
        detection: DetectionTargets,
        masks: Tensor,
        mask_encoding: SegmentationMaskEncoding,
    ) -> crate::Result<Self> {
        validate_segmentation_masks(&detection, &masks, mask_encoding)?;
        let semantic_masks = build_segmentation_semantic_masks(&detection, &masks, mask_encoding)?;
        Ok(Self {
            detection,
            masks,
            semantic_masks,
            mask_encoding,
        })
    }
}

fn build_segmentation_semantic_masks(
    detection: &DetectionTargets,
    masks: &Tensor,
    mask_encoding: SegmentationMaskEncoding,
) -> crate::Result<Tensor> {
    let batch = detection.boxes_xyxy.dim(0)?;
    let objects = detection.boxes_xyxy.dim(1)?;
    let channels = masks.dim(1)?;
    let height = masks.dim(2)?;
    let width = masks.dim(3)?;
    let classes = detection
        .class_ids
        .to_dtype(candle_core::DType::U32)?
        .to_vec2::<u32>()?;
    let valid = detection
        .valid
        .to_dtype(candle_core::DType::F32)?
        .to_vec2::<f32>()?;
    let mask_data = masks
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut semantic = vec![0u32; batch * height * width];

    match mask_encoding {
        SegmentationMaskEncoding::Overlap => {
            let layout = MaskLayout {
                channels,
                height,
                width,
            };
            for b in 0..batch {
                for y in 0..height {
                    for x in 0..width {
                        let raw = mask_value(&mask_data, b, 0, y, x, layout);
                        if raw <= 0.0 {
                            continue;
                        }
                        let obj = raw.round().max(1.0) as usize - 1;
                        if obj < objects && valid[b][obj] > 0.0 {
                            semantic[(b * height + y) * width + x] = classes[b][obj];
                        }
                    }
                }
            }
        }
        SegmentationMaskEncoding::PerInstance => {
            let layout = MaskLayout {
                channels,
                height,
                width,
            };
            let mut areas = vec![vec![0f32; objects]; batch];
            for (b, area_row) in areas.iter_mut().enumerate().take(batch) {
                for (obj, area_slot) in area_row.iter_mut().enumerate().take(objects) {
                    let mut area = 0f32;
                    for y in 0..height {
                        for x in 0..width {
                            if mask_value(&mask_data, b, obj, y, x, layout) > 0.0 {
                                area += 1.0;
                            }
                        }
                    }
                    *area_slot = area;
                }
            }
            for (b, area_row) in areas.iter().enumerate().take(batch) {
                for y in 0..height {
                    for x in 0..width {
                        let mut best: Option<(usize, f32)> = None;
                        for obj in 0..objects {
                            if valid[b][obj] <= 0.0
                                || mask_value(&mask_data, b, obj, y, x, layout) <= 0.0
                            {
                                continue;
                            }
                            let area = area_row[obj];
                            if best.is_none_or(|(_, best_area)| area < best_area) {
                                best = Some((obj, area));
                            }
                        }
                        if let Some((obj, _)) = best {
                            semantic[(b * height + y) * width + x] = classes[b][obj];
                        }
                    }
                }
            }
        }
    }

    Tensor::from_vec(semantic, (batch, height, width), masks.device()).map_err(Into::into)
}

#[derive(Clone, Copy)]
struct MaskLayout {
    channels: usize,
    height: usize,
    width: usize,
}

fn mask_value(
    data: &[f32],
    batch_idx: usize,
    channel_idx: usize,
    y: usize,
    x: usize,
    layout: MaskLayout,
) -> f32 {
    data[((batch_idx * layout.channels + channel_idx) * layout.height + y) * layout.width + x]
}

fn validate_segmentation_masks(
    detection: &DetectionTargets,
    masks: &Tensor,
    mask_encoding: SegmentationMaskEncoding,
) -> crate::Result<()> {
    if masks.rank() != 4 {
        return Err(crate::Error::InvalidTensor(format!(
            "segmentation masks must have rank 4, got {:?}",
            masks.dims()
        )));
    }
    match mask_encoding {
        SegmentationMaskEncoding::PerInstance => {
            let expected_prefix = detection.boxes_xyxy.dims()[..2].to_vec();
            if masks.dims()[..2] != expected_prefix[..] {
                return Err(crate::Error::InvalidTensor(format!(
                    "segmentation masks must have shape [batch, max_objects, H, W], got {:?}",
                    masks.dims()
                )));
            }
        }
        SegmentationMaskEncoding::Overlap => {
            let batch = detection.boxes_xyxy.dim(0)?;
            if masks.dim(0)? != batch || masks.dim(1)? != 1 {
                return Err(crate::Error::InvalidTensor(format!(
                    "overlap segmentation masks must have shape [batch, 1, H, W], got {:?}",
                    masks.dims()
                )));
            }
        }
    }
    Ok(())
}

/// Pose targets in model-image coordinates.
pub struct PoseTargets {
    /// Detection targets shared with the box/class branches.
    pub detection: DetectionTargets,
    /// Keypoint x/y targets shaped `[batch, max_objects, keypoints, 2]`.
    pub keypoints: Tensor,
    /// Visibility mask shaped `[batch, max_objects, keypoints]`.
    pub visibility: Tensor,
    /// Optional left/right keypoint permutation for horizontal flipping,
    /// mirroring the official data-YAML `flip_idx`. When `None`, the COCO-17
    /// table is used for 17 keypoints and the identity permutation otherwise.
    pub flip_indices: Option<Vec<usize>>,
}
