use super::*;

impl PoseTargets {
    /// Creates validated pose targets.
    pub fn new(
        detection: DetectionTargets,
        keypoints: Tensor,
        visibility: Tensor,
    ) -> crate::Result<Self> {
        let batch = detection.boxes_xyxy.dim(0)?;
        let objects = detection.boxes_xyxy.dim(1)?;
        if keypoints.rank() != 4
            || keypoints.dim(0)? != batch
            || keypoints.dim(1)? != objects
            || keypoints.dim(3)? != 2
        {
            return Err(crate::Error::InvalidTensor(format!(
                "pose keypoints must have shape [{batch}, {objects}, keypoints, 2], got {:?}",
                keypoints.dims()
            )));
        }
        let keypoints_count = keypoints.dim(2)?;
        if visibility.dims() != [batch, objects, keypoints_count] {
            return Err(crate::Error::InvalidTensor(format!(
                "pose visibility must have shape [{batch}, {objects}, {keypoints_count}], got {:?}",
                visibility.dims()
            )));
        }
        Ok(Self {
            detection,
            keypoints,
            visibility,
            flip_indices: None,
        })
    }
}

/// Oriented bounding-box targets in model-image coordinates.
pub struct ObbTargets {
    /// Detection targets shared with the box/class branches.
    pub detection: DetectionTargets,
    /// Rotation angle targets shaped `[batch, max_objects]`, in radians.
    pub angles: Tensor,
    /// Oriented boxes shaped `[batch, max_objects, 5]` as `[cx, cy, w, h, angle]`.
    pub rboxes_xywhr: Tensor,
}

impl ObbTargets {
    /// Creates validated OBB targets.
    pub fn new(detection: DetectionTargets, angles: Tensor) -> crate::Result<Self> {
        let rboxes_xywhr = infer_rboxes_xywhr(&detection, &angles)?;
        Self::new_with_rboxes(detection, angles, rboxes_xywhr)
    }

    /// Creates validated OBB targets with explicit official `xywhr` boxes.
    pub fn new_with_rboxes(
        detection: DetectionTargets,
        angles: Tensor,
        rboxes_xywhr: Tensor,
    ) -> crate::Result<Self> {
        let batch = detection.boxes_xyxy.dim(0)?;
        let objects = detection.boxes_xyxy.dim(1)?;
        if angles.dims() != [batch, objects] {
            return Err(crate::Error::InvalidTensor(format!(
                "obb angles must have shape [{batch}, {objects}], got {:?}",
                angles.dims()
            )));
        }
        if rboxes_xywhr.dims() != [batch, objects, 5] {
            return Err(crate::Error::InvalidTensor(format!(
                "obb rboxes must have shape [{batch}, {objects}, 5], got {:?}",
                rboxes_xywhr.dims()
            )));
        }
        Ok(Self {
            detection,
            angles,
            rboxes_xywhr,
        })
    }
}

fn infer_rboxes_xywhr(detection: &DetectionTargets, angles: &Tensor) -> crate::Result<Tensor> {
    let boxes = detection
        .boxes_xyxy
        .to_dtype(candle_core::DType::F32)?
        .to_vec3::<f32>()?;
    let angles_data = angles.to_dtype(candle_core::DType::F32)?.to_vec2::<f32>()?;
    let mut flat = Vec::with_capacity(detection.boxes_xyxy.elem_count() / 4 * 5);
    for b in 0..boxes.len() {
        for obj in 0..boxes[b].len() {
            let xyxy = &boxes[b][obj];
            flat.push((xyxy[0] + xyxy[2]) * 0.5);
            flat.push((xyxy[1] + xyxy[3]) * 0.5);
            flat.push((xyxy[2] - xyxy[0]).max(0.0));
            flat.push((xyxy[3] - xyxy[1]).max(0.0));
            flat.push(angles_data[b][obj]);
        }
    }
    let batch = detection.boxes_xyxy.dim(0)?;
    let objects = detection.boxes_xyxy.dim(1)?;
    Tensor::from_vec(flat, (batch, objects, 5), detection.boxes_xyxy.device())?
        .to_dtype(detection.boxes_xyxy.dtype())
        .map_err(Into::into)
}

/// Weights for the current supervised detection loss.
#[derive(Debug, Clone, Copy, PartialEq)]

pub struct DetectionLossConfig {
    /// Weight applied to box distance regression loss.
    pub box_weight: f64,
    /// Weight applied to multi-label class BCE loss.
    pub class_weight: f64,
    /// Weight applied to l/t/r/b distance loss when `reg_max == 1`.
    pub distance_weight: f64,
    /// Weight applied to pose keypoint position loss, matching Ultralytics `pose`.
    pub pose_weight: f64,
    /// Weight applied to keypoint visibility/objectness loss, matching Ultralytics `kobj`.
    pub keypoint_objectness_weight: f64,
    /// Weight applied to oriented-box angle loss, matching Ultralytics `angle`.
    pub angle_weight: f64,
    /// Number of candidate anchors retained per target by the task-aligned assigner.
    pub tal_topk: usize,
    /// Optional second-stage candidate count after duplicate-anchor conflict resolution.
    pub tal_topk2: Option<usize>,
    /// Classification confidence exponent in the task-aligned metric.
    pub tal_alpha: f32,
    /// IoU exponent in the task-aligned metric.
    pub tal_beta: f32,
}

impl Default for DetectionLossConfig {
    fn default() -> Self {
        Self {
            box_weight: 7.5,
            class_weight: 0.5,
            distance_weight: 1.5,
            pose_weight: 12.0,
            keypoint_objectness_weight: 1.0,
            angle_weight: 1.0,
            tal_topk: 10,
            tal_topk2: None,
            tal_alpha: 0.5,
            tal_beta: 6.0,
        }
    }
}

/// Returns whether a variable name belongs to the task-specific prediction head.
pub fn is_task_head_variable(task: Task, name: &str) -> bool {
    match task {
        Task::Detect | Task::Segment | Task::Pose | Task::Obb => name.starts_with("model.23."),
        Task::Classify => name.starts_with("model.10."),
        Task::Semantic => name.starts_with("model.17."),
    }
}
