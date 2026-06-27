use super::*;

impl DetectionLossConfig {
    /// Creates the YOLO26 one-to-many assignment config used during E2E training.
    pub fn yolo26_one_to_many() -> Self {
        Self {
            tal_topk: 10,
            tal_topk2: None,
            ..Default::default()
        }
    }

    /// Creates the YOLO26 one-to-one assignment config used during E2E training.
    pub fn yolo26_one_to_one() -> Self {
        Self {
            tal_topk: 7,
            tal_topk2: Some(1),
            ..Default::default()
        }
    }

    pub(crate) fn tal_primary_topk(self) -> usize {
        self.tal_topk.max(1)
    }

    pub(crate) fn tal_secondary_topk(self) -> usize {
        self.tal_topk2
            .unwrap_or(self.tal_topk)
            .max(1)
            .min(self.tal_primary_topk())
    }

    pub(crate) fn with_yolo26_one_to_many_assignment(self) -> Self {
        Self {
            tal_topk: 10,
            tal_topk2: None,
            ..self
        }
    }

    pub(crate) fn with_yolo26_one_to_one_assignment(self) -> Self {
        Self {
            tal_topk: 7,
            tal_topk2: Some(1),
            ..self
        }
    }

    /// Validates loss gains and task-aligned assignment parameters.
    pub fn validate(self) -> crate::Result<()> {
        for (name, value) in [
            ("box", self.box_weight),
            ("cls", self.class_weight),
            ("dfl", self.distance_weight),
            ("pose", self.pose_weight),
            ("kobj", self.keypoint_objectness_weight),
            ("angle", self.angle_weight),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(crate::Error::InvalidConfig(format!(
                    "loss gain {name} must be finite and non-negative"
                )));
            }
        }
        if self.tal_topk == 0 || matches!(self.tal_topk2, Some(0)) {
            return Err(crate::Error::InvalidConfig(
                "task-aligned top-k values must be greater than zero".to_string(),
            ));
        }
        if !self.tal_alpha.is_finite()
            || !self.tal_beta.is_finite()
            || self.tal_alpha < 0.0
            || self.tal_beta < 0.0
        {
            return Err(crate::Error::InvalidConfig(
                "task-aligned alpha/beta must be finite and non-negative".to_string(),
            ));
        }
        Ok(())
    }
}
