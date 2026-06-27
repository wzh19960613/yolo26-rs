use super::*;

impl OptionalLossAccumulator {
    fn add(&mut self, value: Option<f32>) {
        if let Some(value) = value {
            self.sum += value as f64;
            self.count += 1;
        }
    }

    fn mean(self) -> Option<f32> {
        (self.count > 0).then(|| (self.sum / self.count as f64) as f32)
    }
}

#[derive(Debug, Clone, Copy, Default)]

pub(crate) struct LossComponentsAccumulator {
    total_sum: f64,
    count: usize,
    classification_loss: OptionalLossAccumulator,
    box_loss: OptionalLossAccumulator,
    cls_loss: OptionalLossAccumulator,
    dfl_loss: OptionalLossAccumulator,
    mask_loss: OptionalLossAccumulator,
    pose_loss: OptionalLossAccumulator,
    kobj_loss: OptionalLossAccumulator,
    angle_loss: OptionalLossAccumulator,
    semantic_loss: OptionalLossAccumulator,
    smoke_loss: OptionalLossAccumulator,
}

impl LossComponentsAccumulator {
    pub(crate) fn add(&mut self, components: LossComponents) {
        self.total_sum += components.total as f64;
        self.count += 1;
        self.classification_loss.add(components.classification_loss);
        self.box_loss.add(components.box_loss);
        self.cls_loss.add(components.cls_loss);
        self.dfl_loss.add(components.dfl_loss);
        self.mask_loss.add(components.mask_loss);
        self.pose_loss.add(components.pose_loss);
        self.kobj_loss.add(components.kobj_loss);
        self.angle_loss.add(components.angle_loss);
        self.semantic_loss.add(components.semantic_loss);
        self.smoke_loss.add(components.smoke_loss);
    }

    pub(crate) fn mean(self) -> LossComponents {
        LossComponents {
            total: if self.count == 0 {
                0.0
            } else {
                (self.total_sum / self.count as f64) as f32
            },
            classification_loss: self.classification_loss.mean(),
            box_loss: self.box_loss.mean(),
            cls_loss: self.cls_loss.mean(),
            dfl_loss: self.dfl_loss.mean(),
            mask_loss: self.mask_loss.mean(),
            pose_loss: self.pose_loss.mean(),
            kobj_loss: self.kobj_loss.mean(),
            angle_loss: self.angle_loss.mean(),
            semantic_loss: self.semantic_loss.mean(),
            smoke_loss: self.smoke_loss.mean(),
        }
    }
}

/// Blends two tensor loss-component snapshots with the given weights, used to
/// combine the one-to-many and one-to-one progressive loss reports.
pub(crate) fn blend_tensor_components(
    many: LossTensorComponents,
    one: LossTensorComponents,
    weight_many: f64,
    weight_one: f64,
) -> crate::Result<LossTensorComponents> {
    let blend = |many: Option<Tensor>, one: Option<Tensor>| -> crate::Result<Option<Tensor>> {
        Ok(match (many, one) {
            (Some(m), Some(o)) => Some(((m * weight_many)? + (o * weight_one))?),
            (Some(m), None) => Some((m * weight_many)?),
            (None, Some(o)) => Some((o * weight_one)?),
            (None, None) => None,
        })
    };
    Ok(LossTensorComponents {
        classification_loss: blend(many.classification_loss, one.classification_loss)?,
        box_loss: blend(many.box_loss, one.box_loss)?,
        cls_loss: blend(many.cls_loss, one.cls_loss)?,
        dfl_loss: blend(many.dfl_loss, one.dfl_loss)?,
        mask_loss: blend(many.mask_loss, one.mask_loss)?,
        pose_loss: blend(many.pose_loss, one.pose_loss)?,
        kobj_loss: blend(many.kobj_loss, one.kobj_loss)?,
        angle_loss: blend(many.angle_loss, one.angle_loss)?,
        semantic_loss: blend(many.semantic_loss, one.semantic_loss)?,
        smoke_loss: blend(many.smoke_loss, one.smoke_loss)?,
    })
}

pub(crate) fn scalar_loss_value(loss: &Tensor) -> crate::Result<f32> {
    Ok(loss.to_dtype(DType::F32)?.to_scalar::<f32>()?)
}

pub(crate) fn scalar_optional_loss(loss: &Option<Tensor>) -> crate::Result<Option<f32>> {
    loss.as_ref().map(scalar_loss_value).transpose()
}

/// Training step result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Report {
    /// Scalar loss value before the optimizer step.
    pub loss: f32,
    /// Component-level loss values before the optimizer step.
    pub components: LossComponents,
}
