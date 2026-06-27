/// Metric family used to select the best training checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BestMetricKind {
    /// Lower training loss is better.
    TrainingLoss,
    /// Higher validation fitness is better.
    ValidationFitness,
}

/// Scalar metric used to decide whether `best.pt` should be updated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BestMetric {
    /// Metric family and comparison direction.
    pub kind: BestMetricKind,
    /// Scalar metric value.
    pub value: f32,
}

impl BestMetric {
    /// Creates a finite training-loss best metric.
    pub fn training_loss(value: f32) -> crate::Result<Self> {
        Self::new(BestMetricKind::TrainingLoss, value)
    }

    /// Creates a finite validation-fitness best metric.
    pub fn validation_fitness(value: f32) -> crate::Result<Self> {
        Self::new(BestMetricKind::ValidationFitness, value)
    }

    /// Parses a sidecar metric token and value.
    pub fn parse(kind: &str, value: f32) -> crate::Result<Self> {
        let kind = match kind {
            "training_loss" => BestMetricKind::TrainingLoss,
            "validation_fitness" => BestMetricKind::ValidationFitness,
            other => {
                return Err(crate::Error::InvalidConfig(format!(
                    "unsupported resume best_metric_name '{other}'"
                )));
            }
        };
        Self::new(kind, value)
    }

    /// Returns the stable JSON token for this metric kind.
    pub const fn name(self) -> &'static str {
        match self.kind {
            BestMetricKind::TrainingLoss => "training_loss",
            BestMetricKind::ValidationFitness => "validation_fitness",
        }
    }

    pub(crate) fn is_better_than(self, best: Option<Self>) -> bool {
        self.value.is_finite()
            && match (self.kind, best) {
                (_, None) => true,
                (BestMetricKind::TrainingLoss, Some(best)) => self.value < best.value,
                (BestMetricKind::ValidationFitness, Some(best)) => self.value >= best.value,
            }
    }

    fn new(kind: BestMetricKind, value: f32) -> crate::Result<Self> {
        if !value.is_finite() {
            return Err(crate::Error::InvalidConfig(
                "best checkpoint metric must be finite".to_string(),
            ));
        }
        Ok(Self { kind, value })
    }
}
