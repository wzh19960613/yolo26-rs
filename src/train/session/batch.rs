use super::*;

impl Session {
    /// Runs one supervised training step.
    pub fn train_batch(&mut self, input: &Tensor, target: &Target) -> crate::Result<Report> {
        self.train_batch_with_loss_config(input, target, DetectionLossConfig::default())
    }

    /// Runs one supervised training step with custom detection-style loss gains.
    pub fn train_batch_with_loss_config(
        &mut self,
        input: &Tensor,
        target: &Target,
        loss_config: DetectionLossConfig,
    ) -> crate::Result<Report> {
        let output = self.model.forward_raw(input)?;
        let report = supervised_loss_report_with_config(&output, target, loss_config)?;
        self.step_loss(report)
    }

    /// Runs one smoke training step over raw outputs.
    pub fn train_smoke_batch(&mut self, input: &Tensor) -> crate::Result<Report> {
        let output = self.model.forward_raw(input)?;
        let loss = smoke_loss(&output)?;
        let report = LossTensorReport {
            loss: loss.clone(),
            components: LossTensorComponents {
                smoke_loss: Some(loss),
                ..Default::default()
            },
        };
        self.step_loss(report)
    }
}
