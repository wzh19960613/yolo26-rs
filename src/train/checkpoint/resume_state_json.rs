use super::{BestMetric, ResumeState};

impl ResumeState {
    /// Serializes the resume state to stable JSON.
    pub fn to_json_pretty(&self) -> String {
        format!(
            "{{\n  \"version\": 1,\n  \"completed_epochs\": {},\n  \"completed_steps\": {},\n  \"best_epoch\": {},\n  \"best_loss\": {},\n  \"best_metric_name\": {},\n  \"best_metric_value\": {}\n}}\n",
            self.completed_epochs,
            self.completed_steps,
            json_option_usize(self.best_epoch),
            json_option_f32(self.best_loss),
            json_option_str(self.best_metric.map(BestMetric::name)),
            json_option_f32(self.best_metric.map(|metric| metric.value))
        )
    }

    /// Parses a resume state from the JSON written by `to_json_pretty`.
    pub fn from_json_str(text: &str) -> crate::Result<Self> {
        let best_loss = parse_optional_f32_field(text, "best_loss")?;
        let best_metric = parse_optional_best_metric(text, best_loss)?;
        Self::new_with_best_metric(
            parse_usize_field(text, "completed_epochs")?,
            parse_usize_field(text, "completed_steps")?,
            parse_optional_usize_field(text, "best_epoch")?,
            best_loss,
            best_metric,
        )
    }

    /// Writes this resume state to a JSON sidecar path.
    pub fn write_json(&self, path: impl AsRef<std::path::Path>) -> crate::Result<()> {
        Ok(std::fs::write(path, self.to_json_pretty())?)
    }

    /// Reads a resume state from a JSON sidecar path.
    pub fn read_json(path: impl AsRef<std::path::Path>) -> crate::Result<Self> {
        Self::from_json_str(&std::fs::read_to_string(path)?)
    }
}

fn json_option_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn json_option_f32(value: Option<f32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn json_option_str(value: Option<&str>) -> String {
    value
        .map(|value| format!("\"{value}\""))
        .unwrap_or_else(|| "null".to_string())
}

fn parse_optional_best_metric(
    text: &str,
    best_loss: Option<f32>,
) -> crate::Result<Option<BestMetric>> {
    let Some(name) = optional_field_value(text, "best_metric_name")? else {
        return best_loss.map(BestMetric::training_loss).transpose();
    };
    if name == "null" {
        return Ok(None);
    }
    let value = optional_field_value(text, "best_metric_value")?.ok_or_else(|| {
        crate::Error::InvalidConfig("missing resume field best_metric_value".to_string())
    })?;
    BestMetric::parse(
        name.trim_matches('"'),
        value.parse().map_err(|err| {
            crate::Error::InvalidConfig(format!("invalid resume best_metric_value: {err}"))
        })?,
    )
    .map(Some)
}

fn parse_usize_field(text: &str, name: &str) -> crate::Result<usize> {
    field_value(text, name)?
        .parse()
        .map_err(|err| crate::Error::InvalidConfig(format!("invalid resume {name}: {err}")))
}

fn parse_optional_usize_field(text: &str, name: &str) -> crate::Result<Option<usize>> {
    let value = field_value(text, name)?;
    if value == "null" {
        Ok(None)
    } else {
        parse_usize_field(text, name).map(Some)
    }
}

fn parse_optional_f32_field(text: &str, name: &str) -> crate::Result<Option<f32>> {
    let value = field_value(text, name)?;
    if value == "null" {
        Ok(None)
    } else {
        value
            .parse()
            .map(Some)
            .map_err(|err| crate::Error::InvalidConfig(format!("invalid resume {name}: {err}")))
    }
}

fn field_value<'a>(text: &'a str, name: &str) -> crate::Result<&'a str> {
    let key = format!("\"{name}\"");
    let after_key = text
        .split_once(&key)
        .and_then(|(_, rest)| rest.split_once(':').map(|(_, value)| value))
        .ok_or_else(|| crate::Error::InvalidConfig(format!("missing resume field {name}")))?;
    Ok(after_key
        .split([',', '\n', '}'])
        .next()
        .unwrap_or_default()
        .trim())
}

fn optional_field_value<'a>(text: &'a str, name: &str) -> crate::Result<Option<&'a str>> {
    let key = format!("\"{name}\"");
    if text.contains(&key) {
        field_value(text, name).map(Some)
    } else {
        Ok(None)
    }
}
