//! YOLOE checkpoint identity parsed from an official filename.
//!
//! Extracted from [`crate::yoloe::config`]: [`Identity`] parses the scale /
//! segmentation / prompt-free semantics out of official `yoloe-26*` names and
//! converts them into a [`Config`](super::Config).

use crate::Scale;

use crate::yoloe::config::Config;
use crate::yoloe::usage::CheckpointKind;

/// YOLOE checkpoint identity parsed from an official filename.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity {
    /// Original checkpoint stem or filename.
    pub name: String,
    /// YOLOE scale.
    pub scale: Scale,
    /// Identity family.
    pub kind: CheckpointKind,
    /// Whether this checkpoint is segmentation-first.
    pub segmentation: bool,
    /// Whether this checkpoint is prompt-free.
    pub prompt_free: bool,
}

impl Identity {
    /// Parses official-style YOLOE checkpoint names such as
    /// `yoloe-26s-seg.pt` / `yoloe-26s-seg.safetensors` or the `-pf` variants.
    ///
    /// Both `.pt` and `.safetensors` suffixes are accepted; official `.pt` is
    /// the primary format (loadable via `pt_loader` / `from_pt_file`, writable
    /// via `Train::save_pt`), and `.safetensors` remains supported.
    pub fn parse(name: impl AsRef<str>) -> crate::Result<Self> {
        let name = name.as_ref();
        let file = name.rsplit_once('/').map(|(_, file)| file).unwrap_or(name);
        let stem = file
            .strip_suffix(".safetensors")
            .or_else(|| file.strip_suffix(".pt"))
            .unwrap_or(file);
        let lower = stem.to_ascii_lowercase();
        let Some(body) = lower.strip_prefix("yoloe-") else {
            return Err(crate::Error::InvalidConfig(format!(
                "YOLOE checkpoint name '{name}' must start with yoloe-"
            )));
        };
        let scale_char = body
            .chars()
            .find(|ch| matches!(ch, 'n' | 's' | 'm' | 'l' | 'x'))
            .ok_or_else(|| {
                crate::Error::InvalidConfig(format!(
                    "YOLOE checkpoint name '{name}' does not contain a scale"
                ))
            })?;
        // `scale_char` is constrained to n/s/m/l/x by the `find` guard above,
        // so this conversion cannot fail; surface a specific message regardless.
        let scale = Scale::try_from(scale_char).map_err(|_| {
            crate::Error::InvalidConfig(format!(
                "invalid YOLOE scale '{scale_char}' in '{name}' (expected n/s/m/l/x)"
            ))
        })?;
        let prompt_free = lower.ends_with("-pf") || lower.contains("-pf-");
        let segmentation = lower.contains("-seg");
        let kind = if prompt_free {
            CheckpointKind::PromptFree
        } else if segmentation {
            CheckpointKind::Segmentation
        } else {
            CheckpointKind::Prompted
        };
        Ok(Self {
            name: stem.to_string(),
            scale,
            kind,
            segmentation,
            prompt_free,
        })
    }

    /// Converts this checkpoint identity into a model config.
    pub fn config(&self) -> Config {
        if self.prompt_free {
            Config::prompt_free(self.scale)
        } else if self.segmentation {
            Config::segmentation(self.scale)
        } else {
            Config {
                scale: self.scale,
                checkpoint: self.kind,
                ..Config::default()
            }
        }
    }
}
