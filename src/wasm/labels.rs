//! Class-name resolution for the WASM API.
//!
//! Class names are **not** part of any safetensors checkpoint and are optional
//! for `.pt` (Ultralytics stores them in the pickle top-level `names` field).
//! This module prefers the checkpoint-embedded names when available (`.pt`) so
//! that custom-dataset checkpoints show their real classes, falling back to the
//! built-in dataset vocabularies enabled by the `default_labels` feature. The one
//! exception is ImageNet classify checkpoints, whose `.pt` names are WordNet IDs
//! (`n03769881`) — for those the built-in ImageNet vocabulary is used so readable
//! names (`minibus`) win. YOLOE prompt-free uses its own `LRPC_VOCAB`.

use wasm_bindgen::prelude::*;

use super::js_error;

/// Task families with a built-in label vocabulary (when `default_labels` is on).
#[derive(Debug, Clone, Copy)]
pub(super) enum LabelSet {
    Coco,
    Dota,
    Cityscapes,
    Imagenet,
    Lrpc,
}

impl LabelSet {
    /// Picks the built-in vocabulary matching a task kind.
    fn for_task(task: &str) -> Option<Self> {
        match task {
            "detect" | "segment" | "pose" => Some(Self::Coco),
            "obb" => Some(Self::Dota),
            "semantic" => Some(Self::Cityscapes),
            "classify" => Some(Self::Imagenet),
            "yoloe-pf" => Some(Self::Lrpc),
            // yoloe-visual classes are caller-defined (visual_class_N), no vocab.
            _ => None,
        }
    }

    /// Returns the built-in vocabulary slice for this set, when compiled in.
    fn vocab(self) -> Option<&'static [&'static str]> {
        match self {
            Self::Coco | Self::Dota | Self::Cityscapes | Self::Imagenet => {
                #[cfg(feature = "default_labels")]
                {
                    match self {
                        Self::Coco => Some(crate::default_labels::COCO),
                        Self::Dota => Some(crate::default_labels::DOTA),
                        Self::Cityscapes => Some(crate::default_labels::CITYSCAPES),
                        Self::Imagenet => Some(crate::default_labels::IMAGENET),
                        _ => None,
                    }
                }
                #[cfg(not(feature = "default_labels"))]
                {
                    None
                }
            }
            Self::Lrpc => {
                #[cfg(feature = "yoloe-pf")]
                {
                    Some(crate::default_labels::LRPC_VOCAB)
                }
                #[cfg(not(feature = "yoloe-pf"))]
                {
                    None
                }
            }
        }
    }
}

/// Returns the class names for a checkpoint byte buffer + task.
///
/// Resolution order (checkpoint-embedded names win, so custom-dataset
/// checkpoints keep their real classes):
/// 1. `.pt` checkpoint `names` — unless they are ImageNet WordNet IDs
///    (`n03769881`), which are not useful to display. Detecting WordNet IDs
///    lets custom classify checkpoints (real class names) still win.
/// 2. Built-in dataset vocabulary matching the task (`default_labels` /
///    `yoloe-pf`), used when there are no usable `.pt` names, or when those
///    names were WordNet IDs. The built-in vocab is always preferred for
///    `yoloe-pf` (its fixed `LRPC_VOCAB`).
/// 3. Empty `Vec<String>` if neither is available.
///
/// Returns a flat `[name0, name1, ...]` array; index into it by `class_id`.
/// For YOLOE prompt-free this returns the 4585-entry `LRPC_VOCAB`; for
/// YOLOE visual it returns empty (classes are caller-defined).
#[wasm_bindgen(js_name = classNames)]
pub fn class_names(bytes: &[u8], task: &str) -> Result<Vec<String>, JsValue> {
    let mut names: Vec<String> = Vec::new();

    // 1. Prefer the checkpoint-embedded names, so a custom-dataset checkpoint
    //    surfaces its real class names. ImageNet WordNet IDs (`n03769881`) are
    //    skipped here — the built-in ImageNet vocab below gives readable names.
    #[cfg(feature = "pt")]
    if crate::model::is_pt_bytes(bytes)
        && let Ok(meta) = crate::pt_loader::load_pt_metadata_from_bytes(bytes)
        && let Some(pt_names) = meta.names
        && !is_wordnet_ids(&pt_names.0)
    {
        names = pt_names.0;
    }

    // 2. Fall back to the built-in dataset vocabulary. This also covers classify
    //    (whose `.pt` names are WordNet IDs) and yoloe-pf (fixed `LRPC_VOCAB`).
    if names.is_empty()
        && let Some(set) = LabelSet::for_task(task)
        && let Some(vocab) = set.vocab()
    {
        names = vocab.iter().map(|s| s.to_string()).collect();
    }

    if names.is_empty() {
        return Err(js_error(format!(
            "no class names available for task '{task}': the checkpoint has no \
             embedded names and the 'default_labels' feature is off"
        )));
    }
    Ok(names)
}

/// Returns true when `names` look like ImageNet WordNet IDs — a leading `n`
/// followed only by digits, for *every* entry. A real (custom-dataset) classify
/// checkpoint with human-readable names will not match, so its names are kept.
fn is_wordnet_ids(names: &[String]) -> bool {
    if names.is_empty() {
        return false;
    }
    names.iter().all(|n| {
        let mut chars = n.chars();
        matches!(chars.next(), Some('n'))
            && chars.clone().count() > 0
            && chars.all(|c| c.is_ascii_digit())
    })
}

#[cfg(test)]
mod tests {
    use super::is_wordnet_ids;

    #[test]
    fn wordnet_ids_detected() {
        let v = vec!["n01440764".to_string(), "n03769881".to_string()];
        assert!(is_wordnet_ids(&v));
    }

    #[test]
    fn human_names_kept() {
        // A custom-dataset classify checkpoint with real names must NOT be
        // treated as WordNet IDs — its embedded names should win.
        let v = vec!["screw".to_string(), "nut".to_string(), "washer".to_string()];
        assert!(!is_wordnet_ids(&v));
    }

    #[test]
    fn empty_is_not_wordnet() {
        assert!(!is_wordnet_ids(&[]));
    }

    #[test]
    fn partial_match_is_not_wordnet() {
        // Mixed content (one real name, one ID) → not all WordNet → keep `.pt`.
        let v = vec!["n01440764".to_string(), "minibus".to_string()];
        assert!(!is_wordnet_ids(&v));
    }

    #[test]
    fn lone_n_is_not_wordnet() {
        // `n` with no trailing digits is not a valid WordNet ID.
        assert!(!is_wordnet_ids(&["n".to_string()]));
    }
}
