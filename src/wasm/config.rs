//! JS-facing config carrier for the WASM API.
//!
//! [`WasmConfig`] is a serializable carrier of the model selection (task kind,
//! scale, input size, label/keypoint counts, [`DeviceSpec`]) so it can live on
//! the JS side without a backend-specific candle `Device`. The per-task
//! `to_*_config()` builders live in [`super::builders`]; the shared option
//! constructors live in [`super::options`].

use wasm_bindgen::prelude::*;

use crate::model::MODEL_INPUT_SIZE;
use crate::{DeviceSpec, Scale};

use super::js_error;

/// Returns `(default input size, default label count)` for a task kind.
/// Written as `if`-chains (not `match`) so it compiles under any subset of
/// gated task variants.
fn defaults_for(kind: TaskKind) -> (usize, usize) {
    #[cfg(feature = "classify")]
    if matches!(kind, TaskKind::Classify) {
        return (224, 1000);
    }
    #[cfg(not(feature = "classify"))]
    let _ = kind;
    (MODEL_INPUT_SIZE, 80)
}

/// Task family selected by a [`WasmConfig`]. Mirrors the crate's task roots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TaskKind {
    Detect,
    #[cfg(feature = "segment")]
    Segment,
    #[cfg(feature = "semantic")]
    Semantic,
    #[cfg(feature = "classify")]
    Classify,
    #[cfg(feature = "pose")]
    Pose,
    #[cfg(feature = "obb")]
    Obb,
    #[cfg(feature = "yoloe-visual")]
    YoloeVisual,
    #[cfg(feature = "yoloe-pf")]
    YoloePromptFree,
}

impl TaskKind {
    /// Parses a task kind from its stable lowercase name.
    fn parse(s: &str) -> Result<Self, JsValue> {
        match s {
            "detect" => Ok(Self::Detect),
            #[cfg(feature = "segment")]
            "segment" => Ok(Self::Segment),
            #[cfg(feature = "semantic")]
            "semantic" => Ok(Self::Semantic),
            #[cfg(feature = "classify")]
            "classify" => Ok(Self::Classify),
            #[cfg(feature = "pose")]
            "pose" => Ok(Self::Pose),
            #[cfg(feature = "obb")]
            "obb" => Ok(Self::Obb),
            #[cfg(feature = "yoloe-visual")]
            "yoloe-visual" => Ok(Self::YoloeVisual),
            #[cfg(feature = "yoloe-pf")]
            "yoloe-pf" => Ok(Self::YoloePromptFree),
            other => Err(js_error(format!("unknown task '{other}'"))),
        }
    }

    /// Returns the stable lowercase name of this task kind.
    fn as_str(self) -> &'static str {
        match self {
            Self::Detect => "detect",
            #[cfg(feature = "segment")]
            Self::Segment => "segment",
            #[cfg(feature = "semantic")]
            Self::Semantic => "semantic",
            #[cfg(feature = "classify")]
            Self::Classify => "classify",
            #[cfg(feature = "pose")]
            Self::Pose => "pose",
            #[cfg(feature = "obb")]
            Self::Obb => "obb",
            #[cfg(feature = "yoloe-visual")]
            Self::YoloeVisual => "yoloe-visual",
            #[cfg(feature = "yoloe-pf")]
            Self::YoloePromptFree => "yoloe-pf",
        }
    }
}

/// JS-facing model configuration that resolves to a native task config.
///
/// Holds the serializable selection (task kind, scale, square input size, label
/// count, keypoint count and a [`DeviceSpec`]) so it can live on the JS side
/// without a backend-specific candle device handle.
#[wasm_bindgen(js_name = Config)]
#[derive(Debug, Clone)]
pub struct WasmConfig {
    pub(super) task: TaskKind,
    pub(super) scale: Scale,
    pub(super) image_size: usize,
    pub(super) labels_count: usize,
    pub(super) keypoints_count: usize,
    pub(super) device: DeviceSpec,
}

#[wasm_bindgen(js_class = Config)]
impl WasmConfig {
    /// Creates a model config for the given task with CPU device and sensible
    /// defaults. `task` is one of `"detect"`, `"segment"`, `"semantic"`,
    /// `"classify"`, `"pose"`, `"obb"`, `"yoloe-visual"`, `"yoloe-pf"`.
    #[wasm_bindgen(constructor)]
    pub fn new(task: &str) -> Result<Self, JsValue> {
        let kind = TaskKind::parse(task)?;
        let (input_size, labels_count) = defaults_for(kind);
        Ok(Self {
            task: kind,
            scale: Scale::N,
            image_size: input_size,
            labels_count,
            keypoints_count: 17,
            device: DeviceSpec::Cpu,
        })
    }

    /// Returns a copy configured for a different task.
    #[wasm_bindgen(js_name = withTask)]
    pub fn with_task(&self, task: &str) -> Result<Self, JsValue> {
        let mut next = self.clone();
        next.task = TaskKind::parse(task)?;
        Ok(next)
    }

    /// Returns a copy with the given model scale.
    #[wasm_bindgen(js_name = withScale)]
    pub fn with_scale(&self, scale: Scale) -> Self {
        let mut next = self.clone();
        next.scale = scale;
        next
    }

    /// Returns a copy with the given square input size.
    #[wasm_bindgen(js_name = withInputSize)]
    pub fn with_input_size(&self, input_size: usize) -> Self {
        let mut next = self.clone();
        next.image_size = input_size;
        next
    }

    /// Returns a copy with the given class label count.
    #[wasm_bindgen(js_name = withLabelsCount)]
    pub fn with_labels_count(&self, labels_count: usize) -> Self {
        let mut next = self.clone();
        next.labels_count = labels_count;
        next
    }

    /// Returns a copy with the given keypoint count (pose only).
    #[wasm_bindgen(js_name = withKeypointsCount)]
    pub fn with_keypoints_count(&self, keypoints_count: usize) -> Self {
        let mut next = self.clone();
        next.keypoints_count = keypoints_count;
        next
    }

    /// Returns a copy configured to run on the CPU.
    #[wasm_bindgen(js_name = withCpuDevice)]
    pub fn with_cpu_device(&self) -> Self {
        let mut next = self.clone();
        next.device = DeviceSpec::Cpu;
        next
    }

    /// Returns a copy configured to run on a CUDA device by index.
    #[wasm_bindgen(js_name = withCudaDevice)]
    pub fn with_cuda_device(&self, index: usize) -> Self {
        let mut next = self.clone();
        next.device = DeviceSpec::Cuda(index);
        next
    }

    /// Returns a copy configured to run on a Metal device by index.
    #[wasm_bindgen(js_name = withMetalDevice)]
    pub fn with_metal_device(&self, index: usize) -> Self {
        let mut next = self.clone();
        next.device = DeviceSpec::Metal(index);
        next
    }

    /// Returns the task kind as its stable lowercase name.
    #[wasm_bindgen(getter, js_name = taskKind)]
    pub fn task_kind(&self) -> String {
        self.task.as_str().to_string()
    }

    /// Returns the device kind as a string ("cpu", "cuda", or "metal").
    #[wasm_bindgen(getter, js_name = deviceKind)]
    pub fn device_kind(&self) -> String {
        match self.device {
            DeviceSpec::Cpu => "cpu",
            DeviceSpec::Cuda(_) => "cuda",
            DeviceSpec::Metal(_) => "metal",
        }
        .to_string()
    }

    /// Returns the device index (0 for CPU).
    #[wasm_bindgen(getter, js_name = deviceIndex)]
    pub fn device_index(&self) -> usize {
        match self.device {
            DeviceSpec::Cpu => 0,
            DeviceSpec::Cuda(i) | DeviceSpec::Metal(i) => i,
        }
    }
}
