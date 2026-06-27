//! Pose/keypoint entry points for the WASM API.

use wasm_bindgen::prelude::*;

use crate::pose::{self, Prediction};
use crate::{FilterOption, Image};

use super::config::WasmConfig;
use super::js_error;
use super::pixel::strip_alpha;

/// Loads SafeTensors/`.pt` bytes into an opaque pose model.
#[wasm_bindgen(js_name = PoseModel)]
pub struct WasmPoseModel {
    inner: pose::Model,
}

#[wasm_bindgen(js_class = PoseModel)]
impl WasmPoseModel {
    /// Creates a pose model from in-memory checkpoint bytes.
    #[wasm_bindgen(constructor)]
    pub fn load(bytes: &[u8], config: &WasmConfig) -> Result<Self, JsValue> {
        let native = config
            .to_pose_config()
            .map_err(|err| js_error(err.to_string()))?;
        let inner =
            pose::Model::from_bytes_with(bytes, native).map_err(|err| js_error(err.to_string()))?;
        Ok(Self { inner })
    }

    /// Runs pose/keypoint estimation on an RGB pixel buffer.
    pub fn predict_rgb(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        filter: &FilterOption,
    ) -> Result<WasmPoses, JsValue> {
        let image =
            Image::new(width, height, pixels.to_vec()).map_err(|err| js_error(err.to_string()))?;
        let poses = self
            .inner
            .predict(&image, filter)
            .map_err(|err| js_error(err.to_string()))?;
        Ok(WasmPoses::new(poses))
    }

    /// Runs pose/keypoint estimation on an RGBA pixel buffer (alpha discarded).
    pub fn predict_rgba(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        filter: &FilterOption,
    ) -> Result<WasmPoses, JsValue> {
        let rgb = strip_alpha(pixels, width, height);
        self.predict_rgb(&rgb, width, height, filter)
    }
}

/// One pose/keypoint prediction returned by the WASM API.
#[wasm_bindgen(js_name = Pose)]
#[derive(Debug, Clone)]
pub struct WasmPose {
    pose: Prediction,
}

/// Indexed pose collection returned by the WASM API.
#[wasm_bindgen(js_name = Poses)]
#[derive(Debug, Clone)]
pub struct WasmPoses {
    poses: Vec<Prediction>,
}

impl WasmPoses {
    pub(super) fn new(poses: Vec<Prediction>) -> Self {
        Self { poses }
    }
}

#[wasm_bindgen]
impl WasmPose {
    /// Returns the box left coordinate.
    #[wasm_bindgen(getter)]
    pub fn x(&self) -> f32 {
        self.pose.bbox.x_min
    }

    /// Returns the box top coordinate.
    #[wasm_bindgen(getter)]
    pub fn y(&self) -> f32 {
        self.pose.bbox.y_min
    }

    /// Returns the box width.
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> f32 {
        self.pose.bbox.width()
    }

    /// Returns the box height.
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> f32 {
        self.pose.bbox.height()
    }

    /// Returns the instance confidence.
    #[wasm_bindgen(getter)]
    pub fn confidence(&self) -> f32 {
        self.pose.confidence
    }

    /// Returns the numeric class id.
    #[wasm_bindgen(getter, js_name = classId)]
    pub fn class_id(&self) -> u32 {
        self.pose.class_id
    }

    /// Returns the number of keypoints.
    #[wasm_bindgen(getter, js_name = keypointsCount)]
    pub fn keypoints_count(&self) -> usize {
        self.pose.keypoints.len()
    }

    /// Returns a flat keypoint vector. Each keypoint contributes either
    /// `[x, y, visibility]` (when the model predicts visibility) or `[x, y]`.
    #[wasm_bindgen(js_name = toKeypointsArray)]
    pub fn to_keypoints_array(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.pose.keypoints.len() * 3);
        for kp in &self.pose.keypoints {
            out.push(kp.x);
            out.push(kp.y);
            if let Some(v) = kp.visibility {
                out.push(v);
            }
        }
        out
    }
}

#[wasm_bindgen]
impl WasmPoses {
    /// Returns the number of poses.
    pub fn len(&self) -> usize {
        self.poses.len()
    }

    /// Returns whether the collection is empty.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.poses.is_empty()
    }

    /// Returns one pose by index.
    pub fn get(&self, index: usize) -> Result<WasmPose, JsValue> {
        let pose = self
            .poses
            .get(index)
            .cloned()
            .ok_or_else(|| js_error(format!("pose index {index} is out of bounds")))?;
        Ok(WasmPose { pose })
    }
}
