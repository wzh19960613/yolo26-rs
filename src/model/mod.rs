// Task-specific helpers (letterbox_rect for YOLOE, flattened_rows for segment,
// ForPose for pose) are only used when the corresponding feature is enabled.
#![allow(dead_code, unused_imports)]

pub mod config;
pub mod dtype_request;
pub mod scale;

mod letterbox;
mod output_viewer;
mod rect_letterbox;
mod scale_inference;

pub use config::ImageSize;
pub use dtype_request::DtypeRequest;
pub use letterbox::MODEL_INPUT_SIZE;
pub use scale::Scale;

#[allow(unused_imports)]
pub(crate) use letterbox::{LetterboxInfo, letterbox, letterbox_with_canvas};
#[allow(unused_imports)]
pub(crate) use output_viewer::{OutputViewer, flattened_rows};
#[allow(unused_imports)]
pub(crate) use rect_letterbox::letterbox_rect;
pub(crate) use scale_inference::{
    InferredTask, checkpoint_shapes, checkpoint_shapes_from_bytes, infer_keypoints_count,
    infer_labels_count_from_shapes, infer_scale_from_shapes, is_pt_bytes, is_pt_path,
    shapes_from_safetensors,
};
#[cfg(feature = "pt")]
pub(crate) use scale_inference::{shapes_from_pt, shapes_from_pt_bytes};
