//! Pure Rust YOLO26 inference runtime.
//!
//! The public API is split into six task roots: [`detect`], [`segment`],
//! [`semantic`], [`classify`], [`pose`], and [`obb`]. Each task exposes its own
//! [`Config`](detect::Config), [`Model`](detect::Model), and strongly typed
//! prediction result.

#![deny(missing_docs)]

#[cfg(not(any(
    feature = "detect",
    feature = "classify",
    feature = "segment",
    feature = "semantic",
    feature = "pose",
    feature = "obb",
    feature = "yoloe-text",
    feature = "yoloe-visual",
    feature = "yoloe-pf",
)))]
compile_error!(
    "enable at least one task feature: detect, classify, segment, semantic, pose, obb, yoloe-text, yoloe-visual, or yoloe-pf"
);

mod bbox;
mod error;
mod image;
mod model;
mod network;
mod options;

/// Pure-Rust loader for official Ultralytics `.pt` checkpoints.
#[cfg(feature = "pt")]
pub mod pt_loader;

#[cfg(feature = "classify")]
pub mod classify;
// The label tables module compiles whenever any of its feature gates is on:
// `default_labels` (COCO/DOTA/CITYSCAPES/IMAGENET) or `yoloe-pf`
// (LRPC_VOCAB). The module itself gates each table internally.
#[cfg(any(feature = "default_labels", feature = "yoloe-pf"))]
pub mod default_labels;
#[cfg(feature = "detect")]
pub mod detect;
pub mod device;
#[cfg(feature = "obb")]
pub mod obb;
#[cfg(feature = "pose")]
pub mod pose;
#[cfg(feature = "segment")]
pub mod segment;
#[cfg(feature = "semantic")]
pub mod semantic;
#[cfg(any(feature = "yoloe-text", feature = "yoloe-visual", feature = "yoloe-pf"))]
pub mod yoloe;

#[cfg(feature = "train")]
pub mod train;

#[cfg(feature = "wasm")]
pub mod wasm;

pub use bbox::BBox;
pub use device::Device;
pub use device::DeviceSpec;
pub use error::Error;
pub use image::Image;
pub use model::DtypeRequest;
pub use model::{ImageSize, Scale};
pub use options::FilterOption;
pub use options::MaskOption;

pub use candle_core::DType;

/// Crate-wide result type.
pub type Result<T> = std::result::Result<T, Error>;
