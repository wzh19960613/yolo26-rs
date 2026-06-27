//! Default labels for official YOLO26 / YOLOE-26 exports.
//!
//! Each label table lives in its own file (one const per file):
//! - [`COCO`] (detect/segment/pose, 80),
//! - [`DOTA`] (obb, 15),
//! - [`CITYSCAPES`] (semantic, 19),
//! - [`IMAGENET`] (classify, 1000),
//! - [`LRPC_VOCAB`] (YOLOE prompt-free open vocabulary, 4585).
//!
//! The standard dataset tables are gated behind `default_labels` (off by
//! default — enable it to embed the class names). The YOLOE prompt-free
//! vocabulary [`LRPC_VOCAB`] is gated by `yoloe-pf` instead, since it
//! only matters for that inference path. Disabling both shrinks the binary by
//! dropping every label table; task configs then fall back to the datasets'
//! canonical class counts.

#[cfg(feature = "default_labels")]
mod cityscapes;
#[cfg(feature = "default_labels")]
mod coco;
#[cfg(feature = "default_labels")]
mod dota;
#[cfg(feature = "default_labels")]
mod imagenet;
#[cfg(feature = "yoloe-pf")]
mod lrpc;

#[cfg(feature = "default_labels")]
pub use cityscapes::CITYSCAPES;
#[cfg(feature = "default_labels")]
pub use coco::COCO;
#[cfg(feature = "default_labels")]
pub use dota::DOTA;
#[cfg(feature = "default_labels")]
pub use imagenet::IMAGENET;
#[cfg(feature = "yoloe-pf")]
pub use lrpc::LRPC_VOCAB;
