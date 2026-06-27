//! YOLOE head infrastructure shared by the detect and segment tasks.
//!
//! Task-specific heads live under [`crate::yoloe::detect::head`] and
//! [`crate::yoloe::segment::head`]. This module keeps only the shared building
//! blocks: the contrastive feature scorer, the head key plan, and the
//! prompt-free LRPC heads used across tasks.

pub mod lrpc;

pub(crate) mod contrastive;
pub(crate) mod key_plan;

pub(crate) use contrastive::BnContrastive;
pub use contrastive::Contrastive;
pub use key_plan::KeyPlan;
