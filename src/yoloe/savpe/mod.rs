//! YOLOE SAVPE visual-prompt encoder and the masked-average fallback pooler.

pub(crate) mod encoder;
pub(crate) mod encoder_forward;
pub(crate) mod encoder_load;
pub(crate) mod pooler;

pub use encoder::Encoder;
pub use pooler::Pooler;
