use std::fmt::{Display, Formatter};

/// Errors returned by image validation, model loading, and inference.
#[derive(Debug)]
pub enum Error {
    /// The supplied image buffer or dimensions are invalid.
    InvalidImage(String),
    /// A configuration value is invalid.
    InvalidConfig(String),
    /// A tensor has an unexpected shape or value.
    InvalidTensor(String),
    /// The requested compute device is unavailable in this build.
    DeviceUnavailable(String),
    /// The requested capability is declared but not implemented by this build.
    Unsupported(String),
    /// An IO error occurred.
    Io(std::io::Error),
    /// Candle returned an error.
    Candle(candle_core::Error),
    /// A PyTorch `.pt` zip archive could not be read.
    PtZip(String),
    /// The CLIP text encoder (`mobileclip2-b-rs`) returned an error.
    #[cfg(feature = "yoloe-text")]
    Clip(mobileclip2_b_rs::Error),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidImage(msg) => write!(f, "invalid image: {msg}"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::InvalidTensor(msg) => write!(f, "invalid tensor: {msg}"),
            Self::DeviceUnavailable(msg) => write!(f, "device unavailable: {msg}"),
            Self::Unsupported(msg) => write!(f, "unsupported operation: {msg}"),
            Self::Io(err) => Display::fmt(err, f),
            Self::Candle(err) => Display::fmt(err, f),
            Self::PtZip(msg) => write!(f, "invalid .pt archive: {msg}"),
            #[cfg(feature = "yoloe-text")]
            Self::Clip(err) => write!(f, "CLIP text encoder error: {err}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<candle_core::Error> for Error {
    fn from(value: candle_core::Error) -> Self {
        Self::Candle(value)
    }
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

#[cfg(feature = "pt")]
impl From<zip::result::ZipError> for Error {
    fn from(value: zip::result::ZipError) -> Self {
        Self::PtZip(value.to_string())
    }
}

#[cfg(feature = "yoloe-text")]
impl From<mobileclip2_b_rs::Error> for Error {
    fn from(value: mobileclip2_b_rs::Error) -> Self {
        Self::Clip(value)
    }
}
