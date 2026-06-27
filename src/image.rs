use crate::Result;

/// In-memory RGB image passed to detectors.
///
/// Pixel data is always 3-channel RGB (matching the official Ultralytics
/// preprocessing, which converts every input to RGB before inference). Images
/// with an alpha channel must be converted to RGB by the caller before
/// construction.
#[derive(Debug, Clone)]
pub struct Image {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Packed RGB pixel bytes in row-major order (`width * height * 3`).
    pub data: Vec<u8>,
}

impl Image {
    /// Creates an RGB image after validating the pixel buffer length.
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        let expected = width as usize * height as usize * 3;
        if data.len() != expected {
            return Err(crate::Error::InvalidImage(format!(
                "expected {expected} RGB bytes for {width}x{height}, got {}",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Returns a cropped RGB image copied from the current image.
    pub fn crop(&self, x: u32, y: u32, width: u32, height: u32) -> Result<Self> {
        if x >= self.width || y >= self.height {
            return Err(crate::Error::InvalidImage(
                "crop origin is outside the image".to_string(),
            ));
        }
        if x + width > self.width || y + height > self.height || width == 0 || height == 0 {
            return Err(crate::Error::InvalidImage(
                "crop size exceeds image bounds".to_string(),
            ));
        }

        let mut out = vec![0u8; width as usize * height as usize * 3];
        for row in 0..height as usize {
            let src_start = ((y as usize + row) * self.width as usize + x as usize) * 3;
            let dst_start = row * width as usize * 3;
            let len = width as usize * 3;
            out[dst_start..dst_start + len].copy_from_slice(&self.data[src_start..src_start + len]);
        }
        Self::new(width, height, out)
    }

    /// Returns the RGB triplet at the given pixel coordinate as a 3-byte slice.
    ///
    /// Cheaper than returning an array: the caller reads the three bytes
    /// directly from the slice with no copy.
    pub fn pixel(&self, x: usize, y: usize) -> &[u8] {
        let idx = (y * self.width as usize + x) * 3;
        &self.data[idx..idx + 3]
    }

    /// Decodes an image file from disk into an RGB [`Image`].
    ///
    /// This is a convenience constructor that replaces the
    /// `image::open(...)?.to_rgb8()` + `Image::new(...)` sequence. The supported
    /// formats depend on the enabled `image` crate features (by default JPEG,
    /// PNG, BMP, WebP). Images with an alpha channel are flattened to RGB.
    ///
    /// Only available when the `image` feature is enabled.
    #[cfg(feature = "image")]
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let rgb = image::open(path)
            .map_err(|err| crate::Error::InvalidImage(format!("decode failed: {err}")))?
            .to_rgb8();
        let (width, height) = rgb.dimensions();
        Self::new(width, height, rgb.into_raw())
    }
}
