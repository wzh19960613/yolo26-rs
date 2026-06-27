//! Shared pixel-buffer helpers for the WASM entry points.
//!
//! All task entry points accept RGB pixel buffers; the `_rgba` variants strip
//! the alpha channel here before forwarding to the shared RGB path.

/// Drops the alpha channel from an RGBA pixel buffer, producing an RGB buffer.
pub(super) fn strip_alpha(pixels: &[u8], width: u32, height: u32) -> Vec<u8> {
    let count = width as usize * height as usize;
    let mut rgb = Vec::with_capacity(count * 3);
    for px in pixels.chunks_exact(4).take(count) {
        rgb.extend_from_slice(&px[..3]);
    }
    rgb
}
