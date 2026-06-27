//! HSV jitter operating on flattened CHW float images in `[0, 1]`.
//!
//! Mirrors `ultralytics/data/augment.py::RandomHSV.apply_image` (8.3.79+):
//!
//! - Hue: additive shift `lut_hue = (x + r_h * 180) % 180` on the OpenCV
//!   hue scale `[0, 180)`.
//! - Saturation: multiplicative `lut_sat = clip(x * (r_s + 1), 0, 255)`,
//!   with `lut_sat[0] = 0` to prevent pure white changing color.
//! - Value: multiplicative `lut_val = clip(x * (r_v + 1), 0, 255)`.
//!
//! The random gains are `r = uniform(-1, 1) * gain` per channel.

use super::SeededRng;

/// Applies HSV jitter in place to a flattened `[C=3, H, W]` image in `[0, 1]`.
///
/// `planar_rgb` is laid out as `[R plane, G plane, B plane]` (planar, channel
/// first). `config` carries the official `hsv_h` / `hsv_s` / `hsv_v` gains.
pub(crate) fn augment_hsv(planar_rgb: &mut [f32], config: HsvGains, rng: &mut SeededRng) {
    if config.hgain == 0.0 && config.sgain == 0.0 && config.vgain == 0.0 {
        return;
    }
    // Official: r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]
    let rh = rng.uniform(-1.0, 1.0) * config.hgain;
    let rs = rng.uniform(-1.0, 1.0) * config.sgain;
    let rv = rng.uniform(-1.0, 1.0) * config.vgain;

    let plane = planar_rgb.len() / 3;
    for px in 0..plane {
        let (r, g, b) = (
            planar_rgb[px],
            planar_rgb[plane + px],
            planar_rgb[2 * plane + px],
        );
        // Convert to OpenCV-style HSV (H: 0-180, S: 0-255, V: 0-255).
        let (h, s, v) = rgb_to_hsv_opencv(r, g, b);
        // Official RandomHSV LUTs:
        //   lut_hue = ((x + rh*180) % 180)   — additive shift on [0,180)
        //   lut_sat = clip(x*(rs+1), 0, 255); lut_sat[0] = 0
        //   lut_val = clip(x*(rv+1), 0, 255)
        let h = (h + rh * 180.0).rem_euclid(180.0);
        let s = if s < 0.5 {
            0.0 // lut_sat[0]=0: prevent pure white changing color
        } else {
            (s * (rs + 1.0)).clamp(0.0, 255.0)
        };
        let v = (v * (rv + 1.0)).clamp(0.0, 255.0);
        let (r, g, b) = hsv_to_rgb_opencv(h, s, v);
        planar_rgb[px] = r;
        planar_rgb[plane + px] = g;
        planar_rgb[2 * plane + px] = b;
    }
}

/// HSV jitter gains (official `hsv_h`, `hsv_s`, `hsv_v`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct HsvGains {
    pub(crate) hgain: f32,
    pub(crate) sgain: f32,
    pub(crate) vgain: f32,
}

/// RGB `[0,1]` → OpenCV HSV (H: 0-180, S: 0-255, V: 0-255), matching
/// `cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)`.
fn rgb_to_hsv_opencv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // cv2 works on 0-255; scale.
    let r255 = r * 255.0;
    let g255 = g * 255.0;
    let b255 = b * 255.0;
    let maxc = r255.max(g255).max(b255);
    let minc = r255.min(g255).min(b255);
    let delta = maxc - minc;
    let v = maxc;
    let s = if maxc <= 0.0 {
        0.0
    } else {
        delta / maxc * 255.0
    };
    // OpenCV hue: 0-180 (half of 0-360).
    let h = if delta <= 0.0 {
        0.0
    } else if (maxc - r255).abs() < f32::EPSILON {
        60.0 * (((g255 - b255) / delta).rem_euclid(6.0)) / 2.0
    } else if (maxc - g255).abs() < f32::EPSILON {
        60.0 * ((b255 - r255) / delta + 2.0) / 2.0
    } else {
        60.0 * ((r255 - g255) / delta + 4.0) / 2.0
    };
    (h, s, v)
}

/// OpenCV HSV (H: 0-180, S: 0-255, V: 0-255) → RGB `[0,1]`.
fn hsv_to_rgb_opencv(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    // Convert H from 0-180 to 0-360.
    let h360 = h * 2.0;
    let s = s / 255.0;
    let v = v / 255.0;
    let c = v * s;
    let hp = h360 / 60.0;
    let x = c * (1.0 - (hp.rem_euclid(2.0) - 1.0).abs());
    let m = v - c;
    let (r1, g1, b1) = match hp.floor() as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (r1 + m, g1 + m, b1 + m)
}
