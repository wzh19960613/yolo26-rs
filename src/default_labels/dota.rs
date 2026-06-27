//! Default labels for official YOLO26 OBB exports.

/// Labels for oriented bounding-box detection
/// ([DOTA Dataset](https://captain-whu.github.io/DOTA/), 15 classes).
#[rustfmt::skip]
pub const DOTA: &[&str] = &[
    "plane", "ship", "storage tank", "baseball diamond", "tennis court", "basketball court",
    "ground track field", "harbor", "bridge", "large vehicle", "small vehicle", "helicopter",
    "roundabout", "soccer ball field", "swimming pool",
];
