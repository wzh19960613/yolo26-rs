//! Default labels for official YOLO26 semantic-segmentation exports.

/// Labels for semantic segmentation
/// ([Cityscapes Dataset](https://www.cityscapes-dataset.com/), 19 classes).
#[rustfmt::skip]
pub const CITYSCAPES: &[&str] = &[
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
    "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
];
