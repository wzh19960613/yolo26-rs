# SAHI 切片检测（detect::sahi）

本篇演示如何用 `detect::sahi` 做 SAHI（Slicing Aided Hyper Inference）风格的小目标检测：把大图切成若干带重叠的切片，分别推理，再把坐标平移回原图并合并去重。对高分辨率、密集小目标场景能显著提升召回。

完整 API 参考见 [任务推理API.md](../任务推理API.md) 的 sahi 段。

## 何时用

- 大尺寸航拍/遥感/监控图像里的小目标（直接缩到 640 会丢小目标）。
- 想在不换模型的前提下提高小目标召回。
- 需要控制切片大小、重叠比例和合并策略。

## 准备

- **Feature**：默认能力，**无需额外 feature**（复用 `detect` 任务）。
- **权重**：普通 detect `.pt` 权重，如 `yolo26s.pt`。
- **图像**：一张高分辨率图（示例用 `examples/boats.jpg`）。

```bash
cargo build --release
```

## Rust 示例

```rust
use yolo26_rs::{
    FilterOption, Image, detect,
    detect::sahi,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = Image::from_file("examples/boats.jpg")?;
    let model = detect::Model::from_file("yolo26s.pt")?;
    let sahi_opts = sahi::Options {
        slice_width: 320,
        slice_height: 320,
        overlap_width_ratio: 0.25,
        overlap_height_ratio: 0.25,
        include_full_image: true,
        ..sahi::Options::default()
    };
    let dets = sahi::sliced_predict(&model, &image, &FilterOption::default(), &sahi_opts)?;
    println!("sahi detections: {}", dets.len());
    for d in &dets {
        println!("class={} conf={:.2}", d.class_id, d.confidence);
    }
    Ok(())
}
```

只生成切片窗口、自己编排推理（例如多线程、跨设备）：

```rust
use yolo26_rs::{FilterOption, Image, detect, detect::sahi};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = Image::from_file("examples/boats.jpg")?;
    let model = detect::Model::from_file("yolo26s.pt")?;
    let sahi_opts = sahi::Options {
        slice_width: 320, slice_height: 320,
        overlap_width_ratio: 0.25, overlap_height_ratio: 0.25,
        ..sahi::Options::default()
    };
    let windows = sahi::generate_slices(image.width, image.height, &sahi_opts);
    for w in &windows {
        let crop = image.crop(w.x, w.y, w.width, w.height)?;
        let mut part = model.predict(&crop, &FilterOption::default())?;
        for d in part.iter_mut() {
            *d = d.translated(w.x as f32, w.y as f32);
        }
    }
    Ok(())
}
```

## 对应官方 Python 代码

官方 Ultralytics 自身不内置 SAHI；社区标准做法用独立的 `sahi` 库 + Ultralytics 模型：

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo26s.pt",
    confidence_threshold=0.25,
)
result = get_sliced_prediction(
    "examples/boats.jpg",
    detection_model,
    slice_height=320, slice_width=320,
    overlap_height_ratio=0.25, overlap_width_ratio=0.25,
    perform_standard_pred=True,
)
for ann in result.object_prediction_list:
    print(ann.category.id, ann.score.value, ann.bbox.to_xyxy())
```

## API 与配置项详解

### `sahi::sliced_predict(detector, image, inference, options)`

端到端：内部 `generate_slices` → 逐切片 `predict` → 坐标平移 → `merge_detections`。

| API | 说明 |
| --- | --- |
| `sahi::sliced_predict(...)` | 切片推理 + 合并，返回 `Vec<detect::Prediction>`。 |
| `sahi::generate_slices(w, h, &options)` | 只生成切片窗口 `Vec<SliceWindow>`。 |
| `sahi::merge_detections(dets, &options)` | 合并/去重跨切片检测。 |

### `sahi::Options`

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `slice_width` / `slice_height` | `640` / `640` | 单个切片像素尺寸。 |
| `overlap_width_ratio` / `overlap_height_ratio` | `0.2` / `0.2` | 相邻切片重叠比例。 |
| `include_full_image` | `false` | 是否额外跑一次全图并合并。 |
| `merge_strategy` | `GreedyNmm` | 合并策略：`Nms`（非极大抑制）或 `GreedyNmm`（贪心非极大合并）。 |
| `match_metric` | `Ios` | 匹配度量：`Iou`（交并比）或 `Ios`（交/小框面积）。 |
| `match_threshold` | `0.5` | 合并的最小匹配度量。 |
| `class_agnostic` | `false` | 是否允许不同类别的框相互合并。 |

### `SliceWindow`（单切片）

| 字段 | 说明 |
| --- | --- |
| `x` / `y` | 切片左上角在原图的坐标。 |
| `width` / `height` | 切片尺寸。 |

## 与官方的差异 / 注意事项

- 本 crate 的 SAHI 是原生 Rust 实现，复用同一 `detect::Model`，不依赖第三方 `sahi` Python 包。
- 切片推理是对同一模型反复调用；如需并行，用 `generate_slices` 自行编排（注意 `Model` 在 Candle 上的线程安全约束）。
- `include_full_image: true` 会把全图结果与切片结果一起合并，常用于兼顾大目标。
- 官方 `sahi` 库的 `Nms`/`GreedyNmm` 行为已在本 crate 对齐；默认 `GreedyNmm + Ios`。
