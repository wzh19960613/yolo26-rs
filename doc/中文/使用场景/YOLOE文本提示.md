# YOLOE text prompt 推理（开放词表分割/检测）

传入任意类别名列表，模型用 CLIP 文本 embedding 作为分类器，做开放词表检测/分割。类名经 `mobileclip2-rs` 的 MobileCLIP2-b CLIP 编码,再由 Model 持有的 RepRTA 对齐,匹配官方 `set_classes` 流程。

完整 API 参考见 [YOLOE接口.md](../YOLOE接口.md)。

## 何时用

- 类别不固定、随业务变化的开放词表（如「person」「bus」「my_custom_thing」）。
- 不想为每个数据集重新训练分类头,用文本即时定义类别。

## 准备

- **Feature**:`--features yoloe-text`(`yoloe` 聚合已含)。
- **权重**:优先使用官方 `.pt`,YOLOE 分割权重如 `yoloe-26s-seg.pt`;检测权重如 `yoloe-26s.pt`。
- **CLIP 资源**:优先使用 MobileCLIP2-b 官方权重 `mobileclip2_b.pt` 和 `tokenizer.json`,由调用方构造 `ClipTextEncoder` 时提供(不在 YOLOE checkpoint 内,无默认路径)。

## Rust 示例

```rust
use yolo26_rs::{FilterOption, MaskOption};
use yolo26_rs::yoloe::segment::Model;
use yolo26_rs::yoloe::prompt::session::Session;
use yolo26_rs::yoloe::prompt::text_encoder::ClipTextEncoder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Model 加载时从 model.23.reprta 构造 RepRTA
    let model = Model::from_file("yoloe-26n-seg.pt")?;
    let image = yolo26_rs::Image::from_file("examples/bus.jpg")?;

    // CLIP encoder 构造一次,跨 session 复用
    let encoder = ClipTextEncoder::from_files(
        "mobileclip2_b.pt",
        "tokenizer.json",
    )?;

    // 类名 → CLIP embedding → RepRTA 对齐 → session
    let session = Session::text(&encoder, model.reprta(), ["person", "bus", "car"])?;

    let segs = model.predict(
        &image, &session,
        &FilterOption::default(),
        &MaskOption { high_resolution: true },
    )?;
    for s in &segs {
        println!("class={} conf={:.2}", s.detection.class_id, s.detection.confidence);
    }
    Ok(())
}
```

检测(非分割)路径用 `yoloe::detect::Model`,签名 `predict(&image, &session, &filter)`(无 `mask` 参数)。

## 已有外部 embedding 时

如果类名 embedding 已经在外部编码好(如官方 CLIP 产出),用 `Session::text_with_embeddings` 直接注入,跳过 CLIP 编码:

```rust
use yolo26_rs::yoloe::{EmbeddingTable, prompt::session::Session};

let table = EmbeddingTable::new(embeddings, class_names)?;
let session = Session::text_with_embeddings(table)?;
```

## 对应官方 Python

```python
from ultralytics import YOLOE

model = YOLOE("yoloe-26n-seg.pt")
model.set_classes(["person", "bus", "car"], model.get_text_pe(["person", "bus", "car"]))
results = model.predict("examples/bus.jpg")
```

## API 速查

### 构造与预测

| API | 说明 |
| --- | --- |
| `Model::from_file(path)` | 加载 YOLOE seg 模型,同时构造 RepRTA(`model.reprta()`)。 |
| `ClipTextEncoder::from_files(weights, tokenizer)` | 构造可重用 CLIP encoder(也支持 `from_bytes`、`new`)。 |
| `Session::text(&encoder, model.reprta(), classes)` | CLIP 编码 + RepRTA 对齐,构造 text prompt session。`classes` 接受 `AsRef<str>` 迭代器(`["a","b"]`、`Vec<&str>` 等)。 |
| `Session::text_with_embeddings(table)` | 注入外部 embedding,跳过 CLIP 编码。 |
| `model.predict(&image, &session, &filter, &mask)` | seg 预测;detect 为 `predict(&image, &session, &filter)`。 |

### Config builder

| API | 说明 |
| --- | --- |
| `yoloe::config_builder()` | 返回 `Config` builder。 |
| `.with_rep_rta_enabled(bool)` | 启用/禁用 RepRTA 对齐(默认启用)。 |
| `.with_image_size(size)` | 输入分辨率。 |
| `.with_scale(scale)` | 模型 scale(N/S/M/L/X)。 |
| `.build()` | 构造 `Config`,传给 `Session::text_with_config`。 |

## 与官方的差异 / 注意事项

- 官方 `set_classes` 在 Python 侧调用 CLIP;本 crate 经 `mobileclip2-rs` 依赖调用 MobileCLIP2-b CLIP(`ClipTextEncoder`),`Session::text` 把类名编码成 `[classes, 512]` L2-normalized embedding。
- 官方推理路径是 CLIP → RepRTA → score。RepRTA 由 `Model` 加载时从 `model.23.reprta` 持有(`Model::reprta()`),`Session::text` 借用对齐,无需手动加载或传 checkpoint 路径。
- CLIP encoder 由调用方构造一次(`ClipTextEncoder::from_files` / `from_bytes` / `new`),传 `&encoder` 复用。CLIP 资源不在 YOLOE checkpoint 内,无默认路径。
- YOLOE one-to-one head 设计上不需要 NMS;`FilterOption.agnostic_nms` 当前是 no-op。
- `.safetensors` 转换权重仍可用,但文档和示例默认以官方 `.pt` 为主。
