# YOLOE prompt-free 推理（内置 LRPC 开放词表）

checkpoint 内置固定 4585 类的 LRPC 开放词表,推理时无需任何 text/visual prompt,直接预测 class id。类名由内置 `LRPC_VOCAB` 提供可读名。

完整 API 参考见 [YOLOE接口.md](../YOLOE接口.md)。

## 何时用

- 不想手动指定类别,直接用 checkpoint 自带的 4585 类开放词表。
- 类别集合固定且与 LRPC 词表对齐的场景。

## 准备

- **Feature**:`--features yoloe-pf`(`yoloe` 聚合已含)。内置 4585 词表 `LRPC_VOCAB` 归 `yoloe-pf`。
- **权重**:优先使用官方 prompt-free `.pt` checkpoint,如 `yoloe-26s-seg-pf.pt`(`-pf` 后缀,含 LRPC 头)。

## Rust 示例

```rust
use yolo26_rs::{FilterOption, MaskOption};
use yolo26_rs::yoloe::segment::Model;
use yolo26_rs::yoloe::prompt::session::Session;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::from_file("yoloe-26s-seg-pf.pt")?;
    let image = yolo26_rs::Image::from_file("examples/bus.jpg")?;

    // 用内置 4585 词表名构造 session(需 yoloe-pf feature)
    let session = Session::prompt_free_default()?;

    let segs = model.predict(
        &image, &session,
        &FilterOption::default(),
        &MaskOption { high_resolution: true },
    )?;
    for s in &segs {
        let name = yolo26_rs::default_labels::LRPC_VOCAB
            .get(s.detection.class_id as usize)
            .copied()
            .unwrap_or("unknown");
        println!("{} conf={:.2}", name, s.detection.confidence);
    }
    Ok(())
}
```

检测(非分割)路径用 `yoloe::detect::Model`,签名 `predict(&image, &session, &filter)`(无 `mask` 参数)。

## 不启用 yoloe-pf 时

`LRPC_VOCAB` 只在 `yoloe-pf` feature 下编译。未启用时改用 `Session::prompt_free()`(类名为 `pf_XXXX` 占位符,打分仍来自 checkpoint 的 LRPC 头):

```rust
let session = Session::prompt_free()?; // 类名是 pf_0, pf_1, ...
```

或用 `.pt` 权重时直接从 checkpoint metadata 读类名:`pt_loader::load_pt_metadata(path).names`。

## 对应官方 Python

```python
from ultralytics import YOLOE

model = YOLOE("yoloe-26s-seg-pf.pt")
results = model.predict("examples/bus.jpg")
for r in results:
    for b in r.boxes:
        print(model.names[b.cls.item()], b.conf.item())
```

## API 速查

| API | 说明 |
| --- | --- |
| `Session::prompt_free_default()` | 用内置 `LRPC_VOCAB`(4585 名)构造 session(需 `yoloe-pf`)。预测 class id 直接索引该表得可读名。 |
| `Session::prompt_free()` | 无内置词表时的退化入口,类名为 `pf_XXXX` 占位符。 |
| `Session::prompt_free_with_embeddings(table)` | 用外部 LRPC embedding 构造(高级用法)。 |
| `Model::from_file(path)` | 加载 prompt-free checkpoint。 |
| `default_labels::LRPC_VOCAB` | 内置 4585 类名 `&[&str; 4585]`,行序与 checkpoint `vocab.weight` 校准。 |

## 与官方的差异 / 注意事项

- 官方 `YOLOE("*-pf.pt").predict(...)` 用 checkpoint 自带的 `model.names`;本 crate 用 `.pt` 权重时也可通过 `pt_loader::load_pt_metadata(path).names` 直接读出同名表。`LRPC_VOCAB` 是同一张表的内置副本(行序校准一致),用于无需读取 metadata 或使用 `.safetensors` 的场景。
- prompt-free checkpoint 的 `model.23.lrpc.*.vocab.weight` 是固定 4585 行,不是 COCO 80。预测 class id 直接索引 `LRPC_VOCAB` 得可读名。
- prompt-free 不需要任何 prompt(table/visual),直接用 checkpoint 内部 LRPC `loc`/`vocab` 输出打分。
