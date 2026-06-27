# YOLOE visual prompt 推理（框/掩码示例提示）

在当前图上给几个示例目标的框(或 mask),模型据此为同类目标生成预测。内部完成 letterbox、SAVPE 编码和后处理。

完整 API 参考见 [YOLOE接口.md](../YOLOE接口.md)。

## 何时用

- 同一张图里有多个同类目标,用一两个框标注示例就能召回全部。
- 类别难以用文本准确描述,更适合「给个例子」。
- 需要逐图 image-specific 的视觉提示推理。

## 准备

- **Feature**:`--features yoloe-visual`(`yoloe` 聚合已含)。
- **权重**:需含官方 SAVPE 权重的 `-seg` `.pt` checkpoint(如 `yoloe-26s-seg.pt`)。**纯检测权重(`yoloe-26s.pt`,或对应 `.safetensors`)不含 SAVPE**,在其上调用会运行时报错。

## Rust 示例（box visual prompt 分割）

```rust
use yolo26_rs::{FilterOption, Image, MaskOption};
use yolo26_rs::yoloe::segment::Model;
use yolo26_rs::yoloe::prompt::session::Session;
use yolo26_rs::yoloe::prompt::visual::Visual;
use yolo26_rs::yoloe::visuals::VisualSource;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = Image::from_file("examples/bus.jpg")?;
    let model = Model::from_file("yoloe-26s-seg.pt")?;

    // visual prompt 是 per-image 的:每图新建一个 session
    let prompts = vec![
        Visual::from_box(0, [10.0, 20.0, 90.0, 160.0])?,
        Visual::from_box(1, [120.0, 30.0, 220.0, 180.0])?,
    ];
    let session = Session::visual(prompts.clone())?;

    let segs = model.predict_visual_prompts(
        &image, &prompts, VisualSource::Boxes, &session,
        &FilterOption::default(),
        &MaskOption { high_resolution: true },
    )?;
    for s in &segs {
        println!("class={} conf={:.2}", s.detection.class_id, s.detection.confidence);
    }
    Ok(())
}
```

mask 形式:把 `Visual::from_box(...)` 换成 `Visual::from_mask(...)`,并把 `VisualSource::Boxes` 换成 `VisualSource::Masks(&source_masks)`,其中 `source_masks` 是原图坐标的 `[prompts, H, W]` mask 张量。

检测(非分割)路径用 `yoloe::detect::Model`,签名 `predict_visual_prompts(&image, &prompts, source, &session, &filter)`(无 `mask` 参数)。

## 跨图 visual prompt（参考图 → 目标图）

`predict_visual_prompts` 是**单图**路径。若想「在图 A 上给一个示例,再用它去识别图 B」,用两步 API——与官方 `predictor.get_vpe()` + `set_classes(names, vpe)` 对齐:

```rust
use yolo26_rs::{FilterOption, MaskOption};
use yolo26_rs::yoloe::segment::Model;
use yolo26_rs::yoloe::prompt::session::Session;
use yolo26_rs::yoloe::prompt::visual::Visual;
use yolo26_rs::yoloe::visuals::VisualSource;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::from_file("yoloe-26n-seg.pt")?;

    // 第一步:参考图 A + A 上一个 box → 可复用 vpe(image-agnostic embedding)
    let image_a = yolo26_rs::Image::from_file("examples/bus.jpg")?;
    let prompts_a = vec![Visual::from_box(0, [49.0, 399.0, 247.0, 902.0])?];
    let vpe = model.encode_visual_prompts(&image_a, &prompts_a, VisualSource::Boxes)?;

    // 第二步:vpe 当分类器,识别任意图 B(走普通 predict 路径)
    let session = Session::text_with_embeddings(vpe)?;
    let image_b = yolo26_rs::Image::from_file("examples/boats.jpg")?;
    let segs = model.predict(
        &image_b, &session,
        &FilterOption::default(),
        &MaskOption { high_resolution: true },
    )?;
    Ok(())
}
```

对应官方 Python:`vpe = model.predictor.get_vpe(image_a, visuals); model.set_classes(names, vpe); model.predict("boats.jpg")`。

## API 速查

### 视觉提示类型

| API | 说明 |
| --- | --- |
| `Visual::from_box(class_id, [x1,y1,x2,y2])` | box prompt 元素(原图坐标)。 |
| `Visual::from_mask(class_id, [x1,y1,x2,y2])` | mask prompt 元素(原图坐标,实际 mask 由 `VisualSource::Masks` 提供)。 |
| `VisualSource::Boxes` / `VisualSource::Masks(&tensor)` | `predict_visual_prompts` 的来源判别,决定 box 光栅化还是吃 source mask。 |
| `BatchVisuals::from_boxes(items, target_size, scale, device)` | 多图 batch helper,返回 `tensor` + `class_ids`。 |

### 模型与预测

| API | 说明 |
| --- | --- |
| `Model::predict_visual_prompts(&image, &prompts, source, &session, &filter, &mask)` | 单图 box 或 mask visual prompt seg。 |
| `Model::encode_visual_prompts(&reference_image, &prompts, source)` | 跨图:在参考图上编码出可复用 `EmbeddingTable`(官方 `vpe`)。 |
| `Session::visual(prompts)` | visual prompt session(每图新建)。 |

> `predict_visual_prompts` 内部已自动 letterbox + SAVPE。session 不预计算 SAVPE embedding(SAVPE 需要 backbone 特征,只能在前向时算)。

## 与官方的差异 / 注意事项

- 官方 `predict(visual_prompts=...)` 直接吃图像/boxes/masks;本 crate 提供 typed `predict_visual_prompts` 单图入口,box/mask 由 `VisualSource` 判别。
- **纯检测权重(`yoloe-26s.pt`,或对应 `.safetensors`)不含 SAVPE**:在其上调用 visual prompt 会运行时报错。要做 visual prompt 的检测,请用 `-seg` 权重加载成 `yoloe::detect::Model`。
- visual prompt 返回的类名为 `visual_class_{原始id}` 占位符,原始 id 由 `BatchVisuals::from_boxes()` 返回的 `class_ids` 提供多图映射。
- vpe 是 image-agnostic 的 embedding,但**编码时仍需参考图 A 的 backbone 特征**(SAVPE 设计如此)。一次编码的 vpe 可识别多张目标图,但 prompt 类别集合在编码时固定。
