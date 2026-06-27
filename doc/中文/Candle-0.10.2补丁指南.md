# Candle 0.10.2 Patch Guide

本文说明如何为使用 `yolo26-rs` 训练能力的项目自行 patch `candle-core 0.10.2`。

训练依赖 Candle autograd 和后端 convolution 实现。如果训练路径会用到 upsample-nearest backward，或在 Metal 上执行 convolution gradient，建议在上游发布包含修复的版本前使用本地 patched `candle-core`。

## 需要修复的问题

1. `upsample_nearest1d` / `upsample_nearest2d` 的 backward 应把 partial gradient 累加到已有 input gradient，而不是覆盖已有值。
2. Metal `conv1d` / `conv2d` 遇到非连续 kernel 时，应在 matmul 中使用已复制出的连续 kernel buffer，并使用零偏移 contiguous layout。

## 准备本地 Candle

```bash
git clone https://github.com/huggingface/candle.git ../candle
cd ../candle
git checkout v0.10.2
git switch -c yolo26-candle-0.10.2-patches
```

如果你已有 Candle fork，也可以直接在 fork 的 `v0.10.2` 基础上建分支。

## Patch 1：累加 upsample-nearest 梯度

编辑 `candle-core/src/backprop.rs`，找到 `Op::UpsampleNearest1D` 和 `Op::UpsampleNearest2D` 两个分支。

把这两处分支中的覆盖写法：

```rust
let sum_grad = grads.or_insert(arg)?;
*sum_grad = conv_sum;
```

改成累加写法：

```rust
let sum_grad = grads.or_insert(arg)?;
*sum_grad = sum_grad.add(&conv_sum)?;
```

## Patch 2：Metal conv 使用连续 kernel

编辑 `candle-core/src/metal_backend/mod.rs`，在 `conv1d` 和 `conv2d` 中找到处理非连续 kernel 的分支。该分支会先创建 `kernel_c`：

```rust
let mut kernel_c = self.device().zeros_impl(kernel_l.shape(), kernel.dtype())?;
kernel.copy_strided_src(&mut kernel_c, 0, kernel_l)?;
```

把后续 matmul 前的 layout 和 storage 使用方式从：

```rust
let kernel_l = Layout::contiguous_with_offset((1, n, k), kernel_l.start_offset())
    .transpose(1, 2)?
    .broadcast_as((b, k, n))?;
col.matmul(kernel, (b, m, n, k), &col_l, &kernel_l)?
```

改成：

```rust
let kernel_l = Layout::contiguous((1, n, k))
    .transpose(1, 2)?
    .broadcast_as((b, k, n))?;
col.matmul(&kernel_c, (b, m, n, k), &col_l, &kernel_l)?
```

`conv1d` 和 `conv2d` 各有一处相同形状的修正。

## 让项目使用 patched Candle

在使用 `yolo26-rs` 的项目 `Cargo.toml` 中添加：

```toml
[patch.crates-io]
candle-core = { path = "../candle/candle-core" }
```

如果 patch 是放在别的目录，改成你的实际路径。不要把完整 Candle 源码提交进 `yolo26-rs` 仓库；用本地 path patch、git dependency 或自己的 fork 管理。

然后重新解析依赖：

```bash
cargo update -p candle-core
```

确认 `Cargo.lock` 中的 `candle-core` 指向本地 path 后再构建。

## 验证

在 Candle 仓库里先跑格式和目标测试：

```bash
cd ../candle
cargo fmt
cargo test -p candle-core --test grad_tests upsample_nearest_grad_accumulates_cpu
cargo test -p candle-core --test conv_tests conv2d_non_contiguous_kernel_cpu
```

如果你使用 Metal，再跑：

```bash
cargo test -p candle-core --features metal --test grad_tests upsample_nearest_grad_accumulates_metal
cargo test -p candle-core --features metal --test conv_tests conv2d_non_contiguous_kernel_metal
```

如果本地 Candle 没有这些测试名，可以先添加针对两个修复点的回归测试，或至少在 `yolo26-rs` 项目中跑训练相关构建：

```bash
cargo check --features train
cargo check --features train,metal
```

## 最小行为检查

upsample-nearest 梯度应累加来自多条下游路径的贡献：

```rust
let x = Var::from_slice(&[1f32, 2., 3., 4.], (1, 1, 2, 2), device)?;
let x = x.as_tensor();
let y = (x.upsample_nearest2d(4, 4)?.sum_all()? + x.sum_all()?)?;
let grads = y.backward()?;
```

对于 2x upsample，输入梯度应为 `[5, 5, 5, 5]`：upsample 分支贡献 `4`，直接 sum 分支贡献 `1`。

Metal 非连续 kernel 的卷积结果应与连续 kernel 一致：

```rust
let input = Tensor::arange(0f32, 75f32, dev)?.reshape((1, 3, 5, 5))?;
let kernel_base = Tensor::arange(0f32, 54f32, dev)?.reshape((3, 2, 3, 3))?;
let kernel = kernel_base.transpose(0, 1)?;

let expected = input.conv2d(&kernel.contiguous()?, 1, 1, 1, 1)?;
let actual = input.conv2d(&kernel, 1, 1, 1, 1)?;
```

`actual` 应与 `expected` 一致。

## 移除本地 patch

当你切到已经包含这些修复的 Candle 版本后，删除项目 `Cargo.toml` 中的 `[patch.crates-io]` 段，然后运行：

```bash
cargo update -p candle-core
```

再重新跑训练相关构建和你的最小训练任务。
