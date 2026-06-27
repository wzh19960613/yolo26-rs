# Candle 0.10.2 Patch Guide

This document explains how projects using `yolo26-rs` training features can patch `candle-core 0.10.2` locally.

Training depends on Candle autograd and backend convolution implementations. If the training path uses upsample-nearest backward, or if convolution gradients run on Metal, use a locally patched `candle-core` until upstream releases a version containing the fixes.

## Issues to Fix

1. `upsample_nearest1d` / `upsample_nearest2d` backward should accumulate partial gradients into the existing input gradient instead of overwriting it.
2. Metal `conv1d` / `conv2d` should use the copied contiguous kernel buffer and a zero-offset contiguous layout for matmul when the kernel is non-contiguous.

## Prepare Local Candle

```bash
git clone https://github.com/huggingface/candle.git ../candle
cd ../candle
git checkout v0.10.2
git switch -c yolo26-candle-0.10.2-patches
```

If you already have a Candle fork, create the branch from your fork's `v0.10.2`.

## Patch 1: Accumulate Upsample-Nearest Gradients

Edit `candle-core/src/backprop.rs` and find the `Op::UpsampleNearest1D` and `Op::UpsampleNearest2D` branches.

Change the overwrite form in both branches:

```rust
let sum_grad = grads.or_insert(arg)?;
*sum_grad = conv_sum;
```

to the accumulating form:

```rust
let sum_grad = grads.or_insert(arg)?;
*sum_grad = sum_grad.add(&conv_sum)?;
```

## Patch 2: Use Contiguous Kernels in Metal conv

Edit `candle-core/src/metal_backend/mod.rs` and find the non-contiguous-kernel branches in `conv1d` and `conv2d`. The branch first creates `kernel_c`:

```rust
let mut kernel_c = self.device().zeros_impl(kernel_l.shape(), kernel.dtype())?;
kernel.copy_strided_src(&mut kernel_c, 0, kernel_l)?;
```

Change the later layout and storage passed into matmul from:

```rust
let kernel_l = Layout::contiguous_with_offset((1, n, k), kernel_l.start_offset())
    .transpose(1, 2)?
    .broadcast_as((b, k, n))?;
col.matmul(kernel, (b, m, n, k), &col_l, &kernel_l)?
```

to:

```rust
let kernel_l = Layout::contiguous((1, n, k))
    .transpose(1, 2)?
    .broadcast_as((b, k, n))?;
col.matmul(&kernel_c, (b, m, n, k), &col_l, &kernel_l)?
```

`conv1d` and `conv2d` each have one fix with the same shape.

## Make Your Project Use Patched Candle

Add this to the `Cargo.toml` of the project that uses `yolo26-rs`:

```toml
[patch.crates-io]
candle-core = { path = "../candle/candle-core" }
```

If the patch is in a different directory, use your actual path. Do not commit the full Candle source tree into the `yolo26-rs` repository; manage it through a local path patch, git dependency, or your own fork.

Then resolve dependencies again:

```bash
cargo update -p candle-core
```

Confirm that `candle-core` in `Cargo.lock` points to the local path before building.

## Verification

In the Candle repository, run formatting and focused tests first:

```bash
cd ../candle
cargo fmt
cargo test -p candle-core --test grad_tests upsample_nearest_grad_accumulates_cpu
cargo test -p candle-core --test conv_tests conv2d_non_contiguous_kernel_cpu
```

If you use Metal, also run:

```bash
cargo test -p candle-core --features metal --test grad_tests upsample_nearest_grad_accumulates_metal
cargo test -p candle-core --features metal --test conv_tests conv2d_non_contiguous_kernel_metal
```

If your local Candle checkout does not have these test names, add regression tests for the two fixes, or at least run training-related builds in the `yolo26-rs` project:

```bash
cargo check --features train
cargo check --features train,metal
```

## Minimal Behavior Check

Upsample-nearest gradients should accumulate contributions from multiple downstream paths:

```rust
let x = Var::from_slice(&[1f32, 2., 3., 4.], (1, 1, 2, 2), device)?;
let x = x.as_tensor();
let y = (x.upsample_nearest2d(4, 4)?.sum_all()? + x.sum_all()?)?;
let grads = y.backward()?;
```

For 2x upsample, the input gradient should be `[5, 5, 5, 5]`: the upsample branch contributes `4`, and the direct sum branch contributes `1`.

Metal convolution results with non-contiguous kernels should match contiguous-kernel results:

```rust
let input = Tensor::arange(0f32, 75f32, dev)?.reshape((1, 3, 5, 5))?;
let kernel_base = Tensor::arange(0f32, 54f32, dev)?.reshape((3, 2, 3, 3))?;
let kernel = kernel_base.transpose(0, 1)?;

let expected = input.conv2d(&kernel.contiguous()?, 1, 1, 1, 1)?;
let actual = input.conv2d(&kernel, 1, 1, 1, 1)?;
```

`actual` should match `expected`.

## Remove the Local Patch

After switching to a Candle version that contains these fixes, remove the `[patch.crates-io]` section from the project `Cargo.toml`, then run:

```bash
cargo update -p candle-core
```

Run training-related builds and your minimal training task again.
