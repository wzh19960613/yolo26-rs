use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::conv_block::ConvBlock;

pub struct Sppf {
    cv1: ConvBlock,
    cv2: ConvBlock,
    kernel_size: usize,
    pool_count: usize,
    shortcut: bool,
}

impl Sppf {
    pub fn load(
        vb: VarBuilder,
        c_in: usize,
        c_out: usize,
        kernel_size: usize,
        pool_count: usize,
        shortcut: bool,
    ) -> Result<Self> {
        if kernel_size == 0 {
            candle_core::bail!("SPPF kernel_size must be greater than zero, got 0");
        }
        let hidden = c_in / 2;
        Ok(Self {
            cv1: ConvBlock::load(vb.pp("cv1"), c_in, hidden, 1, 1, 1, false)?,
            cv2: ConvBlock::load(
                vb.pp("cv2"),
                hidden * (pool_count + 1),
                c_out,
                1,
                1,
                1,
                true,
            )?,
            kernel_size,
            pool_count,
            shortcut: shortcut && c_in == c_out,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = self.cv1.forward(input)?;
        let mut pools = Vec::with_capacity(self.pool_count + 1);
        pools.push(x);
        for _ in 0..self.pool_count {
            // `pools` was just seeded with the input above, so indexing the
            // last element is always valid without an `Option` unwrap.
            let last_index = pools.len() - 1;
            let pooled = maxpool2d_padded(&pools[last_index], self.kernel_size)?;
            pools.push(pooled);
        }
        let refs: Vec<&Tensor> = pools.iter().collect();
        let out = self.cv2.forward(&Tensor::cat(&refs, 1)?)?;
        if self.shortcut { &out + input } else { Ok(out) }
    }
}

fn maxpool2d_padded(x: &Tensor, kernel_size: usize) -> Result<Tensor> {
    let pad = kernel_size / 2;
    let (b, c, h, w) = x.dims4()?;
    let device = x.device();
    let dtype = x.dtype();
    let top = Tensor::full(f32::NEG_INFINITY, (b, c, pad, w), device)?
        .to_dtype(dtype)?
        .contiguous()?;
    let bottom = Tensor::full(f32::NEG_INFINITY, (b, c, pad, w), device)?
        .to_dtype(dtype)?
        .contiguous()?;
    let y = Tensor::cat(&[&top, x, &bottom], 2)?;
    let padded_h = h + 2 * pad;
    let left = Tensor::full(f32::NEG_INFINITY, (b, c, padded_h, pad), device)?
        .to_dtype(dtype)?
        .contiguous()?;
    let right = Tensor::full(f32::NEG_INFINITY, (b, c, padded_h, pad), device)?
        .to_dtype(dtype)?
        .contiguous()?;
    let padded = Tensor::cat(&[&left, &y, &right], 3)?;
    #[cfg(feature = "train")]
    {
        differentiable_maxpool2d(&padded, kernel_size, h, w)
    }
    #[cfg(not(feature = "train"))]
    {
        padded.max_pool2d_with_stride(kernel_size, 1)
    }
}

#[cfg(feature = "train")]
fn differentiable_maxpool2d(
    padded: &Tensor,
    kernel_size: usize,
    out_h: usize,
    out_w: usize,
) -> Result<Tensor> {
    // Kernel size is validated to be non-zero at SPPF construction time, so the
    // first window always seeds the accumulator; no `Option` unwrap is needed.
    let first_y = 0;
    let first_x = 0;
    let mut out = padded
        .narrow(2, first_y, out_h)?
        .narrow(3, first_x, out_w)?;
    for y in 0..kernel_size {
        for x in 0..kernel_size {
            if y == first_y && x == first_x {
                continue;
            }
            let window = padded.narrow(2, y, out_h)?.narrow(3, x, out_w)?;
            out = out.maximum(&window)?;
        }
    }
    Ok(out)
}
