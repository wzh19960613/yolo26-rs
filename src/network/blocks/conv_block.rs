use candle_core::{Module, Result, Tensor};
use candle_nn::{BatchNormConfig, Conv2d, Conv2dConfig, Init, VarBuilder, batch_norm};

#[cfg(feature = "train")]
use candle_nn::{BatchNorm, ModuleT};

#[cfg(feature = "train")]
thread_local! {
    static TRAINING_MODE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static FUSED_CONV_LAYOUT: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(feature = "train")]
pub(crate) fn with_training_mode<T>(train: bool, f: impl FnOnce() -> T) -> T {
    struct Guard(bool);

    impl Drop for Guard {
        fn drop(&mut self) {
            TRAINING_MODE.with(|mode| mode.set(self.0));
        }
    }

    let guard = TRAINING_MODE.with(|mode| Guard(mode.replace(train)));
    let out = f();
    drop(guard);
    out
}

#[cfg(feature = "train")]
pub(crate) fn with_fused_conv_layout<T>(fused: bool, f: impl FnOnce() -> T) -> T {
    struct Guard(Option<bool>);

    impl Drop for Guard {
        fn drop(&mut self) {
            FUSED_CONV_LAYOUT.with(|mode| mode.set(self.0));
        }
    }

    let guard = FUSED_CONV_LAYOUT.with(|mode| Guard(mode.replace(Some(fused))));
    let out = f();
    drop(guard);
    out
}

#[cfg(not(feature = "train"))]
pub(crate) fn with_fused_conv_layout<T>(_fused: bool, f: impl FnOnce() -> T) -> T {
    f()
}

#[cfg(feature = "train")]
fn training_mode() -> bool {
    TRAINING_MODE.with(std::cell::Cell::get)
}

#[cfg(feature = "train")]
fn forced_fused_conv_layout() -> Option<bool> {
    FUSED_CONV_LAYOUT.with(std::cell::Cell::get)
}

pub struct ConvBlock {
    conv: Conv2d,
    #[cfg(feature = "train")]
    bn: Option<BatchNorm>,
    activated: bool,
}

impl ConvBlock {
    pub fn load(
        vb: VarBuilder,
        c_in: usize,
        c_out: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        activated: bool,
    ) -> Result<Self> {
        // Auto-detect BN-fused (deploy) checkpoints: official YOLOE `-pf` files
        // store a conv with bias and no `bn.*` keys, while `-seg` files carry BN.
        #[cfg(feature = "train")]
        let fused = forced_fused_conv_layout()
            .unwrap_or_else(|| vb.contains_tensor("conv.bias") && !vb.contains_tensor("bn.weight"));
        #[cfg(not(feature = "train"))]
        let fused = vb.contains_tensor("conv.bias") && !vb.contains_tensor("bn.weight");
        if fused {
            return Self::load_fused(vb, c_in, c_out, kernel_size, stride, groups, activated);
        }
        let cfg = Conv2dConfig {
            padding: kernel_size / 2,
            stride,
            groups,
            ..Default::default()
        };
        let conv = pytorch_conv2d_no_bias(c_in, c_out, kernel_size, cfg, vb.pp("conv"))?;
        let bn_cfg = BatchNormConfig {
            eps: 1e-3,
            momentum: 0.03,
            ..Default::default()
        };
        let bn = batch_norm(c_out, bn_cfg, vb.pp("bn"))?;
        #[cfg(feature = "train")]
        {
            Ok(Self {
                conv,
                bn: Some(bn),
                activated,
            })
        }
        #[cfg(not(feature = "train"))]
        Ok(Self {
            conv: conv.absorb_bn(&bn)?,
            activated,
        })
    }

    /// Loads a BN-fused ConvBlock from deploy checkpoints (e.g. official
    /// `yoloe-26n-seg-pf`), which store a single conv with bias and no BN keys.
    ///
    /// The conv reads `vb.pp("conv")` (weight + bias); BN is omitted.
    pub fn load_fused(
        vb: VarBuilder,
        c_in: usize,
        c_out: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        activated: bool,
    ) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: kernel_size / 2,
            stride,
            groups,
            ..Default::default()
        };
        let conv = pytorch_conv2d(c_in, c_out, kernel_size, cfg, vb.pp("conv"))?;
        #[cfg(feature = "train")]
        {
            Ok(Self {
                conv,
                bn: None,
                activated,
            })
        }
        #[cfg(not(feature = "train"))]
        Ok(Self { conv, activated })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.conv.forward(x)?;
        #[cfg(feature = "train")]
        let y = match &self.bn {
            Some(bn) => bn.forward_t(&y, training_mode())?,
            None => y,
        };
        if self.activated {
            candle_nn::ops::silu(&y)
        } else {
            Ok(y)
        }
    }
}

pub(crate) fn pytorch_conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv2dConfig,
    vb: VarBuilder,
) -> Result<Conv2d> {
    let weight = vb.get_with_hints(
        (
            out_channels,
            in_channels / cfg.groups,
            kernel_size,
            kernel_size,
        ),
        "weight",
        pytorch_conv2d_init(in_channels, kernel_size, cfg.groups),
    )?;
    let bias = vb.get_with_hints(
        out_channels,
        "bias",
        pytorch_conv2d_init(in_channels, kernel_size, cfg.groups),
    )?;
    Ok(Conv2d::new(weight, Some(bias), cfg))
}

pub(crate) fn pytorch_conv2d_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv2dConfig,
    vb: VarBuilder,
) -> Result<Conv2d> {
    let weight = vb.get_with_hints(
        (
            out_channels,
            in_channels / cfg.groups,
            kernel_size,
            kernel_size,
        ),
        "weight",
        pytorch_conv2d_init(in_channels, kernel_size, cfg.groups),
    )?;
    Ok(Conv2d::new(weight, None, cfg))
}

pub(crate) fn pytorch_conv2d_init(in_channels: usize, kernel_size: usize, groups: usize) -> Init {
    let fan_in = (in_channels / groups).max(1) * kernel_size.max(1) * kernel_size.max(1);
    let bound = 1.0 / (fan_in as f64).sqrt();
    Init::Uniform {
        lo: -bound,
        up: bound,
    }
}
