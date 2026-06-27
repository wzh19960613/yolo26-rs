use candle_core::Tensor;
use candle_nn::VarBuilder;

use crate::yoloe::head::lrpc::head::{l2_normalize_last_dim, linear_last_dim};
use crate::yoloe::usage::EmbeddingTable;

/// Weights for the YOLOE RepRTA `Residual(SwiGLUFFN)` text-prompt adapter.
#[derive(Debug, Clone)]
pub struct Weights {
    /// First linear weight with shape `[2 * hidden, dim]`.
    pub w12_weight: Tensor,
    /// First linear bias with shape `[2 * hidden]`.
    pub w12_bias: Tensor,
    /// Output linear weight with shape `[dim, hidden]`.
    pub w3_weight: Tensor,
    /// Output linear bias with shape `[dim]`.
    pub w3_bias: Tensor,
}

/// Re-parameterizable Region-Text Alignment adapter.
#[derive(Debug, Clone)]
pub struct RepRta {
    /// Adapter weights.
    pub weights: Weights,
}

impl RepRta {
    /// Creates a RepRTA adapter from explicit weights.
    pub fn new(weights: Weights) -> crate::Result<Self> {
        let (hidden2, dim) = weights.w12_weight.dims2()?;
        let (out_dim, hidden) = weights.w3_weight.dims2()?;
        if hidden2 == 0 || hidden2 % 2 != 0 {
            return Err(crate::Error::InvalidTensor(
                "YOLOE RepRTA w12 weight rows must be a non-empty even number".to_string(),
            ));
        }
        if hidden2 / 2 != hidden {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE RepRTA hidden dim mismatch: w12 rows/2={} w3 cols={hidden}",
                hidden2 / 2
            )));
        }
        if out_dim != dim {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE RepRTA residual dim mismatch: input dim={dim} output dim={out_dim}"
            )));
        }
        if weights.w12_bias.dims() != [hidden2] {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE RepRTA w12 bias must have shape [{hidden2}], got {:?}",
                weights.w12_bias.dims()
            )));
        }
        if weights.w3_bias.dims() != [out_dim] {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE RepRTA w3 bias must have shape [{out_dim}], got {:?}",
                weights.w3_bias.dims()
            )));
        }
        Ok(Self { weights })
    }

    /// Loads official `reprta.m.w12` and `reprta.m.w3` weights.
    pub fn load(vb: VarBuilder) -> crate::Result<Self> {
        let w12 = vb.pp("m").pp("w12");
        let w3 = vb.pp("m").pp("w3");
        Self::new(Weights {
            w12_weight: w12.get_unchecked("weight")?,
            w12_bias: w12.get_unchecked("bias")?,
            w3_weight: w3.get_unchecked("weight")?,
            w3_bias: w3.get_unchecked("bias")?,
        })
    }

    /// Loads RepRTA when the `reprta.m.w12.weight` key is present in `vb`,
    /// otherwise returns `Ok(None)`. Use this during model construction so a
    /// checkpoint without RepRTA (e.g. prompt-free only) does not error.
    pub fn load_optional(vb: VarBuilder) -> crate::Result<Option<Self>> {
        if !vb.contains_tensor("m.w12.weight") {
            return Ok(None);
        }
        Self::load(vb).map(Some)
    }

    /// Loads or initializes RepRTA with an explicit `dim` and `hidden` size.
    ///
    /// Required when constructing from a trainable `VarMap` (which does not
    /// support shapeless `get_unchecked`). Weight shapes:
    /// `w12.weight = [2*hidden, dim]`, `w12.bias = [2*hidden]`,
    /// `w3.weight = [dim, hidden]`, `w3.bias = [dim]`.
    pub fn load_with_hidden(vb: VarBuilder, dim: usize, hidden: usize) -> crate::Result<Self> {
        if dim == 0 || hidden == 0 {
            return Err(crate::Error::InvalidConfig(
                "YOLOE RepRTA dim and hidden must be greater than zero".to_string(),
            ));
        }
        let w12 = vb.pp("m").pp("w12");
        let w3 = vb.pp("m").pp("w3");
        Self::new(Weights {
            w12_weight: w12.get((2 * hidden, dim), "weight")?,
            w12_bias: w12.get(2 * hidden, "bias")?,
            w3_weight: w3.get((dim, hidden), "weight")?,
            w3_bias: w3.get(dim, "bias")?,
        })
    }

    /// Applies the residual SwiGLU adapter to text embeddings.
    pub fn forward(&self, text_embeddings: &Tensor) -> crate::Result<Tensor> {
        let hidden2 = self.weights.w12_weight.dim(0)?;
        let hidden = hidden2 / 2;
        let x12 = linear_last_dim(
            text_embeddings,
            &self.weights.w12_weight,
            Some(&self.weights.w12_bias),
        )?;
        let x1 = x12.narrow(x12.rank() - 1, 0, hidden)?;
        let x2 = x12.narrow(x12.rank() - 1, hidden, hidden)?;
        let gated = x1.silu()?.broadcast_mul(&x2)?;
        let projected =
            linear_last_dim(&gated, &self.weights.w3_weight, Some(&self.weights.w3_bias))?;
        Ok(text_embeddings.broadcast_add(&projected)?)
    }

    /// Applies RepRTA and L2-normalizes prompt embeddings.
    pub fn forward_normalized(&self, text_embeddings: &Tensor) -> crate::Result<Tensor> {
        l2_normalize_last_dim(&self.forward(text_embeddings)?)
    }

    /// Applies RepRTA to a prompt table and returns a normalized table.
    pub fn forward_table(&self, table: &EmbeddingTable) -> crate::Result<EmbeddingTable> {
        EmbeddingTable::new(
            self.forward_normalized(&table.embeddings)?,
            table.classes.clone(),
        )
    }
}
