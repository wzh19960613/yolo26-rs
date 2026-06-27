//! Deterministic RNG for reproducible augmentation (numpy MT19937).
use super::numpy_mt19937::NumpyMt19937;

pub(crate) struct SeededRng {
    mt: NumpyMt19937,
}

impl SeededRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            mt: NumpyMt19937::seeded(seed as u32),
        }
    }
    pub(crate) fn for_sample(seed: u64, sample_index: usize) -> Self {
        Self::new(seed.wrapping_add(sample_index as u64))
    }

    pub(crate) fn next_f32(&mut self) -> f32 {
        self.mt.next_f32()
    }
    pub(crate) fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
    pub(crate) fn bernoulli(&mut self, p: f32) -> bool {
        self.next_f32() < p.clamp(0.0, 1.0)
    }
    pub(crate) fn beta(&mut self, a: u32, b: u32) -> f64 {
        let x = self.gamma_sum(a);
        let y = self.gamma_sum(b);
        x / (x + y)
    }
    fn gamma_sum(&mut self, k: u32) -> f64 {
        let mut sum = 0.0_f64;
        for _ in 0..k {
            let u = self.next_f32() as f64;
            sum += -u.ln();
        }
        sum
    }
}
