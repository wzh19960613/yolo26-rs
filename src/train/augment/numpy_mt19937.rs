//! NumPy-compatible MT19937 RNG, matching `np.random.seed(N)` and
//! `np.random.uniform(-1, 1)`.
//!
//! This is needed so the native Rust augmentation pipeline consumes the same
//! random sequence as PyTorch/numpy, making the augmented images bit-identical
//! to the official Ultralytics dataloader (which calls `np.random.uniform`).

/// MT19937 state matching numpy's legacy `RandomState` (`np.random.seed(N)`).
///
/// Enables reproduction of numpy-seeded augmentation experiments from Rust.
/// The official Ultralytics dataloader consumes a global numpy RNG that
/// depends on worker count and iteration order, so multi-step training cannot
/// be bit-aligned across backends — but single-image experiments seeded with
/// `np.random.seed(N)` can be reproduced.
#[allow(dead_code)]
pub(crate) struct NumpyMt19937 {
    state: [u32; 624],
    index: usize,
}

#[allow(dead_code)]
impl NumpyMt19937 {
    /// Creates an RNG seeded exactly like `np.random.seed(seed)`.
    pub(crate) fn seeded(seed: u32) -> Self {
        let mut state = [0u32; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1812433253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self {
            state,
            index: 624, // forces a twist on first use
        }
    }

    fn twist(&mut self) {
        for i in 0..624 {
            let y = (self.state[i] & 0x80000000) | (self.state[(i + 1) % 624] & 0x7fffffff);
            self.state[i] = self.state[(i + 397) % 624] ^ (y >> 1);
            if y & 1 != 0 {
                self.state[i] ^= 0x9908b0df;
            }
        }
        self.index = 0;
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        // Tempering
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }

    /// Generates a `random()` in `[0, 1)` exactly like numpy's `np.random.random()`.
    ///
    /// numpy uses: `(a << 5 | b) * (1.0 / 9007199254740992.0)` where a, b are
    /// two 32-bit outputs (a is the "upper" 27 bits, b the "lower" 26 bits).
    fn random(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as u64; // 27 bits
        let b = (self.next_u32() >> 6) as u64; // 26 bits
        ((a << 26) | b) as f64 * (1.0 / 9007199254740992.0)
    }

    /// Generates `uniform(low, high)` like `np.random.uniform(low, high)`.
    pub(crate) fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.random()
    }

    /// Generates `uniform(low, high, size)` like `np.random.uniform(low, high, size)`.
    pub(crate) fn uniform_n(&mut self, low: f64, high: f64, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.uniform(low, high)).collect()
    }

    /// Generates `random() as f32` in `[0, 1)`, matching numpy's `np.random.random()` cast to f32.
    pub(crate) fn next_f32(&mut self) -> f32 {
        self.random() as f32
    }

    /// Bernoulli trial: returns `true` with probability `p`, matching
    /// `np.random.random() < p`.
    pub(crate) fn bernoulli(&mut self, p: f64) -> bool {
        self.random() < p
    }
}
