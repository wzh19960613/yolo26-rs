/// Dataset sample ordering used by local train and eval loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleOrder {
    seed: u64,
    deterministic: bool,
    dataloader_reset_epoch: Option<usize>,
}

impl SampleOrder {
    /// Creates a sample order from Ultralytics-style `seed` and `deterministic`.
    ///
    /// `seed=0, deterministic=true` keeps the historical sequential order for
    /// non-epoch-aware callers. Training loops should use
    /// [`Self::dataset_index_for_epoch`] so that seed 0 still shuffles each
    /// epoch, matching Ultralytics' dataloader behavior.
    pub fn ultralytics(seed: u64, deterministic: bool) -> Self {
        Self {
            seed,
            deterministic,
            dataloader_reset_epoch: None,
        }
    }

    /// Returns a copy that simulates an Ultralytics `InfiniteDataLoader.reset()`
    /// before the given epoch. `close_mosaic` calls this reset when it disables
    /// mosaic augmentation, which consumes another DataLoader base seed before
    /// the next `RandomSampler` permutation.
    pub fn with_dataloader_reset_epoch(mut self, epoch: usize) -> Self {
        self.dataloader_reset_epoch = Some(epoch);
        self
    }

    /// Returns the configured seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns whether deterministic local ordering is enabled.
    pub fn deterministic(&self) -> bool {
        self.deterministic
    }

    /// Maps a logical sample index into the effective dataset range.
    ///
    /// This preserves the historical sequential default for non-training
    /// callers such as validation/eval loops.
    pub fn dataset_index(&self, logical_index: usize, effective_len: usize) -> usize {
        let sequential = logical_index % effective_len;
        if self.seed == 0 || !self.deterministic || effective_len <= 1 {
            return sequential;
        }
        let (stride, shift) = seeded_stride_and_shift(self.seed, effective_len);
        (sequential * stride + shift) % effective_len
    }

    /// Maps a logical sample index into an epoch-specific deterministic order.
    pub fn dataset_index_for_epoch(
        &self,
        logical_index: usize,
        effective_len: usize,
        epoch: usize,
    ) -> usize {
        self.epoch_indices(effective_len, epoch)[logical_index % effective_len]
    }

    /// Returns the epoch permutation used by Ultralytics' training dataloader.
    pub(crate) fn epoch_indices(&self, effective_len: usize, epoch: usize) -> Vec<usize> {
        if !self.deterministic || effective_len <= 1 {
            return (0..effective_len).collect();
        }
        pytorch_dataloader_epoch_permutation(effective_len, epoch, self.dataloader_reset_epoch)
    }
}

fn pytorch_dataloader_epoch_permutation(
    len: usize,
    epoch: usize,
    dataloader_reset_epoch: Option<usize>,
) -> Vec<usize> {
    let mut rng = TorchMt19937::new(ULTRALYTICS_DATALOADER_SEED);
    // PyTorch DataLoader consumes a base seed when the iterator is created.
    rng.random64();
    for visible_epoch in 0..=epoch {
        if dataloader_reset_epoch == Some(visible_epoch) {
            rng.random64();
        }
        let indices = pytorch_randperm(len, &mut rng);
        if visible_epoch == epoch {
            return indices;
        }
        if dataloader_reset_epoch != Some(visible_epoch + 1) {
            // PyTorch's RandomSampler still evaluates the empty
            // `randperm(n)[:0]` remainder when `num_samples == n`. With the
            // persistent Ultralytics iterator, that happens at the next epoch
            // boundary before the following visible batch is yielded.
            let _ = pytorch_randperm(len, &mut rng);
        }
    }
    unreachable!("inclusive range always returns for current epoch")
}

fn pytorch_randperm(n: usize, rng: &mut TorchMt19937) -> Vec<usize> {
    let mut result = (0..n).collect::<Vec<_>>();
    for i in 0..n.saturating_sub(1) {
        let z = (rng.random() as usize) % (n - i);
        result.swap(i, i + z);
    }
    result
}

const ULTRALYTICS_DATALOADER_SEED: u64 = 6_148_914_691_236_517_204;
const MERSENNE_STATE_N: usize = 624;
const MERSENNE_STATE_M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UMASK: u32 = 0x8000_0000;
const LMASK: u32 = 0x7fff_ffff;

struct TorchMt19937 {
    state: [u32; MERSENNE_STATE_N],
    left: usize,
    next: usize,
}

impl TorchMt19937 {
    fn new(seed: u64) -> Self {
        let mut state = [0u32; MERSENNE_STATE_N];
        state[0] = seed as u32;
        for j in 1..MERSENNE_STATE_N {
            state[j] = 1_812_433_253u32
                .wrapping_mul(state[j - 1] ^ (state[j - 1] >> 30))
                .wrapping_add(j as u32);
        }
        Self {
            state,
            left: 1,
            next: 0,
        }
    }

    fn random64(&mut self) -> u64 {
        let left = self.random() as u64;
        let right = self.random() as u64;
        (left << 32) | right
    }

    fn random(&mut self) -> u32 {
        self.left -= 1;
        if self.left == 0 {
            self.next_state();
        }
        let mut y = self.state[self.next];
        self.next += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    fn next_state(&mut self) {
        self.left = MERSENNE_STATE_N;
        self.next = 0;
        for p in 0..(MERSENNE_STATE_N - MERSENNE_STATE_M) {
            self.state[p] = self.state[p + MERSENNE_STATE_M]
                ^ twist(
                    mix_bits(self.state[p], self.state[p + 1]),
                    self.state[p + 1],
                );
        }
        for p in (MERSENNE_STATE_N - MERSENNE_STATE_M)..(MERSENNE_STATE_N - 1) {
            self.state[p] = self.state[p + MERSENNE_STATE_M - MERSENNE_STATE_N]
                ^ twist(
                    mix_bits(self.state[p], self.state[p + 1]),
                    self.state[p + 1],
                );
        }
        self.state[MERSENNE_STATE_N - 1] = self.state[MERSENNE_STATE_M - 1]
            ^ twist(
                mix_bits(self.state[MERSENNE_STATE_N - 1], self.state[0]),
                self.state[0],
            );
    }
}

fn mix_bits(left: u32, right: u32) -> u32 {
    (left & UMASK) | (right & LMASK)
}

fn twist(mixed: u32, value: u32) -> u32 {
    (mixed >> 1) ^ if value & 1 != 0 { MATRIX_A } else { 0 }
}

impl Default for SampleOrder {
    fn default() -> Self {
        Self::ultralytics(0, true)
    }
}

fn seeded_stride_and_shift(seed: u64, len: usize) -> (usize, usize) {
    let len_u64 = len as u64;
    let mut stride = ((seed / len_u64) % len_u64) as usize;
    if stride == 0 {
        stride = 1;
    }
    while gcd(stride, len) != 1 {
        stride += 1;
        if stride == len {
            stride = 1;
        }
    }
    let mut shift = (seed % len_u64) as usize;
    if stride == 1 && shift == 0 && len > 1 {
        shift = 1;
    }
    (stride, shift)
}

fn gcd(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let rem = left % right;
        left = right;
        right = rem;
    }
    left
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_non_epoch_order_is_sequential() {
        let order = SampleOrder::default();
        let indices: Vec<_> = (0..8).map(|idx| order.dataset_index(idx, 8)).collect();
        assert_eq!(indices, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn default_epoch_order_shuffles_and_changes_per_epoch() {
        let order = SampleOrder::default();
        let epoch0: Vec<_> = (0..17)
            .map(|idx| order.dataset_index_for_epoch(idx, 17, 0))
            .collect();
        let epoch1: Vec<_> = (0..17)
            .map(|idx| order.dataset_index_for_epoch(idx, 17, 1))
            .collect();

        assert_ne!(epoch0, (0..17).collect::<Vec<_>>());
        assert_ne!(epoch0, epoch1);

        let mut sorted = epoch0;
        sorted.sort_unstable();
        assert_eq!(sorted, (0..17).collect::<Vec<_>>());
    }

    #[test]
    fn epoch_order_matches_ultralytics_dataloader_randperm() {
        let order = SampleOrder::default();
        let indices = order.epoch_indices(160, 0);
        assert_eq!(
            &indices[..16],
            vec![
                143, 60, 148, 5, 125, 107, 68, 113, 151, 73, 106, 149, 23, 2, 150, 56
            ]
        );
        assert_eq!(
            &indices[144..],
            vec![
                52, 8, 31, 20, 74, 34, 50, 33, 44, 129, 15, 85, 59, 47, 90, 7
            ]
        );

        let indices = order.epoch_indices(160, 1);
        assert_eq!(
            &indices[..16],
            vec![
                47, 54, 24, 127, 41, 15, 159, 115, 149, 152, 90, 64, 18, 131, 107, 99
            ]
        );
        assert_eq!(
            &indices[144..],
            vec![
                136, 94, 63, 10, 66, 30, 157, 147, 71, 135, 81, 22, 85, 77, 53, 133
            ]
        );

        let indices = order.epoch_indices(160, 2);
        assert_eq!(
            &indices[..16],
            vec![
                91, 120, 111, 116, 75, 78, 94, 50, 55, 74, 108, 58, 30, 133, 26, 57
            ]
        );
        assert_eq!(
            &indices[144..],
            vec![
                62, 17, 100, 135, 35, 8, 36, 118, 77, 11, 125, 122, 145, 114, 33, 64
            ]
        );
    }

    #[test]
    fn epoch_order_matches_ultralytics_after_close_mosaic_reset() {
        let order = SampleOrder::default().with_dataloader_reset_epoch(0);
        let indices = (0..16)
            .map(|idx| order.dataset_index_for_epoch(idx, 160, 0))
            .collect::<Vec<_>>();
        assert_eq!(
            indices,
            vec![
                156, 109, 75, 118, 4, 138, 143, 73, 60, 16, 2, 136, 28, 102, 122, 34
            ]
        );
    }

    #[test]
    fn nondeterministic_epoch_order_is_sequential() {
        let order = SampleOrder::ultralytics(123, false);
        let indices: Vec<_> = (0..8)
            .map(|idx| order.dataset_index_for_epoch(idx, 8, 7))
            .collect();
        assert_eq!(indices, (0..8).collect::<Vec<_>>());
    }
}
