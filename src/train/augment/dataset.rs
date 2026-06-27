//! Dataset wrapper that applies native augmentation in `sample()`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::apply::augment_sample;
use super::mixup::compose_mixup;
use super::mosaic::compose_mosaic;
use super::{AugmentConfig, Dataset, Sample, SeededRng};

/// Wraps a [`Dataset`] and applies native augmentation to each sampled item.
///
/// Construction is cheap; the wrapped dataset defers to `inner` for length and
/// raw loading, then applies HSV jitter, affine scale/translate, flips, mosaic
/// and mixup using a per-sample deterministic RNG derived from `seed` and the
/// sample index so training stays reproducible.
///
/// [`with_close_mosaic`] makes the wrapper epoch-aware: it holds a shared
/// epoch counter that the training loop advances each epoch, and during the
/// last `close_mosaic` epochs it forces `mosaic` and `mixup` to zero (matching
/// the official `close_mosaic` behavior), leaving HSV/affine/flip active.
pub struct AugmentingDataset<D> {
    inner: D,
    config: AugmentConfig,
    seed: u64,
    total_epochs: usize,
    close_mosaic_epochs: usize,
    current_epoch: Arc<AtomicUsize>,
    counter_epoch: Arc<AtomicUsize>,
    sample_counter: Arc<AtomicUsize>,
}

impl<D> AugmentingDataset<D> {
    /// Creates an augmenting wrapper with no close-mosaic window.
    pub fn new(inner: D, config: AugmentConfig, seed: u64) -> Self {
        Self {
            inner,
            config,
            seed,
            total_epochs: 0,
            close_mosaic_epochs: 0,
            current_epoch: Arc::new(AtomicUsize::new(0)),
            counter_epoch: Arc::new(AtomicUsize::new(usize::MAX)),
            sample_counter: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Configures the close-mosaic window: mosaic/mixup are disabled during the
    /// last `close_mosaic_epochs` of `total_epochs`.
    pub fn with_close_mosaic(mut self, total_epochs: usize, close_mosaic_epochs: usize) -> Self {
        self.total_epochs = total_epochs;
        self.close_mosaic_epochs = close_mosaic_epochs;
        self
    }

    /// Returns a handle the training loop updates each epoch so this wrapper can
    /// decide whether mosaic/mixup should be closed for the current epoch.
    pub fn epoch_handle(&self) -> Arc<AtomicUsize> {
        self.current_epoch.clone()
    }

    /// Returns a reference to the wrapped dataset.
    pub fn inner(&self) -> &D {
        &self.inner
    }

    /// Returns the effective mosaic probability for the current epoch.
    fn mosaic_probability(&self) -> f32 {
        if self.mosaic_closed() {
            0.0
        } else {
            self.config.mosaic
        }
    }

    fn mixup_probability(&self) -> f32 {
        if self.mosaic_closed() {
            0.0
        } else {
            self.config.mixup
        }
    }

    fn mosaic_closed(&self) -> bool {
        if self.total_epochs == 0
            || self.close_mosaic_epochs == 0
            || self.close_mosaic_epochs > self.total_epochs
        {
            return false;
        }
        let epoch = self.current_epoch.load(Ordering::Relaxed);
        epoch >= self.total_epochs - self.close_mosaic_epochs
    }

    fn next_sample_ordinal(&self) -> (usize, usize) {
        let epoch = self.current_epoch.load(Ordering::Relaxed);
        if self.counter_epoch.load(Ordering::Relaxed) != epoch {
            self.counter_epoch.store(epoch, Ordering::Relaxed);
            self.sample_counter.store(0, Ordering::Relaxed);
        }
        let ordinal = self.sample_counter.fetch_add(1, Ordering::Relaxed);
        (epoch, ordinal)
    }
}

fn stream_seed(seed: u64, epoch: usize, ordinal: usize, stream: u64) -> u64 {
    let mut value = seed
        ^ (epoch as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (ordinal as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ stream.wrapping_mul(0x94d0_49bb_1331_11eb);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn random_index(rng: &mut SeededRng, len: usize) -> usize {
    if len <= 1 {
        0
    } else {
        (rng.next_f32() * len as f32) as usize
    }
}

impl<D: Dataset> Dataset for AugmentingDataset<D> {
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn sample(&self, index: usize) -> crate::Result<Sample> {
        let primary = self.inner.sample(index)?;
        if self.config.is_identity() {
            return Ok(primary);
        }
        let (epoch, ordinal) = self.next_sample_ordinal();
        let mut geom_rng = SeededRng::new(stream_seed(self.seed, epoch, ordinal, 0));
        let mut color_rng = SeededRng::new(stream_seed(self.seed, epoch, ordinal, 1));
        let len = self.inner.len();
        let mosaic_p = self.mosaic_probability();
        let sample = if mosaic_p > 0.0
            && geom_rng.bernoulli(mosaic_p)
            && len >= 4
            && super::mosaic::compose_supported(&primary.target)
        {
            let s1 = self.inner.sample(random_index(&mut geom_rng, len))?;
            let s2 = self.inner.sample(random_index(&mut geom_rng, len))?;
            let s3 = self.inner.sample(random_index(&mut geom_rng, len))?;
            compose_mosaic([&primary, &s1, &s2, &s3])?
        } else {
            primary
        };
        let mixup_p = self.mixup_probability();
        let sample = if mixup_p > 0.0
            && geom_rng.bernoulli(mixup_p)
            && len >= 2
            && super::mosaic::compose_supported(&sample.target)
        {
            let other = self.inner.sample(random_index(&mut geom_rng, len))?;
            compose_mixup(sample, &other, &mut color_rng)?
        } else {
            sample
        };
        augment_sample(sample, &self.config, &mut geom_rng, &mut color_rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset(total_epochs: usize, close_mosaic_epochs: usize) -> AugmentingDataset<()> {
        AugmentingDataset::new((), AugmentConfig::default(), 0)
            .with_close_mosaic(total_epochs, close_mosaic_epochs)
    }

    #[test]
    fn close_mosaic_does_not_trigger_when_window_exceeds_epochs() {
        let dataset = dataset(3, 10);
        dataset
            .epoch_handle()
            .store(0, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            dataset.mosaic_probability(),
            AugmentConfig::default().mosaic
        );
        dataset
            .epoch_handle()
            .store(2, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            dataset.mosaic_probability(),
            AugmentConfig::default().mosaic
        );
    }

    #[test]
    fn close_mosaic_triggers_at_official_epoch_boundary() {
        let closed_from_start = dataset(10, 10);
        closed_from_start
            .epoch_handle()
            .store(0, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(closed_from_start.mosaic_probability(), 0.0);

        let closes_midway = dataset(20, 10);
        closes_midway
            .epoch_handle()
            .store(9, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            closes_midway.mosaic_probability(),
            AugmentConfig::default().mosaic
        );
        closes_midway
            .epoch_handle()
            .store(10, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(closes_midway.mosaic_probability(), 0.0);
    }
}
