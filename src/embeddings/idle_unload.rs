//! Idle unload for the in-process ONNX models: embedders, cross-encoders and NER.
//!
//! A loaded model keeps its weights and session arenas resident for the life of the process.
//! On a server that is the price of a fast query. On an edge device that sits idle most of the
//! day it is memory the rest of the device needs, so `VELD_MODEL_IDLE_UNLOAD_SECS=<n>` drops a
//! model no call has used for `n` seconds, and the next call loads it again. Unset or `0` keeps
//! every model loaded once used, which is the server behaviour. No build feature or profile
//! marks a deployment as edge, so the operator of one sets the variable.
//!
//! A model is only dropped while no call holds it: callers hold an `Arc` for the length of an
//! inference, and the sweep drops the slot's copy only when it is the last one. Loads are
//! serialised per slot, so callers arriving at an unloaded model wait for one load.

use parking_lot::{Mutex, RwLock};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, OnceLock, Weak};
use std::time::{Duration, Instant};

/// Shortest and longest pause between sweeps; within them a sweep runs every quarter of the
/// idle period, so a model is dropped at most a quarter period late.
const MIN_SWEEP: Duration = Duration::from_secs(1);
const MAX_SWEEP: Duration = Duration::from_secs(60);

/// `VELD_MODEL_IDLE_UNLOAD_SECS`: how long a model may go unused before it is dropped. `None`
/// (unset, unparsable or 0) keeps models loaded.
pub(crate) fn idle_unload_after() -> Option<Duration> {
    static AFTER: OnceLock<Option<Duration>> = OnceLock::new();
    *AFTER.get_or_init(|| {
        std::env::var("VELD_MODEL_IDLE_UNLOAD_SECS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|&secs| secs > 0)
            .map(Duration::from_secs)
    })
}

static EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);

fn now_ms() -> u64 {
    EPOCH.elapsed().as_millis() as u64
}

enum SlotState<T> {
    Empty,
    Loaded(Arc<T>),
    /// A failed load is kept, as before idle unload existed: the shared registries replace an
    /// instance whose load failed rather than retrying it on every call.
    Failed(String),
}

/// One lazily loaded model that can be dropped while idle and loaded again on demand.
pub(crate) struct ModelSlot<T> {
    name: &'static str,
    state: RwLock<SlotState<T>>,
    loading: Mutex<()>,
    last_used_ms: AtomicU64,
    loads: AtomicU64,
}

impl<T: Send + Sync + 'static> ModelSlot<T> {
    /// An empty slot, swept by the process reaper when idle unload is configured.
    pub(crate) fn new(name: &'static str) -> Arc<Self> {
        let slot = Self::unswept(name);
        if let Some(idle) = idle_unload_after() {
            let weak: Weak<dyn Unload> = Arc::downgrade(&slot) as Weak<dyn Unload>;
            Reaper::global(idle).register(weak);
        }
        slot
    }

    /// An empty slot no reaper sweeps.
    pub(crate) fn unswept(name: &'static str) -> Arc<Self> {
        Arc::new(Self {
            name,
            state: RwLock::new(SlotState::Empty),
            loading: Mutex::new(()),
            last_used_ms: AtomicU64::new(now_ms()),
            loads: AtomicU64::new(0),
        })
    }

    /// The loaded model, loading it with `load` if the slot is empty. Concurrent callers on an
    /// empty slot wait for a single load.
    pub(crate) fn get_or_load(
        &self,
        load: impl FnOnce() -> Result<T, String>,
    ) -> Result<Arc<T>, String> {
        self.touch();
        if let Some(ready) = self.ready() {
            return ready;
        }
        let _loading = self.loading.lock();
        if let Some(ready) = self.ready() {
            return ready;
        }
        let previous_loads = self.loads.fetch_add(1, Ordering::Relaxed);
        let started = Instant::now();
        let loaded = load().map(Arc::new);
        *self.state.write() = match &loaded {
            Ok(model) => SlotState::Loaded(Arc::clone(model)),
            Err(e) => SlotState::Failed(e.clone()),
        };
        self.touch();
        if loaded.is_ok() {
            if previous_loads > 0 {
                tracing::info!(
                    model = self.name,
                    load_ms = started.elapsed().as_millis() as u64,
                    "reloaded model after idle unload"
                );
            }
            if idle_unload_after().is_some() {
                arm_exit_guard();
            }
        }
        loaded
    }

    fn ready(&self) -> Option<Result<Arc<T>, String>> {
        match &*self.state.read() {
            SlotState::Empty => None,
            SlotState::Loaded(model) => Some(Ok(Arc::clone(model))),
            SlotState::Failed(e) => Some(Err(e.clone())),
        }
    }

    fn touch(&self) {
        self.last_used_ms.store(now_ms(), Ordering::Relaxed);
    }

    /// Whether a model is resident now.
    pub(crate) fn is_loaded(&self) -> bool {
        matches!(&*self.state.read(), SlotState::Loaded(_))
    }

    /// Whether a load was attempted and failed.
    pub(crate) fn load_failed(&self) -> bool {
        matches!(&*self.state.read(), SlotState::Failed(_))
    }

    /// Drop the model if it is loaded, no call holds it, and at `now` (milliseconds on this
    /// module's clock) it has been unused for `idle`.
    fn unload_if_idle(&self, idle: Duration, now: u64) -> bool {
        let Some(mut state) = self.state.try_write() else {
            return false;
        };
        let SlotState::Loaded(model) = &*state else {
            return false;
        };
        // Clones are only taken under the read lock, which this write lock excludes, so a count
        // of one means no inference is holding the model.
        let unused_ms = now.saturating_sub(self.last_used_ms.load(Ordering::Relaxed));
        if Arc::strong_count(model) > 1 || u128::from(unused_ms) < idle.as_millis() {
            return false;
        }
        let model = std::mem::replace(&mut *state, SlotState::Empty);
        drop(state);
        drop(model);
        tracing::info!(
            model = self.name,
            idle_secs = unused_ms / 1000,
            "unloaded idle model"
        );
        true
    }
}

trait Unload: Send + Sync {
    fn unload_if_idle(&self, idle: Duration, now: u64) -> bool;
}

impl<T: Send + Sync + 'static> Unload for ModelSlot<T> {
    fn unload_if_idle(&self, idle: Duration, now: u64) -> bool {
        ModelSlot::unload_if_idle(self, idle, now)
    }
}

/// Set by the exit hook. Held across every sweep, so the hook waits out a drop in progress
/// and no drop starts after it.
static EXITING: Mutex<bool> = Mutex::new(false);

/// Destroying a session while `exit()` tears down ONNX Runtime's C++ statics reads freed
/// memory (see the pre-warm notes in `cross_encoder.rs`). Exit handlers run newest first and
/// ONNX Runtime registers its teardown inside the first session build, so this hook is armed
/// after a load has built one, which makes it run before that teardown.
fn arm_exit_guard() {
    static ARMED: AtomicBool = AtomicBool::new(false);
    if ARMED.swap(true, Ordering::AcqRel) {
        return;
    }
    extern "C" fn stop_unloading() {
        *EXITING.lock() = true;
    }
    // SAFETY: registers a plain `extern "C"` function with no captured state; it only takes a
    // static mutex, which is never dropped.
    if unsafe { libc::atexit(stop_unloading) } != 0 {
        tracing::warn!("could not register the model unload exit hook");
    }
}

/// Sweeps every registered slot on one background thread.
struct Reaper {
    slots: Mutex<Vec<Weak<dyn Unload>>>,
}

impl Reaper {
    fn global(idle: Duration) -> &'static Reaper {
        static REAPER: OnceLock<&'static Reaper> = OnceLock::new();
        REAPER.get_or_init(|| {
            let reaper: &'static Reaper = Box::leak(Box::new(Reaper {
                slots: Mutex::new(Vec::new()),
            }));
            let sweep_every = (idle / 4).clamp(MIN_SWEEP, MAX_SWEEP);
            if let Err(e) = std::thread::Builder::new()
                .name("veld-model-reaper".to_string())
                .spawn(move || loop {
                    std::thread::sleep(sweep_every);
                    if !reaper.sweep(idle, now_ms()) {
                        return;
                    }
                })
            {
                tracing::warn!("could not start the idle model reaper; models stay loaded: {e}");
            } else {
                tracing::info!(
                    idle_secs = idle.as_secs(),
                    "idle model unload enabled (VELD_MODEL_IDLE_UNLOAD_SECS)"
                );
            }
            reaper
        })
    }

    fn register(&self, slot: Weak<dyn Unload>) {
        self.slots.lock().push(slot);
    }

    /// One pass over the live slots; `false` once the process is exiting.
    fn sweep(&self, idle: Duration, now: u64) -> bool {
        let exiting = EXITING.lock();
        if *exiting {
            return false;
        }
        let live: Vec<Arc<dyn Unload>> = {
            let mut slots = self.slots.lock();
            slots.retain(|slot| slot.strong_count() > 0);
            slots.iter().filter_map(Weak::upgrade).collect()
        };
        for slot in live {
            slot.unload_if_idle(idle, now);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct Model(&'static str);

    fn load_ok(loads: &AtomicUsize) -> Result<Model, String> {
        loads.fetch_add(1, Ordering::SeqCst);
        Ok(Model("weights"))
    }

    const IDLE: Duration = Duration::from_secs(60);

    /// The clock reading `by` after now.
    fn later(by: Duration) -> u64 {
        now_ms() + by.as_millis() as u64
    }

    #[test]
    fn an_idle_model_is_dropped_and_the_next_call_loads_it_again() {
        let slot = ModelSlot::unswept("test");
        let loads = AtomicUsize::new(0);
        assert_eq!(slot.get_or_load(|| load_ok(&loads)).unwrap().0, "weights");
        assert!(slot.is_loaded());

        assert!(!slot.unload_if_idle(IDLE, now_ms()), "used just now");
        assert!(slot.unload_if_idle(IDLE, later(Duration::from_secs(61))));
        assert!(!slot.is_loaded());

        assert_eq!(slot.get_or_load(|| load_ok(&loads)).unwrap().0, "weights");
        assert_eq!(loads.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn a_model_held_by_an_inference_is_not_dropped() {
        let slot = ModelSlot::unswept("test");
        let loads = AtomicUsize::new(0);
        let in_flight = slot.get_or_load(|| load_ok(&loads)).unwrap();
        let idle_long_enough = later(Duration::from_secs(120));
        assert!(!slot.unload_if_idle(IDLE, idle_long_enough));
        assert!(slot.is_loaded());
        drop(in_flight);
        assert!(slot.unload_if_idle(IDLE, idle_long_enough));
    }

    #[test]
    fn callers_racing_on_an_empty_slot_share_one_load() {
        let slot = ModelSlot::unswept("test");
        let loads = Arc::new(AtomicUsize::new(0));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let slot = Arc::clone(&slot);
                let loads = Arc::clone(&loads);
                std::thread::spawn(move || {
                    slot.get_or_load(|| {
                        std::thread::sleep(Duration::from_millis(20));
                        load_ok(&loads)
                    })
                    .map(|m| m.0)
                })
            })
            .collect();
        for t in threads {
            assert_eq!(t.join().unwrap().unwrap(), "weights");
        }
        assert_eq!(loads.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_failed_load_is_kept_and_never_swept() {
        let slot: Arc<ModelSlot<Model>> = ModelSlot::unswept("test");
        assert_eq!(
            slot.get_or_load(|| Err("missing file".to_string())).err(),
            Some("missing file".to_string())
        );
        assert!(slot.load_failed());
        assert!(slot
            .get_or_load(|| unreachable!("a failure is not retried"))
            .is_err());
        assert!(!slot.unload_if_idle(IDLE, later(Duration::from_secs(120))));
        assert!(slot.load_failed());
    }

    #[test]
    fn a_reaper_sweep_drops_only_idle_slots_and_forgets_dropped_ones() {
        let reaper = Reaper {
            slots: Mutex::new(Vec::new()),
        };
        let loads = AtomicUsize::new(0);
        let idle = ModelSlot::unswept("idle");
        let busy = ModelSlot::unswept("busy");
        idle.get_or_load(|| load_ok(&loads)).unwrap();
        let in_flight = busy.get_or_load(|| load_ok(&loads)).unwrap();
        let gone = ModelSlot::<Model>::unswept("gone");
        for slot in [&idle, &busy, &gone] {
            reaper.register(Arc::downgrade(slot) as Weak<dyn Unload>);
        }
        drop(gone);

        assert!(reaper.sweep(IDLE, later(Duration::from_secs(120))));
        assert!(!idle.is_loaded());
        assert!(busy.is_loaded());
        assert_eq!(reaper.slots.lock().len(), 2);
        drop(in_flight);
    }
}
