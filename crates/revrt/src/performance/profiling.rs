use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

static PROFILING_ENABLED: AtomicBool = AtomicBool::new(false);
static PROFILE_DATA: OnceLock<Mutex<HashMap<&'static str, ProfileStat>>> = OnceLock::new();

#[derive(Clone, Copy, Debug, Default)]
struct ProfileStat {
    calls: u64,
    total: Duration,
    max: Duration,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileRecord {
    pub name: String,
    pub calls: u64,
    pub total: Duration,
    pub max: Duration,
}

impl ProfileRecord {
    pub fn average(&self) -> Duration {
        if self.calls == 0 {
            Duration::ZERO
        } else {
            Duration::from_secs_f64(self.total.as_secs_f64() / self.calls as f64)
        }
    }
}

#[must_use = "profiling scopes must be bound to a variable to measure the surrounding scope"]
pub struct ScopeGuard {
    name: Option<&'static str>,
    start: Instant,
}

impl ScopeGuard {
    fn disabled() -> Self {
        Self {
            name: None,
            start: Instant::now(),
        }
    }
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        let Some(name) = self.name else {
            return;
        };

        record(name, self.start.elapsed());
    }
}

fn profile_data() -> &'static Mutex<HashMap<&'static str, ProfileStat>> {
    PROFILE_DATA.get_or_init(|| Mutex::new(HashMap::new()))
}

fn record(name: &'static str, elapsed: Duration) {
    if !PROFILING_ENABLED.load(Ordering::Relaxed) {
        return;
    }

    let mut stats = profile_data()
        .lock()
        .expect("profiling store lock poisoned");
    let entry = stats.entry(name).or_default();
    entry.calls += 1;
    entry.total += elapsed;
    entry.max = entry.max.max(elapsed);
}

pub fn enable() {
    PROFILING_ENABLED.store(true, Ordering::Relaxed);
}

pub fn disable() {
    PROFILING_ENABLED.store(false, Ordering::Relaxed);
}

pub fn reset() {
    let mut stats = profile_data()
        .lock()
        .expect("profiling store lock poisoned");
    stats.clear();
}

pub fn scope(name: &'static str) -> ScopeGuard {
    if !PROFILING_ENABLED.load(Ordering::Relaxed) {
        return ScopeGuard::disabled();
    }

    ScopeGuard {
        name: Some(name),
        start: Instant::now(),
    }
}

pub fn snapshot() -> Vec<ProfileRecord> {
    let stats = profile_data()
        .lock()
        .expect("profiling store lock poisoned");
    let mut records = stats
        .iter()
        .map(|(name, stat)| ProfileRecord {
            name: (*name).to_string(),
            calls: stat.calls,
            total: stat.total,
            max: stat.max,
        })
        .collect::<Vec<_>>();
    records.sort_by_key(|record| std::cmp::Reverse(record.total));
    records
}

#[cfg(test)]
mod tests {
    use super::{disable, enable, reset, scope, snapshot};
    use std::time::Duration;

    #[test]
    fn collects_elapsed_time_for_enabled_scope() {
        reset();
        enable();
        {
            let _scope = scope("profiling::tests::collects_elapsed_time");
            std::thread::sleep(Duration::from_millis(1));
        }
        disable();

        let records = snapshot();
        let record = records
            .iter()
            .find(|record| record.name == "profiling::tests::collects_elapsed_time")
            .expect("missing profiling record");

        assert_eq!(record.calls, 1);
        assert!(record.total >= Duration::from_millis(1));
        assert!(record.max >= Duration::from_millis(1));
        assert!(record.average() >= Duration::from_millis(1));
    }
}
