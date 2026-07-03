use std::time::Duration;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileRecord {
    pub name: String,
    pub calls: u64,
    pub total: Duration,
    pub max: Duration,
}

impl ProfileRecord {
    #[inline(always)]
    pub fn average(&self) -> Duration {
        if self.calls == 0 {
            Duration::ZERO
        } else {
            Duration::from_secs_f64(self.total.as_secs_f64() / self.calls as f64)
        }
    }
}

pub struct ScopeGuard;

#[inline(always)]
pub fn enable() {}

#[inline(always)]
pub fn disable() {}

#[inline(always)]
pub fn reset() {}

#[inline(always)]
pub fn scope(_name: &'static str) -> ScopeGuard {
    ScopeGuard
}

#[inline(always)]
pub fn snapshot() -> Vec<ProfileRecord> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::{disable, enable, reset, scope, snapshot};

    #[test]
    fn profiling_api_is_a_noop_without_feature() {
        reset();
        enable();
        {
            let _scope = scope("profiling::tests::profiling_api_is_a_noop_without_feature");
        }
        disable();

        assert!(snapshot().is_empty());
    }
}
