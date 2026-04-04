use std::sync::{Arc, Condvar, Mutex, MutexGuard};

#[derive(Debug)]
struct BudgetState {
    reserved_bytes: u64,
}

#[derive(Debug)]
pub(crate) struct BudgetReservation {
    coordinator: Arc<BudgetCoordinator>,
    reserved_bytes: u64,
}

impl BudgetReservation {
    pub(crate) fn resize(&mut self, target_bytes: u64) -> Option<()> {
        self.coordinator
            .resize_reservation(&mut self.reserved_bytes, target_bytes)
    }
}

impl Drop for BudgetReservation {
    fn drop(&mut self) {
        self.coordinator.release(self.reserved_bytes);
    }
}

#[derive(Debug)]
pub(crate) struct BudgetCoordinator {
    total_bytes: u64,
    state: Mutex<BudgetState>,
    budget_released: Condvar,
}

impl BudgetCoordinator {
    pub(crate) fn new(total_bytes: u64) -> Self {
        Self {
            total_bytes,
            state: Mutex::new(BudgetState { reserved_bytes: 0 }),
            budget_released: Condvar::new(),
        }
    }

    pub(crate) fn acquire(self: &Arc<Self>, bytes: u64) -> Option<BudgetReservation> {
        if bytes > self.total_bytes {
            return None;
        }

        let mut state = self.lock_state();
        state = self.wait_for_capacity(state, bytes);
        state.reserved_bytes += bytes;
        drop(state);

        Some(BudgetReservation {
            coordinator: Arc::clone(self),
            reserved_bytes: bytes,
        })
    }

    fn resize_reservation(&self, reserved_bytes: &mut u64, target_bytes: u64) -> Option<()> {
        if target_bytes > self.total_bytes {
            return None;
        }

        let current_bytes = *reserved_bytes;
        if target_bytes == current_bytes {
            return Some(());
        }

        let mut state = self.lock_state();
        if target_bytes > current_bytes {
            let additional_bytes = target_bytes - current_bytes;
            state = self.wait_for_capacity(state, additional_bytes);
            state.reserved_bytes += additional_bytes;
        } else {
            let released_bytes = current_bytes - target_bytes;
            state.reserved_bytes -= released_bytes;
            self.budget_released.notify_all();
        }
        *reserved_bytes = target_bytes;
        Some(())
    }

    fn release(&self, bytes: u64) {
        let mut state = self.lock_state();
        state.reserved_bytes -= bytes;
        drop(state);
        self.budget_released.notify_all();
    }

    fn lock_state(&self) -> MutexGuard<'_, BudgetState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn wait_for_capacity<'a>(
        &self,
        mut state: MutexGuard<'a, BudgetState>,
        requested_bytes: u64,
    ) -> MutexGuard<'a, BudgetState> {
        while self.total_bytes.saturating_sub(state.reserved_bytes) < requested_bytes {
            state = self
                .budget_released
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        state
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, mpsc};
    use std::thread;
    use std::time::Duration;

    use super::*;

    #[test]
    fn acquire_rejects_request_larger_than_total_budget() {
        let coordinator = Arc::new(BudgetCoordinator::new(10));

        let reservation = coordinator.acquire(11);

        assert!(reservation.is_none());
        assert_eq!(coordinator.lock_state().reserved_bytes, 0);
    }

    #[test]
    fn acquire_reserves_and_drop_releases_budget() {
        let coordinator = Arc::new(BudgetCoordinator::new(10));
        let reservation = coordinator.acquire(6).unwrap();

        assert_eq!(coordinator.lock_state().reserved_bytes, 6);

        drop(reservation);

        assert_eq!(coordinator.lock_state().reserved_bytes, 0);
    }

    #[test]
    fn resize_rejects_target_larger_than_total_budget() {
        let coordinator = Arc::new(BudgetCoordinator::new(10));
        let mut reservation = coordinator.acquire(4).unwrap();

        let resized = reservation.resize(11);

        assert!(resized.is_none());
        assert_eq!(reservation.reserved_bytes, 4);
        assert_eq!(coordinator.lock_state().reserved_bytes, 4);
    }

    #[test]
    fn resize_to_same_size_preserves_reservation() {
        let coordinator = Arc::new(BudgetCoordinator::new(10));
        let mut reservation = coordinator.acquire(4).unwrap();

        reservation.resize(4).unwrap();

        assert_eq!(reservation.reserved_bytes, 4);
        assert_eq!(coordinator.lock_state().reserved_bytes, 4);
    }

    #[test]
    fn resize_grows_when_capacity_is_available() {
        let coordinator = Arc::new(BudgetCoordinator::new(10));
        let mut reservation = coordinator.acquire(4).unwrap();

        reservation.resize(7).unwrap();

        assert_eq!(reservation.reserved_bytes, 7);
        assert_eq!(coordinator.lock_state().reserved_bytes, 7);
    }

    #[test]
    fn resize_shrinks_and_releases_capacity() {
        let coordinator = Arc::new(BudgetCoordinator::new(10));
        let mut reservation = coordinator.acquire(7).unwrap();

        reservation.resize(3).unwrap();

        assert_eq!(reservation.reserved_bytes, 3);
        assert_eq!(coordinator.lock_state().reserved_bytes, 3);
    }

    #[test]
    fn acquire_blocks_until_budget_is_released() {
        let coordinator = Arc::new(BudgetCoordinator::new(10));
        let reservation = coordinator.acquire(10).unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (acquired_tx, acquired_rx) = mpsc::channel();
        let thread_coordinator = Arc::clone(&coordinator);

        let handle = thread::spawn(move || {
            started_tx.send(()).unwrap();
            let reservation = thread_coordinator.acquire(5).unwrap();
            acquired_tx.send(()).unwrap();
            reservation
        });

        started_rx.recv().unwrap();
        assert!(acquired_rx.recv_timeout(Duration::from_millis(50)).is_err());

        drop(reservation);

        acquired_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        drop(handle.join().unwrap());
    }

    #[test]
    fn resize_blocks_until_capacity_is_released() {
        let coordinator = Arc::new(BudgetCoordinator::new(10));
        let mut primary = coordinator.acquire(6).unwrap();
        let secondary = coordinator.acquire(4).unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (resized_tx, resized_rx) = mpsc::channel();

        let primary = thread::scope(|scope| {
            let handle = scope.spawn(move || {
                started_tx.send(()).unwrap();
                primary.resize(8).unwrap();
                resized_tx.send(()).unwrap();
                primary
            });

            started_rx.recv().unwrap();
            assert!(resized_rx.recv_timeout(Duration::from_millis(50)).is_err());

            drop(secondary);

            resized_rx.recv_timeout(Duration::from_secs(1)).unwrap();
            handle.join().unwrap()
        });

        assert_eq!(coordinator.lock_state().reserved_bytes, 8);
        drop(primary);
    }
}
