use std::io::{Read, Seek, SeekFrom, Write};

use tempfile::NamedTempFile;
use tracing::debug;

const NO_PARENT_SLOT: u64 = u64::MAX;

/// Fixed-width record persisted for a single spilled routing slot
///
/// The record stores the accumulated path cost together with the optional
/// parent slot index. `None` parents are encoded with `NO_PARENT_SLOT` so the
/// on-disk representation stays a constant 16 bytes per slot.
#[derive(Clone, Copy, Debug)]
struct SpillRecord {
    /// Accumulated routing cost stored for this spilled slot
    cost: u64,
    /// Parent slot index encoded as `NO_PARENT_SLOT` when absent
    parent_slot: u64,
}

impl SpillRecord {
    const RECORD_LEN: usize = 8 + 8;

    fn from_parts(cost: u64, parent_slot: Option<usize>) -> Self {
        let parent_slot = parent_slot
            .and_then(|slot| u64::try_from(slot).ok())
            .unwrap_or(NO_PARENT_SLOT);

        Self { cost, parent_slot }
    }

    fn parent_slot(self) -> Option<usize> {
        if self.parent_slot == NO_PARENT_SLOT {
            None
        } else {
            usize::try_from(self.parent_slot).ok()
        }
    }

    fn to_bytes(self) -> [u8; Self::RECORD_LEN] {
        let mut out = [0_u8; Self::RECORD_LEN];
        out[0..8].copy_from_slice(&self.cost.to_le_bytes());
        out[8..16].copy_from_slice(&self.parent_slot.to_le_bytes());
        out
    }

    fn from_bytes(bytes: [u8; Self::RECORD_LEN]) -> Self {
        let mut cost = [0_u8; 8];
        let mut parent_slot = [0_u8; 8];
        cost.copy_from_slice(&bytes[0..8]);
        parent_slot.copy_from_slice(&bytes[8..16]);

        Self {
            cost: u64::from_le_bytes(cost),
            parent_slot: u64::from_le_bytes(parent_slot),
        }
    }
}

/// Buffered backing store for long-range routing state
///
/// `SwapStore` batches slot writes in memory and only persists them to the
/// temporary file once the write buffer reaches `write_buffer_capacity`, or
/// when an explicit flush or read requires durable data. This reduces random
/// disk writes during Dijkstra expansion while still allowing callers to read
/// a consistent slot after pending buffered writes are drained.
#[derive(Debug)]
pub(super) struct SwapStore {
    /// Temporary swap file that receives flushed records
    file: NamedTempFile,
    /// In-memory queue of pending slot writes waiting to be flushed to disk
    write_buffer: Vec<(u64, SpillRecord)>, // (slot, values)
    /// Maximum buffered writes before `write_slot` forces a flush
    write_buffer_capacity: usize,
}

impl SwapStore {
    pub(super) fn new(write_buffer_capacity: usize) -> std::io::Result<Self> {
        let file = tempfile::Builder::new()
            .prefix("revrt-routing-swap-")
            .suffix(".bin")
            .tempfile()?;

        let write_buffer_capacity = write_buffer_capacity.max(1);
        debug!("Swap for Dijkstra graph at {:?}", file.path());
        debug!(
            "Swap buffer capacity set to {} entries",
            write_buffer_capacity
        );
        Ok(Self {
            file,
            write_buffer: Vec::with_capacity(write_buffer_capacity),
            write_buffer_capacity,
        })
    }

    fn slot_offset(slot: u64) -> std::io::Result<u64> {
        slot.checked_mul(SpillRecord::RECORD_LEN as u64)
            .ok_or_else(|| std::io::Error::other("swap slot offset overflow"))
    }

    pub(super) fn write_slot(
        &mut self,
        slot: usize,
        record: (u64, Option<usize>),
    ) -> std::io::Result<()> {
        let slot = u64::try_from(slot).map_err(|_| std::io::Error::other("slot overflow"))?;
        self.write_buffer
            .push((slot, SpillRecord::from_parts(record.0, record.1)));

        if self.write_buffer.len() >= self.write_buffer_capacity {
            self.flush()?;
        }

        Ok(())
    }

    pub(super) fn flush(&mut self) -> std::io::Result<()> {
        if self.write_buffer.is_empty() {
            return Ok(());
        } else {
            // Only log if buffer is non-empty
            debug!("Flushing {} entries to disk", self.write_buffer.len());
        }
        for (slot, record) in self.write_buffer.drain(..) {
            let offset = Self::slot_offset(slot)?;
            let file = self.file.as_file_mut();
            file.seek(SeekFrom::Start(offset))?;
            file.write_all(&record.to_bytes())?;
        }
        self.file.as_file_mut().flush()
    }

    pub(super) fn read_slot(&mut self, slot: usize) -> std::io::Result<(u64, Option<usize>)> {
        self.flush()?;

        let slot = u64::try_from(slot).map_err(|_| std::io::Error::other("slot overflow"))?;
        let offset = Self::slot_offset(slot)?;

        let file = self.file.as_file_mut();
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = [0_u8; SpillRecord::RECORD_LEN];
        file.read_exact(&mut bytes)?;
        let record = SpillRecord::from_bytes(bytes);
        Ok((record.cost, record.parent_slot()))
    }
}

impl Drop for SwapStore {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};
    use std::thread;

    use super::*;

    #[test]
    fn swap_store_reads_written_slot() {
        let mut swap = SwapStore::new(2).unwrap();

        swap.write_slot(42, (7, Some(55))).unwrap();
        swap.flush().unwrap();
        let restored = swap.read_slot(42).unwrap();

        assert_eq!(restored.0, 7);
        assert_eq!(restored.1, Some(55));
    }

    #[test]
    fn swap_store_read_flushes_buffered_writes() {
        let mut swap = SwapStore::new(8).unwrap();

        swap.write_slot(4, (11, Some(3))).unwrap();
        let restored = swap.read_slot(4).unwrap();

        assert_eq!(restored.0, 11);
        assert_eq!(restored.1, Some(3));
    }

    #[test]
    fn swap_store_isolates_parallel_instances() {
        let thread_count = 8;
        let start_barrier = Arc::new(Barrier::new(thread_count));
        let read_barrier = Arc::new(Barrier::new(thread_count));

        thread::scope(|scope| {
            let mut handles = Vec::new();
            for worker in 0..thread_count {
                let start_barrier = Arc::clone(&start_barrier);
                let read_barrier = Arc::clone(&read_barrier);
                handles.push(scope.spawn(move || {
                    let mut swap = SwapStore::new(1).unwrap();
                    start_barrier.wait();
                    swap.write_slot(0, (worker as u64, Some(worker))).unwrap();
                    swap.flush().unwrap();
                    read_barrier.wait();
                    swap.read_slot(0).unwrap()
                }));
            }

            for (worker, handle) in handles.into_iter().enumerate() {
                let restored = handle.join().unwrap();
                assert_eq!(restored.0, worker as u64);
                assert_eq!(restored.1, Some(worker));
            }
        });
    }
}
