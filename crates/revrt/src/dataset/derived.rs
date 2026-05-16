//! Derived dataset materialization helpers
//!
//! This module is responsible for materializing derived arrays in the swap
//! dataset for a single chunk at a time. It computes invariant and
//! length-dependent cost layers, builds the hard barrier mask, and writes
//! cumulative soft barrier masks that can be reused across retry states.

use std::sync::RwLock;

use ndarray::Array3;
use tracing::trace;
use zarrs::storage::{ReadableListableStorage, ReadableWritableListableStorage};

use super::LazySubset;
use super::reader::DerivedDataMaterializer;
use super::swap::SourceLayout;
use super::swap::cumulative_soft_barrier_mask_name;
use crate::cost::{BarrierLayer, CostFunction};

/// Writes derived chunk-level arrays into the swap dataset.
///
/// The writer keeps the source feature storage, the writable swap storage,
/// and the barrier groupings needed to derive all secondary arrays for a
/// given chunk. Barrier layers are separated from the cost function during
/// construction so the cost arrays and barrier masks can be derived
/// independently.
pub(super) struct DerivedDataWriter {
    /// Source storage containing the original feature arrays.
    source: ReadableListableStorage,
    /// Writable swap storage where derived arrays are materialized.
    swap: ReadableWritableListableStorage,
    /// Boolean materialization state indexed by band, row, and column chunk.
    swap_chunk_idx: RwLock<ndarray::Array3<bool>>,
    /// Barrier layers that always behave as hard exclusions.
    hard_barrier_layers: Vec<BarrierLayer>,
    /// Soft barrier layers grouped by importance in ascending retry order.
    pub(super) soft_barrier_groups: Vec<(u32, Vec<BarrierLayer>)>,
    /// Cost function stripped of barrier layers for numeric cost derivation.
    cost_function: CostFunction,
}

impl DerivedDataWriter {
    /// Create a writer for materializing derived swap data.
    ///
    /// # Arguments
    /// `layout`: Source layout whose chunk-grid dimensions determine the size
    ///           of the internal tracking array.
    /// `source`: Storage containing the original input feature arrays.
    /// `swap`: Writable storage that will receive derived arrays.
    /// `cost_function`: Full cost function definition, including barrier
    ///                  layers that will be separated into dedicated masks.
    ///
    /// # Returns
    /// A `DerivedDataWriter` configured to compute cost arrays and barrier
    /// masks for chunk-sized subsets while tracking which chunks have already
    /// been materialized.
    pub(super) fn new(
        layout: &SourceLayout,
        source: ReadableListableStorage,
        swap: ReadableWritableListableStorage,
        cost_function: CostFunction,
    ) -> Self {
        let hard_barrier_layers = cost_function.hard_barrier_layers();
        let soft_barrier_groups = cost_function.soft_barrier_groups();
        let cost_function = cost_function.without_barriers();
        let swap_chunk_idx = Array3::from_elem(
            (
                layout.chunk_grid_bands,
                layout.chunk_grid_rows,
                layout.chunk_grid_cols,
            ),
            false,
        )
        .into();

        Self {
            source,
            swap,
            swap_chunk_idx,
            hard_barrier_layers,
            soft_barrier_groups,
            cost_function,
        }
    }

    /// Materialize every derived array for a single chunk.
    ///
    /// This computes both cost layers and all barrier masks for the chunk
    /// identified by the chunk-grid coordinates `cb`, `ci`, and `cj`, then
    /// stores the results into the swap dataset.
    ///
    /// # Arguments
    /// `cb`: Chunk band index in the swap dataset.
    /// `ci`: Chunk row index in the swap dataset.
    /// `cj`: Chunk column index in the swap dataset.
    fn materialize_chunk(&self, cb: u64, ci: u64, cj: u64) {
        trace!("Creating a LazySubset for ({}, {}, {})", cb, ci, cj);

        let variable = zarrs::array::Array::open(self.swap.clone(), "/cost").unwrap();
        let subset = variable.chunk_subset(&[cb, ci, cj]).unwrap();
        let chunk_subset = zarrs::array_subset::ArraySubset::new_with_ranges(&[
            cb..(cb + 1),
            ci..(ci + 1),
            cj..(cj + 1),
        ]);
        let mut data = LazySubset::<f32>::new(self.source.clone(), subset.clone());

        self.calculate_chunk_cost_single_layer(cb, ci, cj, &mut data, &chunk_subset, true);
        self.calculate_chunk_cost_single_layer(cb, ci, cj, &mut data, &chunk_subset, false);
        self.calculate_chunk_hard_barrier_mask(&mut data, &subset, &chunk_subset);
        self.calculate_chunk_cumulative_soft_barrier_masks(&mut data, &subset, &chunk_subset);
    }

    /// Compute and store one of the two chunk cost arrays.
    ///
    /// The cost function is evaluated either for invariant terms or for
    /// length-dependent terms, and the result is written to the matching
    /// destination array in the swap dataset.
    ///
    /// # Arguments
    /// `cb`: Chunk band index in the swap dataset.
    /// `ci`: Chunk row index in the swap dataset.
    /// `cj`: Chunk column index in the swap dataset.
    /// `features`: Lazily loaded source features for the target chunk.
    /// `chunk_subset`: Chunk-shaped subset describing the destination region
    ///                 in the swap dataset.
    /// `is_invariant`: When true, compute only invariant cost terms;
    ///                 otherwise compute length-dependent terms.
    fn calculate_chunk_cost_single_layer(
        &self,
        cb: u64,
        ci: u64,
        cj: u64,
        features: &mut LazySubset<f32>,
        chunk_subset: &zarrs::array_subset::ArraySubset,
        is_invariant: bool,
    ) {
        let output;
        let layer_name;
        if is_invariant {
            trace!(
                "Calculating invariant cost for chunk ({}, {}, {})",
                cb, ci, cj
            );
            output = self.cost_function.compute(features, true);
            layer_name = "/cost_invariant";
        } else {
            trace!(
                "Calculating length-dependent cost for chunk ({}, {}, {})",
                cb, ci, cj
            );
            output = self.cost_function.compute(features, false);
            layer_name = "/cost";
        }

        trace!("Cost function: {:?}", self.cost_function);

        let cost = zarrs::array::Array::open(self.swap.clone(), layer_name).unwrap();
        cost.store_metadata().unwrap();
        let chunk_indices: Vec<u64> = vec![cb, ci, cj];
        trace!("Storing chunk at {:?}", chunk_indices);
        trace!("Target chunk subset: {:?}", chunk_subset);
        cost.store_chunks_ndarray(chunk_subset, output).unwrap();
    }

    /// Compute and store the hard barrier mask for a chunk.
    ///
    /// The output mask is the logical OR of all hard barrier layers. When no
    /// hard barrier layers are configured, an all-false mask is stored so the
    /// swap dataset maintains a consistent shape.
    ///
    /// # Arguments
    /// `features`: Lazily loaded source features for the target chunk.
    /// `subset`: Source-data subset used to evaluate the barrier layers.
    /// `chunk_subset`: Chunk-shaped subset describing where the output mask
    ///                 should be written in the swap dataset.
    fn calculate_chunk_hard_barrier_mask(
        &self,
        features: &mut LazySubset<f32>,
        subset: &zarrs::array_subset::ArraySubset,
        chunk_subset: &zarrs::array_subset::ArraySubset,
    ) {
        trace!("Calculating hard barrier mask for subset {:?}", subset);

        let output = if self.hard_barrier_layers.is_empty() {
            empty_bool_mask(subset)
        } else {
            let barrier_masks = self
                .hard_barrier_layers
                .iter()
                .map(|layer| crate::cost::build_single_barrier_layer(layer, features))
                .collect::<Vec<_>>();

            let mut output =
                ndarray::ArrayD::<bool>::from_elem(ndarray::IxDyn(barrier_masks[0].shape()), false);
            for mask in barrier_masks {
                ndarray::Zip::from(&mut output)
                    .and(mask.view())
                    .for_each(|out, value| *out = *out || *value);
            }
            output
        };

        let variable = zarrs::array::Array::open(self.swap.clone(), "/hard_barrier_mask").unwrap();
        variable.store_metadata().unwrap();
        variable.store_chunks_ndarray(chunk_subset, output).unwrap();
    }

    /// Compute and store cumulative soft barrier masks for every retry state.
    ///
    /// Soft barriers are grouped by importance. For each retry state, this
    /// method combines the masks for the remaining groups so routing can
    /// progressively relax the least important barriers while reusing the same
    /// derived dataset.
    ///
    /// # Arguments
    /// `features`: Lazily loaded source features for the target chunk.
    /// `subset`: Source-data subset used to evaluate the barrier layers.
    /// `chunk_subset`: Chunk-shaped subset describing where each output mask
    ///                 should be written in the swap dataset.
    fn calculate_chunk_cumulative_soft_barrier_masks(
        &self,
        features: &mut LazySubset<f32>,
        subset: &zarrs::array_subset::ArraySubset,
        chunk_subset: &zarrs::array_subset::ArraySubset,
    ) {
        trace!(
            "Calculating cumulative soft barrier masks for subset {:?}",
            subset
        );

        let empty_mask = empty_bool_mask(subset);
        let group_masks = self
            .soft_barrier_groups
            .iter()
            .map(|(_, layers)| {
                combine_barrier_layers_for_subset(layers, features, subset)
                    .unwrap_or_else(|| empty_mask.clone())
            })
            .collect::<Vec<_>>();

        for retry_state in 0..=self.soft_barrier_groups.len() {
            let layer_name = cumulative_soft_barrier_mask_name(retry_state);
            let target =
                zarrs::array::Array::open(self.swap.clone(), &format!("/{layer_name}")).unwrap();

            let mut output = empty_mask.clone();
            for mask in group_masks.iter().skip(retry_state) {
                ndarray::Zip::from(&mut output)
                    .and(mask.view())
                    .for_each(|out, value| *out = *out || *value);
            }

            target.store_metadata().unwrap();
            target.store_chunks_ndarray(chunk_subset, output).unwrap();
        }
    }
}

impl DerivedDataMaterializer for DerivedDataWriter {
    fn has_hard_barriers(&self) -> bool {
        !self.hard_barrier_layers.is_empty()
    }

    /// Materialize any missing derived chunks overlapping a subset.
    ///
    /// This method first determines which chunk-grid cells intersect the
    /// requested subset. Each chunk is checked under a read lock and, when
    /// still missing, rechecked under a write lock before invoking chunk
    /// materialization. This avoids duplicate work when multiple threads
    /// request the same chunk concurrently.
    ///
    /// # Arguments
    /// `array`: Swap array whose chunk grid is used to map the subset to
    ///          chunk indices.
    /// `subset`: Requested array subset that may span one or more chunks.
    fn ensure_derived_data_for_subset(
        &self,
        array: &zarrs::array::Array<dyn zarrs::storage::ReadableStorageTraits>,
        subset: &zarrs::array_subset::ArraySubset,
    ) {
        let chunks = &array.chunks_in_array_subset(subset).unwrap().unwrap();
        trace!("Derived-data chunks: {:?}", chunks);
        trace!(
            "Derived-data subset extends to {:?} chunks",
            chunks.num_elements_usize()
        );

        for cb in chunks.start()[0]..(chunks.start()[0] + chunks.shape()[0]) {
            for ci in chunks.start()[1]..(chunks.start()[1] + chunks.shape()[1]) {
                for cj in chunks.start()[2]..(chunks.start()[2] + chunks.shape()[2]) {
                    trace!(
                        "Checking if derived data for chunk ({}, {}, {}) has been calculated",
                        cb, ci, cj
                    );
                    if self.swap_chunk_idx.read().unwrap()[[cb as usize, ci as usize, cj as usize]]
                    {
                        trace!(
                            "Derived data for chunk ({}, {}, {}) already calculated",
                            cb, ci, cj
                        );
                        continue;
                    }

                    let mut chunk_idx = self
                        .swap_chunk_idx
                        .write()
                        .expect("Failed to acquire write lock");
                    if chunk_idx[[cb as usize, ci as usize, cj as usize]] {
                        trace!(
                            "Derived data for chunk ({}, {}, {}) already calculated while waiting for the lock",
                            cb, ci, cj
                        );
                    } else {
                        self.materialize_chunk(cb, ci, cj);
                        chunk_idx[[cb as usize, ci as usize, cj as usize]] = true;
                        trace!(
                            "Recorded derived data for chunk ({}, {}, {}) as calculated. Total number of computed chunks: {}",
                            cb,
                            ci,
                            cj,
                            chunk_idx.iter().filter(|&&value| value).count()
                        );
                    }
                }
            }
        }
    }
}

/// Create an all-false boolean mask for a subset shape.
///
/// # Arguments
/// `subset`: Subset whose shape should be mirrored in the output mask.
///
/// # Returns
/// A boolean array with the same dimensionality as `subset`, initialized to
/// `false` in every cell.
fn empty_bool_mask(subset: &zarrs::array_subset::ArraySubset) -> ndarray::ArrayD<bool> {
    ndarray::ArrayD::<bool>::from_elem(
        ndarray::IxDyn(
            &subset
                .shape()
                .iter()
                .map(|&dim| usize::try_from(dim).expect("subset dimension exceeds usize range"))
                .collect::<Vec<_>>(),
        ),
        false,
    )
}

/// Combine multiple barrier layers into a single mask for a subset.
///
/// The returned mask is the logical OR of every barrier layer evaluated for
/// the provided subset. If `barrier_layers` is empty, the function returns
/// `None` so callers can decide how to handle the absence of barriers.
///
/// # Arguments
/// `barrier_layers`: Barrier layer definitions to evaluate.
/// `features`: Lazily loaded source features for the target subset.
/// `subset`: Subset whose shape should be used for the output mask.
///
/// # Returns
/// `Some(mask)` containing the combined barrier mask when at least one layer
/// is provided, or `None` when there is nothing to combine.
fn combine_barrier_layers_for_subset(
    barrier_layers: &[BarrierLayer],
    features: &mut LazySubset<f32>,
    subset: &zarrs::array_subset::ArraySubset,
) -> Option<ndarray::ArrayD<bool>> {
    if barrier_layers.is_empty() {
        return None;
    }

    let barrier_masks = barrier_layers
        .iter()
        .map(|layer| crate::cost::build_single_barrier_layer(layer, features))
        .collect::<Vec<_>>();
    let mut output = empty_bool_mask(subset);
    for mask in barrier_masks {
        ndarray::Zip::from(&mut output)
            .and(mask.view())
            .for_each(|out, value| *out = *out || *value);
    }

    Some(output)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use ndarray::{ArrayD, IxDyn};
    use tempfile::TempDir;
    use zarrs::array::Array;
    use zarrs::array_subset::ArraySubset;
    use zarrs::filesystem::FilesystemStore;
    use zarrs::storage::ReadableListableStorage;

    use super::*;
    use crate::dataset::samples;
    use crate::dataset::samples::{FillStrategy, LayerConfig, ZarrTestBuilder};
    use crate::dataset::swap::{initialize_swap, inspect_source_layout};

    fn hard_barrier_a(_band: u64, row: u64, col: u64) -> f32 {
        if row == 0 && col == 0 { 1.0 } else { 0.0 }
    }

    fn hard_barrier_b(_band: u64, row: u64, col: u64) -> f32 {
        if row == 0 && col == 1 { 1.0 } else { 0.0 }
    }

    fn soft_barrier_low(_band: u64, row: u64, col: u64) -> f32 {
        if row == 1 && col == 0 { 1.0 } else { 0.0 }
    }

    fn soft_barrier_high(_band: u64, row: u64, col: u64) -> f32 {
        if row == 1 && col == 1 { 1.0 } else { 0.0 }
    }

    fn make_source_store() -> (TempDir, ReadableListableStorage) {
        let source_tmp = ZarrTestBuilder::new()
            .dimensions(1, 4, 4)
            .chunks(1, 2, 2)
            .layer(LayerConfig::constant("cost_length", 2.0))
            .layer(LayerConfig::constant("cost_invariant_src", 3.0))
            .layer(LayerConfig::new(
                "hard_barrier_a",
                FillStrategy::Custom(hard_barrier_a),
            ))
            .layer(LayerConfig::new(
                "hard_barrier_b",
                FillStrategy::Custom(hard_barrier_b),
            ))
            .layer(LayerConfig::new(
                "soft_barrier_low",
                FillStrategy::Custom(soft_barrier_low),
            ))
            .layer(LayerConfig::new(
                "soft_barrier_high",
                FillStrategy::Custom(soft_barrier_high),
            ))
            .build()
            .unwrap();
        let source: ReadableListableStorage =
            Arc::new(FilesystemStore::new(source_tmp.path()).unwrap());

        (source_tmp, source)
    }

    fn read_subset_values<T: zarrs::array::ElementOwned + Clone>(
        store: &zarrs::storage::ReadableWritableListableStorage,
        path: &str,
        subset: &ArraySubset,
    ) -> Vec<T> {
        zarrs::array::Array::open(store.clone(), path)
            .unwrap()
            .retrieve_array_subset_elements(subset)
            .unwrap()
    }

    #[test]
    fn empty_bool_mask_matches_subset_shape() {
        let subset = ArraySubset::new_with_start_shape(vec![0, 1, 2], vec![1, 2, 3]).unwrap();

        let result = empty_bool_mask(&subset);

        assert_eq!(result.shape(), &[1, 2, 3]);
        assert!(result.iter().all(|value| !value));
    }

    #[test]
    fn combine_barrier_layers_returns_none_for_empty_input() {
        let (_source_tmp, source) = make_source_store();
        let subset = ArraySubset::new_with_start_shape(vec![0, 0, 0], vec![1, 2, 2]).unwrap();
        let mut features = LazySubset::<f32>::new(source, subset.clone());

        let result = combine_barrier_layers_for_subset(&[], &mut features, &subset);

        assert_eq!(result, None);
    }

    #[test]
    fn combine_barrier_layers_ors_masks_for_subset() {
        let (_source_tmp, source) = make_source_store();
        let subset = ArraySubset::new_with_start_shape(vec![0, 0, 0], vec![1, 2, 2]).unwrap();
        let mut features = LazySubset::<f32>::new(source.clone(), subset.clone());
        let cost_function = CostFunction::from_json(
            r#"{
                "cost_layers": [{"layer_name": "cost_length"}],
                "barrier_layers": [
                    {
                        "layer_name": "hard_barrier_a",
                        "barrier_operator": "eq",
                        "barrier_threshold": 1.0
                    },
                    {
                        "layer_name": "hard_barrier_b",
                        "barrier_operator": "eq",
                        "barrier_threshold": 1.0
                    }
                ]
            }"#,
        )
        .unwrap();
        let layers = cost_function.hard_barrier_layers();

        let result = combine_barrier_layers_for_subset(&layers, &mut features, &subset).unwrap();

        assert_eq!(
            result,
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![true, true, false, false],).unwrap(),
        );
    }

    #[test]
    fn materialize_chunk_writes_costs_and_barrier_masks() {
        let (_source_tmp, source) = make_source_store();
        let cost_function = CostFunction::from_json(
            r#"{
                "cost_layers": [
                    {"layer_name": "cost_length"},
                    {
                        "layer_name": "cost_invariant_src",
                        "is_invariant": true
                    }
                ],
                "barrier_layers": [
                    {
                        "layer_name": "hard_barrier_a",
                        "barrier_operator": "eq",
                        "barrier_threshold": 1.0
                    },
                    {
                        "layer_name": "hard_barrier_b",
                        "barrier_operator": "eq",
                        "barrier_threshold": 1.0
                    },
                    {
                        "layer_name": "soft_barrier_low",
                        "barrier_operator": "eq",
                        "barrier_threshold": 1.0,
                        "barrier_importance": 1
                    },
                    {
                        "layer_name": "soft_barrier_high",
                        "barrier_operator": "eq",
                        "barrier_threshold": 1.0,
                        "barrier_importance": 2
                    }
                ]
            }"#,
        )
        .unwrap();
        let layout = super::super::swap::inspect_source_layout(&source).unwrap();
        let swap_tmp = TempDir::new().unwrap();
        let swap = super::super::swap::initialize_swap(swap_tmp.path(), &layout, 2).unwrap();
        let writer = DerivedDataWriter::new(&layout, source, swap.clone(), cost_function);
        let subset = ArraySubset::new_with_start_shape(vec![0, 0, 0], vec![1, 2, 2]).unwrap();

        assert!(writer.has_hard_barriers());
        assert_eq!(writer.soft_barrier_groups.len(), 2);

        writer.materialize_chunk(0, 0);

        assert_eq!(
            read_subset_values::<f32>(&swap, "/cost", &subset),
            vec![2.0, 2.0, 2.0, 2.0],
        );
        assert_eq!(
            read_subset_values::<f32>(&swap, "/cost_invariant", &subset),
            vec![3.0, 3.0, 3.0, 3.0],
        );
        assert_eq!(
            read_subset_values::<bool>(&swap, "/hard_barrier_mask", &subset),
            vec![true, true, false, false],
        );
        assert_eq!(
            read_subset_values::<bool>(&swap, "/soft_barrier_mask_retry_0", &subset),
            vec![false, false, true, true],
        );
        assert_eq!(
            read_subset_values::<bool>(&swap, "/soft_barrier_mask_retry_1", &subset),
            vec![false, false, false, true],
        );
        assert_eq!(
            read_subset_values::<bool>(&swap, "/soft_barrier_mask_retry_2", &subset),
            vec![false, false, false, false],
        );
    }

    #[test]
    fn materialize_chunk_extracts_hard_barriers_and_preserves_costs() {
        let json = r#"
        {
            "cost_layers": [{"layer_name": "A"}],
            "barrier_layers": [
                {
                    "layer_name": "B",
                    "barrier_operator": "eq",
                    "barrier_threshold": 1.0
                }
            ]
        }
        "#;

        let source_dir = ZarrTestBuilder::new()
            .dimensions(1, 3, 3)
            .chunks(1, 3, 3)
            .layer(LayerConfig::sequential("A", 1))
            .layer(LayerConfig::new(
                "B",
                FillStrategy::Values(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
            ))
            .build()
            .expect("Error creating test zarr");
        let source: ReadableListableStorage =
            Arc::new(FilesystemStore::new(source_dir.path()).expect("could not open source"));
        let cost_function = CostFunction::from_json(json).unwrap();
        let layout = inspect_source_layout(&source).expect("Error inspecting source layout");
        let swap_dir = tempfile::TempDir::new().expect("could not create swap dir");
        let swap = initialize_swap(
            swap_dir.path(),
            &layout,
            cost_function.soft_barrier_groups().len(),
        )
        .expect("Error initializing swap dataset");
        let writer = DerivedDataWriter::new(&layout, source, swap.clone(), cost_function);

        assert!(writer.has_hard_barriers());

        writer.materialize_chunk(0, 0);

        let subset = ArraySubset::new_with_ranges(&[0..1, 0..3, 0..3]);
        let cost_values: Vec<f32> = Array::open(swap.clone(), "/cost")
            .expect("could not open derived cost array")
            .retrieve_array_subset_elements(&subset)
            .expect("could not read derived costs");
        let hard_barrier_mask: Vec<bool> = Array::open(swap, "/hard_barrier_mask")
            .expect("could not open hard barrier mask")
            .retrieve_array_subset_elements(&subset)
            .expect("could not read hard barrier mask");

        assert_eq!(
            cost_values,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );
        assert_eq!(
            hard_barrier_mask,
            vec![false, true, false, true, false, true, false, true, false]
        );
    }

    #[test]
    fn new_initializes_all_chunks_as_not_materialized() {
        let tmp = samples::multi_variable_random(1, 8, 8, 1, 4, 4, &["A"]);
        let source: ReadableListableStorage =
            Arc::new(FilesystemStore::new(tmp.path()).expect("could not open test store"));
        let layout = inspect_source_layout(&source).expect("source layout inspection failed");
        let swap_tmp = TempDir::new().expect("could not create swap dir");
        let swap = initialize_swap(swap_tmp.path(), &layout, 0)
            .expect("failed to initialize swap dataset");
        let cost_function = CostFunction::from_json(r#"{"cost_layers": [{"layer_name": "A"}]}"#)
            .expect("failed to construct cost function");

        let writer = DerivedDataWriter::new(&layout, source, swap, cost_function);
        let chunk_idx = writer
            .swap_chunk_idx
            .read()
            .expect("failed to acquire read lock");

        assert_eq!(chunk_idx.dim(), (2, 2));
        assert!(chunk_idx.iter().all(|&value| !value));
    }

    #[test]
    fn ensure_derived_data_for_subset_only_materializes_missing_chunks() {
        let tmp = samples::multi_variable_random(1, 8, 8, 1, 4, 4, &["A"]);
        let source: ReadableListableStorage =
            Arc::new(FilesystemStore::new(tmp.path()).expect("could not open test store"));
        let layout = inspect_source_layout(&source).expect("source layout inspection failed");
        let readable_source: Arc<dyn zarrs::storage::ReadableStorageTraits> = Arc::new(
            FilesystemStore::new(tmp.path()).expect("could not reopen readable test store"),
        );
        let array =
            zarrs::array::Array::open(readable_source, "/A").expect("failed to open source array");
        let swap_tmp = TempDir::new().expect("could not create swap dir");
        let swap = initialize_swap(swap_tmp.path(), &layout, 0)
            .expect("failed to initialize swap dataset");
        let cost_function = CostFunction::from_json(r#"{"cost_layers": [{"layer_name": "A"}]}"#)
            .expect("failed to construct cost function");
        let writer = DerivedDataWriter::new(&layout, source, swap, cost_function);
        let materialized = Mutex::new(Vec::new());

        let first_subset = ArraySubset::new_with_ranges(&[0..1, 1..7, 1..3]);
        writer.ensure_derived_data_for_subset(&array, &first_subset);
        {
            let chunk_idx = writer
                .swap_chunk_idx
                .read()
                .expect("failed to acquire read lock");
            for (ci, cj) in [(0, 0), (1, 0)] {
                if chunk_idx[[ci, cj]] {
                    materialized
                        .lock()
                        .expect("failed to record materialized chunk")
                        .push((ci as u64, cj as u64));
                }
            }
        }

        let second_subset = ArraySubset::new_with_ranges(&[0..1, 3..6, 2..7]);
        writer.ensure_derived_data_for_subset(&array, &second_subset);
        {
            let chunk_idx = writer
                .swap_chunk_idx
                .read()
                .expect("failed to acquire read lock");
            for (ci, cj) in [(0, 1), (1, 1)] {
                if chunk_idx[[ci, cj]] {
                    materialized
                        .lock()
                        .expect("failed to record materialized chunk")
                        .push((ci as u64, cj as u64));
                }
            }
        }

        writer.ensure_derived_data_for_subset(&array, &second_subset);

        assert_eq!(
            *materialized
                .lock()
                .expect("failed to read materialized chunks"),
            vec![(0, 0), (1, 0), (0, 1), (1, 1)]
        );

        let chunk_idx = writer
            .swap_chunk_idx
            .read()
            .expect("failed to acquire read lock");
        assert_eq!(*chunk_idx, Array2::from_elem((2, 2), true));
    }
}
