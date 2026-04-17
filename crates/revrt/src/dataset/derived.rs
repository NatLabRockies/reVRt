use tracing::trace;
use zarrs::storage::{ReadableListableStorage, ReadableWritableListableStorage};

use super::LazySubset;
use super::swap::cumulative_soft_barrier_mask_name;
use crate::cost::{BarrierLayer, CostFunction};

pub(super) struct DerivedDataWriter {
    source: ReadableListableStorage,
    swap: ReadableWritableListableStorage,
    hard_barrier_layers: Vec<BarrierLayer>,
    soft_barrier_groups: Vec<(u32, Vec<BarrierLayer>)>,
    cost_function: CostFunction,
}

impl DerivedDataWriter {
    pub(super) fn new(
        source: ReadableListableStorage,
        swap: ReadableWritableListableStorage,
        cost_function: CostFunction,
    ) -> Self {
        let hard_barrier_layers = cost_function.hard_barrier_layers();
        let soft_barrier_groups = cost_function.soft_barrier_groups();
        let cost_function = cost_function.without_barriers();

        Self {
            source,
            swap,
            hard_barrier_layers,
            soft_barrier_groups,
            cost_function,
        }
    }

    pub(super) fn materialize_chunk(&self, ci: u64, cj: u64) {
        trace!("Creating a LazySubset for ({}, {})", ci, cj);

        let variable = zarrs::array::Array::open(self.swap.clone(), "/cost").unwrap();
        let subset = variable.chunk_subset(&[0, ci, cj]).unwrap();
        let chunk_subset =
            zarrs::array_subset::ArraySubset::new_with_ranges(&[0..1, ci..(ci + 1), cj..(cj + 1)]);
        let mut data = LazySubset::<f32>::new(self.source.clone(), subset.clone());

        self.calculate_chunk_cost_single_layer(ci, cj, &mut data, &chunk_subset, true);
        self.calculate_chunk_cost_single_layer(ci, cj, &mut data, &chunk_subset, false);
        self.calculate_chunk_hard_barrier_mask(&mut data, &subset, &chunk_subset);
        self.calculate_chunk_cumulative_soft_barrier_masks(&mut data, &subset, &chunk_subset);
    }

    pub(super) fn hard_barrier_layers(&self) -> &[BarrierLayer] {
        &self.hard_barrier_layers
    }

    pub(super) fn soft_barrier_group_count(&self) -> usize {
        self.soft_barrier_groups.len()
    }

    fn calculate_chunk_cost_single_layer(
        &self,
        ci: u64,
        cj: u64,
        features: &mut LazySubset<f32>,
        chunk_subset: &zarrs::array_subset::ArraySubset,
        is_invariant: bool,
    ) {
        let output;
        let layer_name;
        if is_invariant {
            trace!("Calculating invariant cost for chunk ({}, {})", ci, cj);
            output = self.cost_function.compute(features, true);
            layer_name = "/cost_invariant";
        } else {
            trace!(
                "Calculating length-dependent cost for chunk ({}, {})",
                ci, cj
            );
            output = self.cost_function.compute(features, false);
            layer_name = "/cost";
        }

        trace!("Cost function: {:?}", self.cost_function);

        let cost = zarrs::array::Array::open(self.swap.clone(), layer_name).unwrap();
        cost.store_metadata().unwrap();
        let chunk_indices: Vec<u64> = vec![0, ci, cj];
        trace!("Storing chunk at {:?}", chunk_indices);
        trace!("Target chunk subset: {:?}", chunk_subset);
        cost.store_chunks_ndarray(chunk_subset, output).unwrap();
    }

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
