//! Barriers: locations the router should avoid
//!
//! A barrier flags cells of the routing domain that a path should not
//! cross. Barriers come in two kinds:
//!
//! - **Hard barriers** are impassable and are never crossed.
//! - **Soft barriers** are ranked by importance. When no valid path
//!   exists under the current restrictions, soft barriers can be
//!   progressively relaxed, dropping the least important ranks first,
//!   until a solution is found.
//!
//! This module is I/O agnostic: it does not care where the feature data
//! comes from, nor where the resulting barrier is stored or cached. It
//! only defines the procedures specific to the barrier concept, reading
//! features through the [`FeatureSource`] abstraction.
// TODO: remove once these types are wired into the rest of the library.
#![allow(dead_code)]

use std::collections::BTreeSet;

use futures::future::try_join_all;
use tracing::{trace, warn};

use crate::cost::components::BarrierOperator;
use crate::error::Result;
use crate::routing::features::FeatureSource;

/// Rank of a barrier, encoding how strongly a cell must be avoided
///
/// With `u8`, `0` is no barrier and the maximum value encodes a hard
/// barrier, leaving 254 soft barrier levels in between.
type BarrierRank = u8;

/// A multi-dimensional array of barrier ranks
type BarrierArray = ndarray::Array<BarrierRank, ndarray::Dim<ndarray::IxDynImpl>>;

/// A single barrier layer
///
/// A barrier layer defines a criterion over one feature variable that
/// decides, for each cell, whether that cell is a barrier. Cells meeting
/// the criterion carry this layer's `rank`; the rest are free. Multiple
/// layers can be combined into a single [`Barrier`].
///
/// # Note
///
/// This type is based on the previous barrier definition and is expected
/// to change in the future. It will keep backward compatibility for
/// loading from JSON, but its internal structure might change.
#[derive(Clone, Debug, serde::Deserialize)]
struct BarrierLayer {
    /// Name of the feature variable this barrier is derived from
    #[serde(rename = "layer_name")]
    name: String,
    /// Comparison applied between feature values and `threshold`
    #[serde(rename = "barrier_operator")]
    operator: BarrierOperator,
    /// Value the operator compares each feature value against
    #[serde(rename = "barrier_threshold")]
    threshold: f32,
    /// Barrier rank used to relax soft barriers
    ///
    /// Lower ranks are dropped first, so `rank == 1` is the first to be
    /// relaxed. `None` marks a hard (impassable) barrier that is never
    /// dropped.
    #[serde(rename = "barrier_importance")]
    rank: Option<BarrierRank>,
}

impl BarrierLayer {
    /// Warn once if a soft rank falls outside the valid range
    fn warn_on_out_of_range_rank(&self) {
        if let Some(rank) = self.rank
            && (rank == 0 || rank == BarrierRank::MAX)
        {
            warn!(
                "Barrier layer {:?} has rank {rank}, which lies outside the \
                 valid soft-barrier range 1..{}; rank 0 disables the barrier \
                 and {} collides with the hard-barrier sentinel",
                self.name,
                BarrierRank::MAX,
                BarrierRank::MAX,
            );
        }
    }

    /// Rank of this layer, or `None` for a hard barrier
    ///
    /// Ranks are ordered by increasing restriction: `0` is no barrier,
    /// `1` is the least important soft barrier (dropped first), and larger
    /// values are progressively harder to drop. `None` is a hard barrier
    /// that is never dropped.
    fn rank(&self) -> Option<BarrierRank> {
        self.rank
    }

    /// Evaluate the layer into a per-cell barrier array
    ///
    /// The result has the same shape as the input feature array. Each cell
    /// that meets the criterion carries this layer's rank; every other cell
    /// is `0` (no barrier). Hard barriers (`rank == None`) use
    /// `BarrierRank::MAX` so they always dominate soft ranks when combined.
    ///
    /// # Errors
    ///
    /// Returns an error if the feature source cannot provide this layer's
    /// variable.
    async fn compute<F>(&self, features: &F) -> Result<BarrierArray>
    where
        F: FeatureSource,
        F::Elem: From<f32> + PartialOrd + Copy,
    {
        trace!("Building barrier layer: {:?}", self);

        // Hard barriers carry no rank, so mark them as maximally important
        let rank = self.rank.unwrap_or(BarrierRank::MAX);
        let threshold = F::Elem::from(self.threshold);

        let array = features.get(&self.name).await?;
        Ok(array.mapv(|value| {
            let is_barrier = match self.operator {
                BarrierOperator::NotEqual => value != threshold,
                BarrierOperator::GreaterThan => value > threshold,
                BarrierOperator::GreaterThanOrEqual => value >= threshold,
                BarrierOperator::LessThan => value < threshold,
                BarrierOperator::LessThanOrEqual => value <= threshold,
                BarrierOperator::Equal => value == threshold,
            };
            if is_barrier { rank } else { 0 }
        }))
    }
}

/// Locations over the routing domain that must be avoided
///
/// A barrier marks, for each cell, how strongly routing should avoid it.
/// It is composed of one or many [`BarrierLayer`]s. Each layer decides,
/// per cell, whether that cell is a barrier and, if so, its rank, i.e. how
/// important the barrier is. The barrier combines its layers into an
/// n-dimensional array over the routing domain by keeping, per cell, the
/// most restrictive rank. Values encode the restriction: `0` is no
/// barrier, `1` is the least important soft barrier (the first that would
/// be dropped), higher values are progressively harder to drop, and the
/// maximum value of the array data type is a hard barrier that is never
/// dropped.
///
/// Keeping only the most restrictive rank is sufficient because ranks are
/// relaxed from the bottom up: while any higher rank remains, the cell
/// stays a barrier, so lower ranks are redundant and only the highest one
/// affects the outcome.
#[derive(Clone, Debug)]
struct Barrier {
    /// Layers making up the barrier, or `None` when no barrier applies
    layers: Option<Vec<BarrierLayer>>,
}

impl Barrier {
    /// Combine all layers into a single cell-wise maximum-rank array
    ///
    /// Returns `Ok(None)` when the barrier has no layers.
    ///
    /// # Errors
    ///
    /// Returns an error if any layer's feature cannot be provided.
    async fn compute<F>(&self, features: &F) -> Result<Option<BarrierArray>>
    where
        F: FeatureSource,
        F::Elem: From<f32> + PartialOrd + Copy,
    {
        let Some(layers) = self.layers.as_ref() else {
            return Ok(None);
        };

        let computed = try_join_all(layers.iter().map(|layer| layer.compute(features))).await?;

        let mut computed = computed.into_iter();
        let Some(first) = computed.next() else {
            return Ok(None);
        };
        Ok(Some(computed.fold(first, |mut acc, layer| {
            acc.zip_mut_with(&layer, |a, &b| *a = (*a).max(b));
            acc
        })))
    }

    /// Distinct barrier ranks in ascending order
    ///
    /// Hard barriers carry no rank and are reported as `BarrierRank::MAX`.
    fn ranks(&self) -> impl Iterator<Item = BarrierRank> {
        self.layers
            .iter()
            .flatten()
            .map(|layer| layer.rank().unwrap_or(BarrierRank::MAX))
            .collect::<BTreeSet<_>>()
            .into_iter()
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;
    use std::future::Future;

    use ndarray::{ArrayD, IxDyn};

    use super::*;
    use crate::error::Error;
    use crate::routing::features::FeatureArray;

    /// In-memory `FeatureSource` used to drive `compute` without any I/O
    struct MockFeatures {
        data: HashMap<String, FeatureArray<f32>>,
    }

    impl MockFeatures {
        fn new(entries: impl IntoIterator<Item = (&'static str, FeatureArray<f32>)>) -> Self {
            Self {
                data: entries
                    .into_iter()
                    .map(|(name, values)| (name.to_string(), values))
                    .collect(),
            }
        }

        fn empty() -> Self {
            Self {
                data: HashMap::new(),
            }
        }
    }

    impl FeatureSource for MockFeatures {
        type Elem = f32;

        fn get(&self, varname: &str) -> impl Future<Output = Result<FeatureArray<f32>>> + Send {
            let result = self.data.get(varname).cloned().ok_or_else(|| {
                Error::IO(std::io::Error::other(format!(
                    "missing variable '{varname}'"
                )))
            });
            async move { result }
        }
    }

    fn feature(values: [f32; 4]) -> FeatureArray<f32> {
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), values.to_vec()).unwrap()
    }

    /// In-memory `f64` `FeatureSource` used to exercise the generic path
    struct MockFeaturesF64 {
        data: HashMap<String, FeatureArray<f64>>,
    }

    impl MockFeaturesF64 {
        fn new(entries: impl IntoIterator<Item = (&'static str, FeatureArray<f64>)>) -> Self {
            Self {
                data: entries
                    .into_iter()
                    .map(|(name, values)| (name.to_string(), values))
                    .collect(),
            }
        }
    }

    impl FeatureSource for MockFeaturesF64 {
        type Elem = f64;

        fn get(&self, varname: &str) -> impl Future<Output = Result<FeatureArray<f64>>> + Send {
            let result = self.data.get(varname).cloned().ok_or_else(|| {
                Error::IO(std::io::Error::other(format!(
                    "missing variable '{varname}'"
                )))
            });
            async move { result }
        }
    }

    fn feature_f64(values: [f64; 4]) -> FeatureArray<f64> {
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), values.to_vec()).unwrap()
    }

    fn barriers(values: [BarrierRank; 4]) -> BarrierArray {
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), values.to_vec()).unwrap()
    }

    fn layer(
        name: &str,
        operator: BarrierOperator,
        threshold: f32,
        rank: Option<u8>,
    ) -> BarrierLayer {
        BarrierLayer {
            name: name.to_string(),
            operator,
            threshold,
            rank,
        }
    }

    #[tokio::test]
    async fn compute_flags_cells_for_each_operator() {
        let features = MockFeatures::new([("slope", feature([1.0, 2.0, 3.0, 4.0]))]);
        let cases = [
            (BarrierOperator::Equal, [0, 5, 0, 0]),
            (BarrierOperator::NotEqual, [5, 0, 5, 5]),
            (BarrierOperator::GreaterThan, [0, 0, 5, 5]),
            (BarrierOperator::GreaterThanOrEqual, [0, 5, 5, 5]),
            (BarrierOperator::LessThan, [5, 0, 0, 0]),
            (BarrierOperator::LessThanOrEqual, [5, 5, 0, 0]),
        ];

        for (operator, expected) in cases {
            let computed = layer("slope", operator, 2.0, Some(5))
                .compute(&features)
                .await
                .unwrap();
            assert_eq!(computed, barriers(expected), "operator {operator:?}");
        }
    }

    #[tokio::test]
    async fn compute_supports_f64_features() {
        let features = MockFeaturesF64::new([("slope", feature_f64([1.0, 2.0, 3.0, 4.0]))]);
        let computed = layer("slope", BarrierOperator::GreaterThan, 2.0, Some(5))
            .compute(&features)
            .await
            .unwrap();
        assert_eq!(computed, barriers([0, 0, 5, 5]));
    }

    #[tokio::test]
    async fn compute_uses_max_sentinel_for_hard_barrier() {
        let features = MockFeatures::new([("mask", feature([0.0, 1.0, 1.0, 0.0]))]);
        let computed = layer("mask", BarrierOperator::Equal, 1.0, None)
            .compute(&features)
            .await
            .unwrap();
        assert_eq!(
            computed,
            barriers([0, BarrierRank::MAX, BarrierRank::MAX, 0])
        );
    }

    #[tokio::test]
    async fn compute_uses_rank_value_for_soft_barrier() {
        let features = MockFeatures::new([("mask", feature([1.0, 0.0, 0.0, 0.0]))]);
        let computed = layer("mask", BarrierOperator::Equal, 1.0, Some(200))
            .compute(&features)
            .await
            .unwrap();
        assert_eq!(computed, barriers([200, 0, 0, 0]));
    }

    #[tokio::test]
    async fn compute_errors_when_feature_missing() {
        let features = MockFeatures::empty();
        let computed = layer("absent", BarrierOperator::Equal, 1.0, Some(1))
            .compute(&features)
            .await;
        assert!(computed.is_err());
    }

    #[tokio::test]
    async fn barrier_combines_layers_by_maximum() {
        let features = MockFeatures::new([
            ("low", feature([1.0, 1.0, 0.0, 0.0])),
            ("high", feature([0.0, 1.0, 1.0, 0.0])),
        ]);
        let barrier = Barrier {
            layers: Some(vec![
                layer("low", BarrierOperator::Equal, 1.0, Some(1)),
                layer("high", BarrierOperator::Equal, 1.0, Some(3)),
            ]),
        };

        let computed = barrier.compute(&features).await.unwrap();
        assert_eq!(computed, Some(barriers([1, 3, 3, 0])));
    }

    #[tokio::test]
    async fn barrier_combines_layers_with_equal_rank() {
        let features = MockFeatures::new([
            ("left", feature([1.0, 1.0, 0.0, 0.0])),
            ("right", feature([0.0, 1.0, 1.0, 0.0])),
        ]);
        let barrier = Barrier {
            layers: Some(vec![
                layer("left", BarrierOperator::Equal, 1.0, Some(4)),
                layer("right", BarrierOperator::Equal, 1.0, Some(4)),
            ]),
        };

        let computed = barrier.compute(&features).await.unwrap();
        assert_eq!(computed, Some(barriers([4, 4, 4, 0])));
    }

    #[tokio::test]
    async fn barrier_hard_barrier_dominates_soft() {
        let features = MockFeatures::new([
            ("soft", feature([1.0, 1.0, 0.0, 0.0])),
            ("hard", feature([1.0, 0.0, 0.0, 0.0])),
        ]);
        let barrier = Barrier {
            layers: Some(vec![
                layer("soft", BarrierOperator::Equal, 1.0, Some(7)),
                layer("hard", BarrierOperator::Equal, 1.0, None),
            ]),
        };

        let computed = barrier.compute(&features).await.unwrap();
        assert_eq!(computed, Some(barriers([BarrierRank::MAX, 7, 0, 0])));
    }

    #[tokio::test]
    async fn barrier_combination_is_order_independent() {
        let features = MockFeatures::new([
            ("a", feature([1.0, 1.0, 0.0, 0.0])),
            ("b", feature([0.0, 1.0, 1.0, 0.0])),
        ]);
        let first = layer("a", BarrierOperator::Equal, 1.0, Some(2));
        let second = layer("b", BarrierOperator::Equal, 1.0, Some(5));

        let forward = Barrier {
            layers: Some(vec![first.clone(), second.clone()]),
        };
        let reversed = Barrier {
            layers: Some(vec![second, first]),
        };

        assert_eq!(
            forward.compute(&features).await.unwrap(),
            reversed.compute(&features).await.unwrap()
        );
    }

    #[tokio::test]
    async fn barrier_single_layer_matches_layer_compute() {
        let features = MockFeatures::new([("mask", feature([1.0, 0.0, 1.0, 0.0]))]);
        let single = layer("mask", BarrierOperator::Equal, 1.0, Some(4));
        let barrier = Barrier {
            layers: Some(vec![single.clone()]),
        };

        assert_eq!(
            barrier.compute(&features).await.unwrap(),
            Some(single.compute(&features).await.unwrap())
        );
    }

    #[tokio::test]
    async fn barrier_without_layers_returns_none() {
        let features = MockFeatures::empty();
        let none = Barrier { layers: None };
        let empty = Barrier {
            layers: Some(vec![]),
        };

        assert!(none.compute(&features).await.unwrap().is_none());
        assert!(empty.compute(&features).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn barrier_errors_when_any_layer_missing() {
        let features = MockFeatures::new([("present", feature([1.0, 0.0, 0.0, 0.0]))]);
        let barrier = Barrier {
            layers: Some(vec![
                layer("present", BarrierOperator::Equal, 1.0, Some(1)),
                layer("absent", BarrierOperator::Equal, 1.0, Some(2)),
            ]),
        };

        assert!(barrier.compute(&features).await.is_err());
    }

    #[test]
    fn ranks_are_sorted_deduplicated_and_include_hard_barriers() {
        let barrier = Barrier {
            layers: Some(vec![
                layer("a", BarrierOperator::Equal, 1.0, Some(3)),
                layer("b", BarrierOperator::Equal, 1.0, Some(1)),
                layer("c", BarrierOperator::Equal, 1.0, Some(3)),
                layer("d", BarrierOperator::Equal, 1.0, None),
            ]),
        };

        let ranks = barrier.ranks().collect::<Vec<_>>();
        assert_eq!(ranks, vec![1, 3, BarrierRank::MAX]);
    }

    #[test]
    fn ranks_are_empty_without_layers() {
        assert_eq!(Barrier { layers: None }.ranks().count(), 0);
        assert_eq!(
            Barrier {
                layers: Some(vec![])
            }
            .ranks()
            .count(),
            0
        );
    }

    #[test]
    fn deserializes_soft_barrier_from_routing_payload() {
        let payload = r#"{
            "layer_name": "slope",
            "barrier_operator": "ge",
            "barrier_threshold": 15.0,
            "barrier_importance": 10
        }"#;

        let layer: BarrierLayer = serde_json::from_str(payload).unwrap();
        assert_eq!(layer.name, "slope");
        assert!(matches!(
            layer.operator,
            BarrierOperator::GreaterThanOrEqual
        ));
        assert_eq!(layer.threshold, 15.0);
        assert_eq!(layer.rank, Some(10));
    }

    #[test]
    fn deserializes_hard_barrier_without_rank() {
        let payload = r#"{
            "layer_name": "mask",
            "barrier_operator": "eq",
            "barrier_threshold": 1.0
        }"#;

        let layer: BarrierLayer = serde_json::from_str(payload).unwrap();
        assert_eq!(layer.rank, None);
    }
}
