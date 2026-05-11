use std::cell::Cell;

use crate::ArrayIndex;

pub(super) fn astar_successors<C, FN, IN>(
    index: &ArrayIndex,
    successors: &mut FN,
    min_cost: &Cell<Option<u64>>,
) -> Vec<(ArrayIndex, C)>
where
    C: Copy,
    FN: FnMut(&ArrayIndex) -> IN,
    IN: IntoIterator<Item = (ArrayIndex, C)>,
    u64: From<C>,
{
    let mut neighbors = Vec::with_capacity(16);
    let mut lowest_cost_from_neighbors: Option<u64> = None;
    for (neighbor, neighbor_cell_cost) in successors(index) {
        lowest_cost_from_neighbors = Some(match lowest_cost_from_neighbors {
            Some(curr_lowest_neighbor_cost) => {
                curr_lowest_neighbor_cost.min(u64::from(neighbor_cell_cost))
            }
            None => u64::from(neighbor_cell_cost),
        });
        neighbors.push((neighbor, neighbor_cell_cost));
    }
    if let Some(maybe_new_min_cost) = lowest_cost_from_neighbors {
        min_cost.set(Some(match min_cost.get() {
            Some(current_min_cost) => current_min_cost.min(maybe_new_min_cost),
            None => maybe_new_min_cost,
        }));
    }

    neighbors
}

pub(super) fn octile_heuristic<C>(
    index: &ArrayIndex,
    goals: &[ArrayIndex],
    min_cost: &Cell<Option<u64>>,
) -> C
where
    C: From<u64>,
{
    let estimated_min_cost = min_cost.get().unwrap_or(1);
    let heuristic = (octile_distance(index, goals) * estimated_min_cost as f64) as u64;
    C::from(heuristic)
}

fn octile_distance(start: &ArrayIndex, goals: &[ArrayIndex]) -> f64 {
    goals
        .iter()
        .map(|goal| {
            let di = start.i.abs_diff(goal.i) as f64;
            let dj = start.j.abs_diff(goal.j) as f64;
            let diagonal = di.min(dj);
            let straight = di.max(dj) - diagonal;
            straight + diagonal * std::f64::consts::SQRT_2
        })
        .min_by(|left, right| left.total_cmp(right))
        .unwrap_or(0.0)
}

#[allow(dead_code)]
/// Manhattan distance
///
/// For a given start point, calculates the shortest manhattan distance to a
/// collection of possible end points, i.e. assume that there are multiple
/// possible ends.
fn manhattan_distance(start: &ArrayIndex, end: &[ArrayIndex]) -> u64 {
    end.iter()
        .map(|end| {
            let di = start.i.abs_diff(end.i);
            let dj = start.j.abs_diff(end.j);
            di + dj
        })
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap()
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use crate::ArrayIndex;

    #[test]
    fn octile_distance_uses_shortest_goal() {
        let goals = [ArrayIndex::new_ij(3, 3), ArrayIndex::new_ij(8, 8)];
        let distance = super::octile_distance(&ArrayIndex::new_ij(0, 1), &goals);

        assert_eq!(distance, 1.0 + 2.0 * std::f64::consts::SQRT_2);
    }

    #[test]
    fn octile_heuristic_defaults_to_one_when_min_cost_unknown() {
        let min_cost = Cell::new(None);
        let goals = [ArrayIndex::new_ij(3, 3)];

        let heuristic =
            super::octile_heuristic::<u64>(&ArrayIndex::new_ij(0, 1), &goals, &min_cost);

        assert_eq!(heuristic, 3);
    }

    #[test]
    fn astar_successors_tracks_lowest_seen_cost() {
        let min_cost = Cell::new(Some(70));

        let neighbors = super::astar_successors(
            &ArrayIndex::new_ij(1, 1),
            &mut |_| {
                vec![
                    (ArrayIndex::new_ij(1, 2), 15_u64),
                    (ArrayIndex::new_ij(2, 2), 12),
                ]
            },
            &min_cost,
        );

        assert_eq!(neighbors.len(), 2);
        assert_eq!(min_cost.get(), Some(12));
    }
}
