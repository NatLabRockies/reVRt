//! Support cost for a node in a network graph.
use std::cmp::Ordering;

#[derive(Debug)]
/// Cost for a node in a network graph
///
/// Provides support to compare nodes based on their cost and estimated cost.
// Consider renaming to NodePriority
pub(super) struct NodeCost<I, C> {
    pub(super) index: I,
    /// Cost (cheapest) to reach this node from the start node.
    pub(super) cost: C,
    /// Estimated cost to reach the goal from this node.
    /// This is not used now, but preparing for A* algorithm.
    pub(super) estimated_cost: C,
}

impl<I, C: PartialEq> PartialEq for NodeCost<I, C> {
    fn eq(&self, other: &Self) -> bool {
        self.estimated_cost.eq(&other.estimated_cost) && self.cost.eq(&other.cost)
    }
}

impl<I, C: PartialEq> Eq for NodeCost<I, C> {}

impl<I> PartialOrd for NodeCost<I, u32> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<I> PartialOrd for NodeCost<I, u64> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<I> PartialOrd for NodeCost<I, i32> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<I> PartialOrd for NodeCost<I, i64> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<I> PartialOrd for NodeCost<I, f32> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<I> PartialOrd for NodeCost<I, f64> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/*
impl<T: Ord> PartialOrd for NodeCost<I,T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
*/

/// Priority order for the node cost.
///
/// Note that the order is reversed compared to the default ordering.
///
impl<I> Ord for NodeCost<I, u32> {
    fn cmp(&self, other: &Self) -> Ordering {
        match other.estimated_cost.cmp(&self.estimated_cost) {
            Ordering::Equal => other.cost.cmp(&self.cost),
            ordering => ordering,
        }
    }
}
impl<I> Ord for NodeCost<I, u64> {
    fn cmp(&self, other: &Self) -> Ordering {
        match other.estimated_cost.cmp(&self.estimated_cost) {
            Ordering::Equal => other.cost.cmp(&self.cost),
            ordering => ordering,
        }
    }
}
impl<I> Ord for NodeCost<I, i32> {
    fn cmp(&self, other: &Self) -> Ordering {
        match other.estimated_cost.cmp(&self.estimated_cost) {
            Ordering::Equal => other.cost.cmp(&self.cost),
            ordering => ordering,
        }
    }
}
impl<I> Ord for NodeCost<I, i64> {
    fn cmp(&self, other: &Self) -> Ordering {
        match other.estimated_cost.cmp(&self.estimated_cost) {
            Ordering::Equal => other.cost.cmp(&self.cost),
            ordering => ordering,
        }
    }
}
impl<I> Ord for NodeCost<I, f32> {
    fn cmp(&self, other: &Self) -> Ordering {
        match other.estimated_cost.total_cmp(&self.estimated_cost) {
            Ordering::Equal => other.cost.total_cmp(&self.cost),
            ordering => ordering,
        }
    }
}
impl<I> Ord for NodeCost<I, f64> {
    fn cmp(&self, other: &Self) -> Ordering {
        match other.estimated_cost.total_cmp(&self.estimated_cost) {
            Ordering::Equal => other.cost.total_cmp(&self.cost),
            ordering => ordering,
        }
    }
}
/*
impl<T: Ord> Ord for NodeCost<I,T> {
    fn cmp(&self, other: &Self) -> Ordering {
        match other.estimated_cost.cmp(&self.estimated_cost) {
            Ordering::Equal => self.cost.partial_cmp(&other.cost).unwrap(),
            s => s,
        }
    }
}
*/

#[cfg(test)]
mod test {
    use super::*;
    use crate::ArrayIndex;

    #[test]
    fn equal_integers() {
        let a = NodeCost {
            index: ArrayIndex::new(0, 0),
            cost: 5,
            estimated_cost: 10,
        };
        let b = NodeCost {
            index: ArrayIndex::new(1, 1),
            cost: 5,
            estimated_cost: 10,
        };
        assert!(a == b);
    }

    #[test]
    /// Order of estimated_cost leads to the comparison.
    fn gt_integer() {
        let a = NodeCost {
            index: ArrayIndex::new(0, 0),
            cost: 3,
            estimated_cost: 10,
        };
        let b = NodeCost {
            index: ArrayIndex::new(1, 1),
            cost: 5,
            estimated_cost: 7,
        };
        assert!(a < b);
    }

    #[test]
    fn cost_gt_integer() {
        let a = NodeCost {
            index: ArrayIndex::new(0, 0),
            cost: 5,
            estimated_cost: 10,
        };
        let b = NodeCost {
            index: ArrayIndex::new(1, 1),
            cost: 3,
            estimated_cost: 10,
        };
        assert!(a < b);
    }

    #[test]
    fn estimated_gt_integer() {
        let a = NodeCost {
            index: ArrayIndex::new(0, 0),
            cost: 3,
            estimated_cost: 10,
        };
        let b = NodeCost {
            index: ArrayIndex::new(1, 1),
            cost: 3,
            estimated_cost: 7,
        };
        assert!(a < b);
    }

    #[test]
    fn equal_floats() {
        let a = NodeCost {
            index: ArrayIndex::new(0, 0),
            cost: 5.0,
            estimated_cost: 10.0,
        };
        let b = NodeCost {
            index: ArrayIndex::new(1, 1),
            cost: 5.0,
            estimated_cost: 10.0,
        };
        assert!(a == b);
    }

    #[test]
    fn gt_float() {
        let a = NodeCost {
            index: ArrayIndex::new(0, 0),
            cost: 3.0,
            estimated_cost: 10.0,
        };
        let b = NodeCost {
            index: ArrayIndex::new(1, 1),
            cost: 5.0,
            estimated_cost: 7.0,
        };
        assert!(a < b);
    }

    #[test]
    fn cost_gt_float() {
        let a = NodeCost {
            index: ArrayIndex::new(0, 0),
            cost: 5.0,
            estimated_cost: 10.0,
        };
        let b = NodeCost {
            index: ArrayIndex::new(1, 1),
            cost: 3.0,
            estimated_cost: 10.0,
        };
        assert!(a < b);
    }

    #[test]
    fn binary_heap_integer() {
        let mut heap = std::collections::BinaryHeap::new();
        heap.push(NodeCost {
            index: ArrayIndex::new(0, 0),
            cost: 5,
            estimated_cost: 10,
        });
        heap.push(NodeCost {
            index: ArrayIndex::new(1, 0),
            cost: 3,
            estimated_cost: 10,
        });
        heap.push(NodeCost {
            index: ArrayIndex::new(0, 1),
            cost: 3,
            estimated_cost: 12,
        });
        assert_eq!(heap.pop().unwrap().index, ArrayIndex::new(1, 0));
        assert_eq!(heap.pop().unwrap().index, ArrayIndex::new(0, 0));
        assert_eq!(heap.pop().unwrap().index, ArrayIndex::new(0, 1));
    }

    #[test]
    fn binary_heap_float() {
        let mut heap = std::collections::BinaryHeap::new();
        heap.push(NodeCost {
            index: ArrayIndex::new(0, 0),
            cost: 5.0,
            estimated_cost: 10.0,
        });
        heap.push(NodeCost {
            index: ArrayIndex::new(1, 0),
            cost: 3.0,
            estimated_cost: 10.0,
        });
        heap.push(NodeCost {
            index: ArrayIndex::new(0, 1),
            cost: 3.0,
            estimated_cost: 12.0,
        });
        assert_eq!(heap.pop().unwrap().index, ArrayIndex::new(1, 0));
        assert_eq!(heap.pop().unwrap().index, ArrayIndex::new(0, 0));
        assert_eq!(heap.pop().unwrap().index, ArrayIndex::new(0, 1));
    }

    #[test]
    fn order_u64() {
        let a = NodeCost {
            index: ArrayIndex::new(0, 0),
            cost: 5_u64,
            estimated_cost: 10_u64,
        };
        let b = NodeCost {
            index: ArrayIndex::new(1, 1),
            cost: 3_u64,
            estimated_cost: 10_u64,
        };

        assert!(a < b);
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Less));
    }

    #[test]
    fn order_i32() {
        let a = NodeCost {
            index: ArrayIndex::new(0, 0),
            cost: 3_i32,
            estimated_cost: 10_i32,
        };
        let b = NodeCost {
            index: ArrayIndex::new(1, 1),
            cost: 5_i32,
            estimated_cost: 7_i32,
        };

        assert!(a < b);
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Less));
    }

    #[test]
    fn order_i64() {
        let a = NodeCost {
            index: ArrayIndex::new(0, 0),
            cost: 5_i64,
            estimated_cost: 10_i64,
        };
        let b = NodeCost {
            index: ArrayIndex::new(1, 1),
            cost: 3_i64,
            estimated_cost: 10_i64,
        };

        assert!(a < b);
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Less));
    }

    #[test]
    fn binary_heap_u64() {
        let mut heap = std::collections::BinaryHeap::new();
        heap.push(NodeCost {
            index: ArrayIndex::new(0, 0),
            cost: 5_u64,
            estimated_cost: 10_u64,
        });
        heap.push(NodeCost {
            index: ArrayIndex::new(1, 0),
            cost: 3_u64,
            estimated_cost: 10_u64,
        });
        heap.push(NodeCost {
            index: ArrayIndex::new(0, 1),
            cost: 3_u64,
            estimated_cost: 12_u64,
        });

        assert_eq!(heap.pop().unwrap().index, ArrayIndex::new(1, 0));
        assert_eq!(heap.pop().unwrap().index, ArrayIndex::new(0, 0));
        assert_eq!(heap.pop().unwrap().index, ArrayIndex::new(0, 1));
    }

    #[test]
    fn binary_heap_i32() {
        let mut heap = std::collections::BinaryHeap::new();
        heap.push(NodeCost {
            index: ArrayIndex::new(0, 0),
            cost: 5_i32,
            estimated_cost: 10_i32,
        });
        heap.push(NodeCost {
            index: ArrayIndex::new(1, 0),
            cost: 3_i32,
            estimated_cost: 10_i32,
        });
        heap.push(NodeCost {
            index: ArrayIndex::new(0, 1),
            cost: 3_i32,
            estimated_cost: 12_i32,
        });

        assert_eq!(heap.pop().unwrap().index, ArrayIndex::new(1, 0));
        assert_eq!(heap.pop().unwrap().index, ArrayIndex::new(0, 0));
        assert_eq!(heap.pop().unwrap().index, ArrayIndex::new(0, 1));
    }

    #[test]
    fn binary_heap_i64() {
        let mut heap = std::collections::BinaryHeap::new();
        heap.push(NodeCost {
            index: ArrayIndex::new(0, 0),
            cost: 5_i64,
            estimated_cost: 10_i64,
        });
        heap.push(NodeCost {
            index: ArrayIndex::new(1, 0),
            cost: 3_i64,
            estimated_cost: 10_i64,
        });
        heap.push(NodeCost {
            index: ArrayIndex::new(0, 1),
            cost: 3_i64,
            estimated_cost: 12_i64,
        });

        assert_eq!(heap.pop().unwrap().index, ArrayIndex::new(1, 0));
        assert_eq!(heap.pop().unwrap().index, ArrayIndex::new(0, 0));
        assert_eq!(heap.pop().unwrap().index, ArrayIndex::new(0, 1));
    }
}
