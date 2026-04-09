mod cost;

use crate::ArrayIndex;
use cost::NodeCost;

#[derive(Debug)]
enum Direction {
    Incoming,
    Outgoing,
}

#[derive(Debug)]
struct Edge<C> {
    neighbor: ArrayIndex,
    cost: C,
    direction: Direction,
}

#[derive(Debug)]
struct Node<C> {
    id: usize,
    position: ArrayIndex,
    cost: C,
    edges: std::collections::HashMap<ArrayIndex, Edge<C>>,
}

#[derive(Debug)]
struct Network<C> {
    counter: usize,
    nodes: std::collections::HashMap<ArrayIndex, Node<C>>,
    /// Priority queue to sort nodes by cost
    queue: std::collections::BinaryHeap<NodeCost<C>>,
}

impl<C> Network<C>
where
    C: std::cmp::Ord + Clone + PartialEq,
    NodeCost<C>: Ord,
{
    pub(super) fn new() -> Self {
        Network {
            counter: 0,
            nodes: std::collections::HashMap::new(),
            queue: std::collections::BinaryHeap::new(),
        }
    }

    pub(super) fn add_node(
        &mut self,
        index: ArrayIndex,
        cost: C,
        origin: Option<ArrayIndex>,
    ) -> usize {
        let id = self.counter;

        self.nodes
            .entry(index)
            .and_modify(|node| {
                // If the node already exists, we update its cost and edges
                if cost < node.cost {
                    node.cost = cost.clone();
                }
                if let Some(o) = origin {
                    node.edges.entry(o).or_insert(Edge {
                        neighbor: o,
                        cost: cost.clone(),
                        direction: Direction::Incoming,
                    });
                }
            })
            .or_insert(Node {
                id: id.clone(),
                position: index,
                cost: cost.clone(),
                edges: match origin {
                    Some(o) => {
                        let mut edges = std::collections::HashMap::new();
                        edges.insert(
                            o,
                            Edge {
                                neighbor: o,
                                cost: cost.clone(),
                                direction: Direction::Incoming,
                            },
                        );
                        edges
                    }
                    None => std::collections::HashMap::new(),
                },
            });

        // Insert in the priority queue
        self.queue.push(NodeCost {
            index: index,
            cost: cost.clone(),
            estimated_cost: cost,
        });
        self.counter += 1;
        id
    }

    /// Extract the cheapest node
    ///
    /// Return the cheapest node while removing it from the queue.
    /// Note that one node can be reached from multiple edges, and each
    /// edge results in one entry in the queue. Therefore, the queue
    /// can lead to a node already removed.
    fn pop(&mut self) -> Option<&Node<C>> {
        while let Some(node_ij) = self.queue.pop() {
            if let Some(node) = self.nodes.get(&node_ij.index) {
                // If it exists, we can return it
                return Some(node);
            }
        }
        None
    }

    /*
    pub fn get_neighbors(&self, id: usize) -> Option<&Vec<Edge>> {
        self.nodes.get(&id).map(|node| &node.edges)
    }
    */
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Adding the first node to an empty network
    fn first_node() {
        let mut network = Network::new();
        let id = network.add_node(ArrayIndex::new(10, 10), 0, None);
        assert_eq!(id, 0);
        assert!(network.nodes.contains_key(&ArrayIndex::new(10, 10)));

        let next = network.pop().unwrap();
        assert_eq!(next.position, ArrayIndex::new(10, 10));
    }

    #[test]
    fn test_add_sequence_of_nodes() {
        let mut network = Network::new();
        // First node
        let id1 = network.add_node(ArrayIndex::new(10, 10), 0, None);
        // Next node
        let id2 = network.add_node(ArrayIndex::new(9, 9), 3, Some(ArrayIndex::new(10, 10)));
        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert!(network.nodes.contains_key(&ArrayIndex::new(10, 10)));
        assert!(network.nodes.contains_key(&ArrayIndex::new(9, 9)));
    }

    #[test]
    fn retrieve_cheapest_node() {
        let mut network = Network::new();
        network.add_node(ArrayIndex::new(10, 10), 100, None);
        network.add_node(ArrayIndex::new(9, 9), 50, None);
        network.add_node(ArrayIndex::new(8, 8), 75, None);

        let next = network.pop().unwrap();
        assert_eq!(next.position, ArrayIndex::new(9, 9));
        assert_eq!(next.cost, 50);
    }

    #[test]
    fn multiple_edges_to_node() {
        let mut network = Network::new();
        network.add_node(ArrayIndex::new(10, 10), 10, None);
        network.add_node(ArrayIndex::new(11, 11), 11, None);
        network.add_node(ArrayIndex::new(9, 9), 13, Some(ArrayIndex::new(10, 10)));
        network.add_node(ArrayIndex::new(9, 9), 12, Some(ArrayIndex::new(11, 11)));

        assert_eq!(network.nodes.len(), 3);
        assert!(network.nodes.contains_key(&ArrayIndex::new(9, 9)));
        let node = network.nodes.get(&ArrayIndex::new(9, 9)).unwrap();
        assert_eq!(node.edges.len(), 2);
        assert!(node.edges.contains_key(&ArrayIndex::new(10, 10)));
        assert_eq!(node.edges[&ArrayIndex::new(10, 10)].cost, 13);
        assert!(node.edges.contains_key(&ArrayIndex::new(11, 11)));
        assert_eq!(node.edges[&ArrayIndex::new(11, 11)].cost, 12);
    }
}
