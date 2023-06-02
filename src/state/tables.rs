use crate::state::graph::Graph;
use crate::Ensemble;
use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{Hash, Hasher};

const NEW_VEC_USIZE: Vec<usize> = Vec::new();

#[rustfmt::skip]
mod unformatted {
    pub const SHARD0_DUAL: &[(usize, usize)] = &[(0, 1)];
    pub const SHARD2_DUAL: &[(usize, usize)] = &[(0, 2), (1, 3), (2, 3), (2, 3), (2, 3)];
    pub const FLIP2_DUAL: &[(usize, usize)] = &[(0, 6), (1, 7), (2, 6), (3, 7), (4, 6), (5, 7), (6, 7)];
    pub const FLIP3_DUAL: &[(usize, usize)] = &[(0, 6), (1, 8), (2, 7), (3, 6), (4, 8), (5, 7), (6, 8), (6, 7), (7, 8)];

    pub const SHARD0_VERTEX: &[(usize, usize)] = &[(0, 1), (1, 2), (0, 2)];
    pub const SHARD2_VERTEX: &[(usize, usize)] = &[(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)];
    pub const FLIP2_VERTEX: &[(usize, usize)] = &[(1, 2), (2, 3), (1, 3), (1, 4), (2, 4), (3, 4), (0, 1), (0, 2), (0, 3)];
    pub const FLIP3_VERTEX: &[(usize, usize)] = &[(1, 2), (2, 3), (1, 3), (1, 4), (2, 4), (3, 4), (0, 1), (0, 2), (0, 3), (0, 4)];

    pub const SHARD0_MIDDLE: &[(usize, usize)] = &[(0, 0), (0, 1), (0, 2)];
    pub const SHARD2_MIDDLE: &[(usize, usize)] = &[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 3), (2, 4), (3, 1), (3, 4), (3, 5), (4, 2), (4, 3), (4, 5)];
    pub const FLIP2_MIDDLE: &[(usize, usize)] = &[(0, 0), (0, 6), (0, 7), (1, 1), (1, 4), (1, 5), (2, 2), (2, 6), (2, 8), (3, 0), (3, 3), (3, 4), (4, 1), (4, 7), (4, 8), (5, 2), (5, 3), (5, 5), (6, 0), (6, 1), (6, 2)];
    pub const FLIP3_MIDDLE: &[(usize, usize)] = &[(0, 0), (0, 6), (0, 7), (1, 1), (1, 4), (1, 5), (2, 2), (2, 6), (2, 8), (3, 0), (3, 3), (3, 4), (4, 1), (4, 7), (4, 8), (5, 2), (5, 3), (5, 5), (6, 4), (6, 7), (6, 9), (7, 3), (7, 6), (7, 9), (8, 5), (8, 8), (8, 9)];
}
use unformatted::*;

#[derive(Debug, Clone)]
pub struct TreeTables {
    shard02: HashMap<TreeStateId, Vec<TreeStateId>>,
    shard20: HashMap<TreeStateId, Vec<TreeStateId>>,
    flip23: HashMap<TreeStateId, Vec<TreeStateId>>,
    flip32: HashMap<TreeStateId, Vec<TreeStateId>>,
}

impl TreeTables {
    pub fn new(ensemble: Ensemble) -> Self {
        if matches!(ensemble, Ensemble::Undecorated) {
            TreeTables {
                shard02: HashMap::new(),
                shard20: HashMap::new(),
                flip23: HashMap::new(),
                flip32: HashMap::new(),
            }
        } else {
            let shard0 = TreeState::valid_states(StateType::Shard0, ensemble);
            let shard2 = TreeState::valid_states(StateType::Shard2, ensemble);
            let flip2 = TreeState::valid_states(StateType::Flip2, ensemble);
            let flip3 = TreeState::valid_states(StateType::Flip3, ensemble);

            let (shard02, shard20) = transitions::<2, 32, 2>(shard0, shard2);
            let (flip23, flip32) = transitions::<128, 512, 6>(flip2, flip3);

            TreeTables {
                shard02,
                shard20,
                flip23,
                flip32,
            }
        }
    }

    pub fn destinations(
        &self,
        state_type: StateType,
        id: TreeStateId,
    ) -> Option<&Vec<TreeStateId>> {
        match state_type {
            StateType::Shard0 => self.shard02.get(&id),
            StateType::Shard2 => self.shard20.get(&id),
            StateType::Flip2 => self.flip23.get(&id),
            StateType::Flip3 => self.flip32.get(&id),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TreeState {
    triangles: Vec<bool>,
    edges: Vec<bool>,
}

#[derive(Debug, Copy, Clone)]
pub enum StateType {
    Shard0,
    Shard2,
    Flip2,
    Flip3,
}

impl StateType {
    pub fn middle_map(&self) -> &[(usize, usize)] {
        match *self {
            StateType::Shard0 => SHARD0_MIDDLE,
            StateType::Shard2 => SHARD2_MIDDLE,
            StateType::Flip2 => FLIP2_MIDDLE,
            StateType::Flip3 => FLIP3_MIDDLE,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Connectivity {
    dual: Vec<Option<usize>>,
    vertex: Vec<Option<usize>>,
    middle: Vec<Option<usize>>,
    boundary_edges: Vec<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TreeStateId {
    triangle: usize,
    edge: usize,
    n_triangles: usize,
    n_edges: usize,
}

impl TreeState {
    pub fn new(triangles: Vec<bool>, edges: Vec<bool>) -> Self {
        TreeState { triangles, edges }
    }

    pub fn triangles(&self) -> &Vec<bool> {
        &self.triangles
    }

    pub fn edges(&self) -> &Vec<bool> {
        &self.edges
    }

    pub fn id(&self, n_triangles: usize, n_edges: usize) -> TreeStateId {
        let mut triangle_id = 0;
        self.triangles
            .iter()
            .enumerate()
            .for_each(|(i, &triangle)| {
                if triangle {
                    triangle_id += 2_usize.pow(i.try_into().unwrap());
                }
            });
        let mut edge_id = 0;
        self.edges.iter().enumerate().for_each(|(i, &edge)| {
            if edge {
                edge_id += 2_usize.pow(i.try_into().unwrap());
            }
        });
        TreeStateId {
            triangle: triangle_id,
            edge: edge_id,
            n_triangles,
            n_edges,
        }
    }

    fn valid_states(state_type: StateType, ensemble: Ensemble) -> HashMap<TreeStateId, u64> {
        let mut map = HashMap::new();
        let (n_triangles, n_edges) = match state_type {
            StateType::Shard0 => (1, 3),
            StateType::Shard2 => (5, 6),
            StateType::Flip2 => (7, 9),
            StateType::Flip3 => (9, 10),
        };
        (0..(2_usize.pow(n_triangles.try_into().unwrap()))).for_each(|t| {
            (0..(2_usize.pow(n_edges.try_into().unwrap()))).for_each(|e| {
                let state_id = TreeStateId {
                    triangle: t,
                    edge: e,
                    n_triangles,
                    n_edges,
                };
                let state: TreeState = state_id.into();
                if let Some(conn) = state.connectivity(state_type, ensemble) {
                    map.insert(state_id, conn);
                }
            })
        });
        map
    }

    fn connectivity(&self, state_type: StateType, ensemble: Ensemble) -> Option<u64> {
        let dual_opt = self.dual_connectivity(state_type);
        let vertex_opt = self.vertex_connectivity(state_type);
        let middle_opt = self.middle_connectivity(state_type);
        let boundary_edges = match state_type {
            StateType::Shard0 | StateType::Shard2 => (0..3).map(|i| self.edges[i]).collect(),
            StateType::Flip2 | StateType::Flip3 => (0..9).map(|i| self.edges[i]).collect(),
        };
        match ensemble {
            Ensemble::Undecorated => unreachable!(),
            Ensemble::TripleTrees => {
                if let (Some(dual), Some(vertex), Some(middle)) = (dual_opt, vertex_opt, middle_opt)
                {
                    let conn = Connectivity {
                        dual,
                        vertex,
                        middle,
                        boundary_edges,
                    };
                    let mut hasher = DefaultHasher::new();
                    conn.hash(&mut hasher);
                    Some(hasher.finish())
                } else {
                    None
                }
            }

            Ensemble::Spanning => {
                if let (Some(dual), Some(vertex)) = (dual_opt, vertex_opt) {
                    let conn = Connectivity {
                        dual,
                        vertex,
                        middle: vec![],
                        boundary_edges,
                    };
                    let mut hasher = DefaultHasher::new();
                    conn.hash(&mut hasher);
                    Some(hasher.finish())
                } else {
                    None
                }
            }
        }
    }

    fn middle_connectivity(&self, state_type: StateType) -> Option<Vec<Option<usize>>> {
        let n_middle_edges = self
            .edges
            .iter()
            .enumerate()
            .filter(|(i, edge)| {
                !**edge
                    && match state_type {
                        StateType::Shard0 | StateType::Shard2 => (0..3).contains(i),
                        StateType::Flip2 | StateType::Flip3 => (0..9).contains(i),
                    }
            })
            .count();
        let boundary = (0..n_middle_edges).collect();

        let (graph, edge_count) = self.middle_graph(state_type);
        graph.connectivity(boundary, edge_count)
    }

    fn vertex_connectivity(&self, state_type: StateType) -> Option<Vec<Option<usize>>> {
        let boundary = match state_type {
            StateType::Shard0 | StateType::Shard2 => vec![0, 1, 2],
            StateType::Flip2 | StateType::Flip3 => vec![0, 1, 2, 3, 4],
        };

        let edge_count = self.vertex_edge_count();
        let graph = self.vertex_graph(state_type);
        graph.connectivity(boundary, edge_count)
    }

    fn dual_connectivity(&self, state_type: StateType) -> Option<Vec<Option<usize>>> {
        let boundary = match state_type {
            StateType::Shard0 | StateType::Shard2 => vec![0, 1],
            StateType::Flip2 | StateType::Flip3 => vec![0, 1, 2, 3, 4, 5],
        };
        let edge_count = self.dual_edge_count();
        let graph = self.dual_graph(state_type);
        graph.connectivity(boundary, edge_count)
    }

    fn dual_graph(&self, state_type: StateType) -> Graph {
        let skeleton = match state_type {
            StateType::Shard0 => SHARD0_DUAL,
            StateType::Shard2 => SHARD2_DUAL,
            StateType::Flip2 => FLIP2_DUAL,
            StateType::Flip3 => FLIP3_DUAL,
        };

        let n_nodes = match state_type {
            StateType::Shard0 => 2,
            StateType::Shard2 => 4,
            StateType::Flip2 => 8,
            StateType::Flip3 => 9,
        };

        let mut adj = vec![NEW_VEC_USIZE; n_nodes];
        skeleton.iter().enumerate().for_each(|(triangle, &(i, j))| {
            if self.triangles[triangle] {
                adj[i].push(j);
                adj[j].push(i);
            }
        });

        Graph::new(adj)
    }

    fn vertex_graph(&self, state_type: StateType) -> Graph {
        let skeleton = match state_type {
            StateType::Shard0 => SHARD0_VERTEX,
            StateType::Shard2 => SHARD2_VERTEX,
            StateType::Flip2 => FLIP2_VERTEX,
            StateType::Flip3 => FLIP3_VERTEX,
        };

        let n_nodes = match state_type {
            StateType::Shard0 => 3,
            StateType::Shard2 => 4,
            StateType::Flip2 => 5,
            StateType::Flip3 => 5,
        };

        let mut adj = vec![NEW_VEC_USIZE; n_nodes];
        skeleton.iter().enumerate().for_each(|(edge, &(i, j))| {
            if self.edges[edge] {
                adj[i].push(j);
                adj[j].push(i);
            }
        });

        Graph::new(adj)
    }

    fn middle_graph(&self, state_type: StateType) -> (Graph, usize) {
        let skeleton = match state_type {
            StateType::Shard0 => SHARD0_MIDDLE,
            StateType::Shard2 => SHARD2_MIDDLE,
            StateType::Flip2 => FLIP2_MIDDLE,
            StateType::Flip3 => FLIP3_MIDDLE,
        };

        let mut i = 0;
        let middle_edges = {
            let mut middle_edges = vec![];
            for edge in &self.edges {
                if !*edge {
                    middle_edges.push(Some(i));
                    i += 1;
                } else {
                    middle_edges.push(None);
                }
            }
            middle_edges
        };
        let middle_triangles = {
            let mut middle_triangles = vec![];
            for triangle in &self.triangles {
                if !*triangle {
                    middle_triangles.push(Some(i));
                    i += 1;
                } else {
                    middle_triangles.push(None);
                }
            }
            middle_triangles
        };

        let n_edges = middle_edges.iter().filter(|opt| opt.is_some()).count();
        let n_triangles = middle_triangles.iter().filter(|opt| opt.is_some()).count();

        let mut adj = vec![NEW_VEC_USIZE; n_edges + n_triangles];
        let mut edge_count = 0;
        skeleton.iter().for_each(|&(t, e)| {
            if let (Some(i), Some(j)) = (middle_edges[e], middle_triangles[t]) {
                adj[i].push(j);
                adj[j].push(i);
                edge_count += 1;
            }
        });
        (Graph::new(adj), edge_count)
    }

    fn dual_edge_count(&self) -> usize {
        self.triangles.iter().filter(|&triangle| *triangle).count()
    }

    fn vertex_edge_count(&self) -> usize {
        self.edges.iter().filter(|&edge| *edge).count()
    }
}

impl From<TreeStateId> for TreeState {
    fn from(value: TreeStateId) -> Self {
        let triangles = (0..(value.n_triangles))
            .map(|i| {
                value.triangle % 2_usize.pow((i + 1).try_into().unwrap())
                    >= 2_usize.pow(i.try_into().unwrap())
            })
            .collect::<Vec<bool>>();
        let edges = (0..(value.n_edges))
            .map(|i| {
                value.edge % 2_usize.pow((i + 1).try_into().unwrap())
                    >= 2_usize.pow(i.try_into().unwrap())
            })
            .collect::<Vec<bool>>();
        TreeState { triangles, edges }
    }
}

fn transitions<const N_SMALL: usize, const N_LARGE: usize, const B: usize>(
    states_small: HashMap<TreeStateId, u64>,
    states_large: HashMap<TreeStateId, u64>,
) -> (
    HashMap<TreeStateId, Vec<TreeStateId>>,
    HashMap<TreeStateId, Vec<TreeStateId>>,
) {
    let mut inflate: HashMap<TreeStateId, Vec<TreeStateId>> = HashMap::new();
    let mut deflate: HashMap<TreeStateId, Vec<TreeStateId>> = HashMap::new();

    states_small.iter().for_each(|(small_id, small_conn)| {
        states_large.iter().for_each(|(large_id, large_conn)| {
            if small_conn == large_conn {
                inflate
                    .entry(*small_id)
                    .and_modify(|ids| ids.push(*large_id))
                    .or_insert(vec![*large_id]);
                deflate
                    .entry(*large_id)
                    .and_modify(|ids| ids.push(*small_id))
                    .or_insert(vec![*small_id]);
            }
        })
    });

    (inflate, deflate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_transition() {
        let id_small = TreeStateId {
            triangle: 0,
            edge: 6,
            n_triangles: 1,
            n_edges: 3,
        };
        let id_large = TreeStateId {
            triangle: 17,
            edge: 14,
            n_triangles: 5,
            n_edges: 6,
        };

        let state_small: TreeState = id_small.into();
        let state_large: TreeState = id_large.into();

        let conn_small = state_small.connectivity(StateType::Shard0, Ensemble::TripleTrees);
        let conn_large = state_large.connectivity(StateType::Shard2, Ensemble::TripleTrees);

        assert!(conn_small.is_some());
        assert!(conn_large.is_some());
        assert_eq!(conn_small, conn_large);
    }
}
