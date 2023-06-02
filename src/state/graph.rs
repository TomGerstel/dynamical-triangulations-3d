use std::fmt;

#[derive(Clone, Debug)]
pub struct Graph {
    adj: Vec<Vec<usize>>,
}

#[derive(Clone, Debug)]
pub struct GraphEdge(usize, usize);

impl Graph {
    pub fn new(adj: Vec<Vec<usize>>) -> Self {
        Graph { adj }
    }

    pub fn connectivity(
        &self,
        boundary: Vec<usize>,
        edge_count: usize,
    ) -> Option<Vec<Option<usize>>> {
        let mut comps = self.connected_components();
        let no_islands = comps
            .iter()
            .all(|component| component.iter().any(|&node| boundary.contains(&node)));
        let no_loops = { edge_count + comps.len() == self.adj.len() };
        if no_islands && no_loops {
            Some(boundary_connectivity(&mut comps, boundary))
        } else {
            None
        }
    }

    pub fn shells(&self, seed: usize) -> Vec<usize> {
        let mut queue = Vec::with_capacity(self.adj.len());
        queue.push(seed);

        let mut distance = vec![None; self.adj.len()];
        distance[seed] = Some(0);

        let mut shells = vec![0; self.adj.len()];
        shells[0] = 1;

        while let Some(node) = queue.pop() {
            let dist = distance[node].unwrap() + 1;
            for nbr in &self.adj[node] {
                if distance[*nbr].is_none() {
                    shells[dist] += 1;
                    distance[*nbr] = Some(dist);
                    queue.push(*nbr);
                }
            }
        }

        let first_zero = shells.iter().position(|&x| x == 0);
        if let Some(index) = first_zero {
            shells.truncate(index)
        }

        shells
    }

    pub fn edges(&self) -> Vec<GraphEdge> {
        let mut edges = vec![];
        for i in 0..(self.adj.len()) {
            for node in &self.adj[i] {
                if i <= *node {
                    edges.push(GraphEdge(i, *node));
                }
            }
        }
        edges
    }

    fn connected_components(&self) -> Vec<Vec<usize>> {
        let size = self.adj.len();
        let mut comps = (0..size)
            .map(|node| vec![node])
            .collect::<Vec<Vec<usize>>>();

        self.adj.iter().enumerate().for_each(|(i, nbrs)| {
            nbrs.iter().for_each(|&j| {
                let (a, _) = comps
                    .iter()
                    .enumerate()
                    .find(|(_, component)| component.contains(&i))
                    .unwrap();
                let (b, _) = comps
                    .iter()
                    .enumerate()
                    .find(|(_, component)| component.contains(&j))
                    .unwrap();
                let a_new = a.min(b);
                let b_new = a.max(b);
                if a_new != b_new {
                    let mut part = comps.swap_remove(b_new);
                    comps[a_new].append(&mut part);
                }
            })
        });
        comps
    }
}

fn boundary_connectivity(comps: &mut [Vec<usize>], boundary: Vec<usize>) -> Vec<Option<usize>> {
    // sort components
    for comp in comps.iter_mut() {
        comp.sort();
    }
    comps.sort();

    let mut boundary_components = vec![None; boundary.len()];
    let mut i = 0;

    comps.iter().for_each(|component| {
        if component
            .iter()
            .filter(|&&node| boundary.contains(&node))
            .count()
            > 1
        {
            for boundary_node in &boundary {
                if component.contains(boundary_node) {
                    let boundary_index = boundary
                        .iter()
                        .find(|&&node| node == *boundary_node)
                        .unwrap();
                    boundary_components[*boundary_index] = Some(i);
                }
            }
            i += 1;
        }
    });

    boundary_components
}

impl PartialEq for GraphEdge {
    fn eq(&self, other: &Self) -> bool {
        (self.0 == other.0 && self.1 == other.1) || (self.0 == other.1 && self.1 == other.0)
    }
}

impl fmt::Display for GraphEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}, {}", self.0, self.1)
    }
}
