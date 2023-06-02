mod data_structs;

pub use data_structs::Label;

use crate::{state::graph::Graph, state::Outcome, Ensemble};

use data_structs::{Bag, LinkCutTree, Node, Pool};
use serde_derive::{Deserialize, Serialize};
use std::{
    fmt,
    ops::{Index, IndexMut},
};

/// Structure containing the current state of the geometry.
#[derive(Debug, Clone)]
pub struct Triangulation {
    vertices: Pool<Vertex>,
    edges: Pool<Edge>,
    triangles: Pool<Triangle>,
    tets: Pool<Tetrahedron>,
    forest: Pool<Node<Simplex>>,
    tet_bag: Bag<Tetrahedron>,
    middle_edges: Bag<Edge>,
    middle_triangles: Bag<Triangle>,
}

/// Vertex structure.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vertex {
    degree: usize,
    mouse: Mouse,
    node: Label<Node<Simplex>>,
}

/// Edge structure.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Edge {
    degree: usize,
    middle_degree: usize,
    mouse: Mouse,
    node: Label<Node<Simplex>>,
}

/// Triangle structure.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Triangle {
    middle_degree: usize,
    mouse: Mouse,
    node: Label<Node<Simplex>>,
}

/// Tetrahedron structure.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Tetrahedron {
    adj: [Mouse; 12],
    vertices: [Label<Vertex>; 4],
    edges: [Label<Edge>; 6],
    triangles: [Label<Triangle>; 4],
    node: Label<Node<Simplex>>,
}

/// Structure that can be used to walk around the geometry.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Mouse {
    tet: Label<Tetrahedron>,
    half_edge: HalfEdge,
}

/// Specifies possible simplex types.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Simplex {
    Vertex(Label<Vertex>),
    Edge(Label<Edge>),
    Triangle(Label<Triangle>),
    Tetrahedron(Label<Tetrahedron>),
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum HalfEdge {
    Zero = 0,
    One = 1,
    Two = 2,
    Three = 3,
    Four = 4,
    Five = 5,
    Six = 6,
    Seven = 7,
    Eight = 8,
    Nine = 9,
    Ten = 10,
    Eleven = 11,
}

fn histogram(data: Vec<usize>) -> Vec<usize> {
    let max = data.iter().max().unwrap();
    data.iter().fold(vec![0; *max + 1], |mut hist, i| {
        hist[*i] += 1;
        hist
    })
}

fn freqdist<T: PartialEq>(data: Vec<T>) -> Vec<usize> {
    let mut max = 1;
    let mut freqdist = data
        .iter()
        .fold(vec![0; data.len()], |mut freqdist, item| {
            let count = data.iter().filter(|&other| item == other).count();
            max = max.max(count);
            freqdist[count - 1] += 1;
            freqdist
        })
        .iter()
        .enumerate()
        .map(|(i, &freq)| freq / (i + 1))
        .collect::<Vec<_>>();
    freqdist.truncate(max);
    freqdist
}

impl Triangulation {
    pub fn new(volume: usize, ensemble: Ensemble) -> Self {
        // initialise triangulation struct and fields
        let vertices = Pool::<Vertex>::with_capacity(2 * volume + 1);
        let edges = Pool::<Edge>::with_capacity(4 * volume + 1);
        let triangles = Pool::<Triangle>::with_capacity(8 * volume + 1);
        let tets = Pool::<Tetrahedron>::with_capacity(2 * volume + 1);
        let forest = Pool::<Node<Simplex>>::with_capacity(16 * volume + 4);
        let tet_bag = Bag::with_capacity(2 * volume + 1);
        let middle_edges = Bag::with_capacity(4 * volume + 1);
        let middle_triangles = Bag::with_capacity(8 * volume + 1);

        // find the label of the to be inserted tetrahedron
        let tet = tets.next_label().unwrap();

        let mut triangulation = Triangulation {
            vertices,
            edges,
            triangles,
            tets,
            forest,
            tet_bag,
            middle_edges,
            middle_triangles,
        };

        // insert 4 vertices
        let v0 = triangulation.insert_vertex(
            Mouse {
                tet,
                half_edge: HalfEdge::Zero,
            },
            3,
        );
        let v1 = triangulation.insert_vertex(
            Mouse {
                tet,
                half_edge: HalfEdge::One,
            },
            3,
        );
        let v2 = triangulation.insert_vertex(
            Mouse {
                tet,
                half_edge: HalfEdge::Two,
            },
            3,
        );
        let v3 = triangulation.insert_vertex(
            Mouse {
                tet,
                half_edge: HalfEdge::Five,
            },
            3,
        );

        // insert 6 edges
        let e0 = triangulation.insert_edge(
            Mouse {
                tet,
                half_edge: HalfEdge::Zero,
            },
            2,
        );
        let e1 = triangulation.insert_edge(
            Mouse {
                tet,
                half_edge: HalfEdge::One,
            },
            2,
        );
        let e2 = triangulation.insert_edge(
            Mouse {
                tet,
                half_edge: HalfEdge::Two,
            },
            2,
        );
        let e3 = triangulation.insert_edge(
            Mouse {
                tet,
                half_edge: HalfEdge::Four,
            },
            2,
        );
        let e4 = triangulation.insert_edge(
            Mouse {
                tet,
                half_edge: HalfEdge::Five,
            },
            2,
        );
        let e5 = triangulation.insert_edge(
            Mouse {
                tet,
                half_edge: HalfEdge::Six,
            },
            2,
        );

        // insert 4 triangles
        let t0 = triangulation.insert_triangle(Mouse {
            tet,
            half_edge: HalfEdge::Zero,
        });
        let t1 = triangulation.insert_triangle(Mouse {
            tet,
            half_edge: HalfEdge::Three,
        });
        let t2 = triangulation.insert_triangle(Mouse {
            tet,
            half_edge: HalfEdge::Six,
        });
        let t3 = triangulation.insert_triangle(Mouse {
            tet,
            half_edge: HalfEdge::Nine,
        });

        // insert 2 tetrahedra
        let tet0 =
            triangulation.insert_tet([v0, v1, v2, v3], [e0, e1, e2, e3, e4, e5], [t0, t1, t2, t3]);
        let tet1 =
            triangulation.insert_tet([v1, v0, v2, v3], [e0, e2, e1, e4, e3, e5], [t0, t1, t3, t2]);
        triangulation.tet_bag.insert(tet0);
        triangulation.tet_bag.insert(tet1);

        // glue the tetrahedra together
        triangulation.tets.glue_triangles(
            Mouse {
                tet: tet0,
                half_edge: HalfEdge::Zero,
            },
            Mouse {
                tet: tet1,
                half_edge: HalfEdge::Zero,
            },
        );
        triangulation.tets.glue_triangles(
            Mouse {
                tet: tet0,
                half_edge: HalfEdge::Three,
            },
            Mouse {
                tet: tet1,
                half_edge: HalfEdge::Three,
            },
        );
        triangulation.tets.glue_triangles(
            Mouse {
                tet: tet0,
                half_edge: HalfEdge::Seven,
            },
            Mouse {
                tet: tet1,
                half_edge: HalfEdge::Eleven,
            },
        );
        triangulation.tets.glue_triangles(
            Mouse {
                tet: tet0,
                half_edge: HalfEdge::Eleven,
            },
            Mouse {
                tet: tet1,
                half_edge: HalfEdge::Seven,
            },
        );

        // link the trees
        match ensemble {
            Ensemble::Undecorated => (),
            _ => {
                triangulation.link_vertex_tree(v0, v1, e0);
                triangulation.link_vertex_tree(v0, v2, e2);
                triangulation.link_vertex_tree(v0, v3, e3);
                triangulation.link_dual_tree(tet0, tet1, t0);
            }
        }

        if matches!(ensemble, Ensemble::TripleTrees) {
            triangulation.link_middle_tree(t1, e4);
            triangulation.link_middle_tree(t2, e1);
            triangulation.link_middle_tree(t2, e4);
            triangulation.link_middle_tree(t2, e5);
            triangulation.link_middle_tree(t3, e5);
        }

        triangulation
    }

    pub fn volume(&self) -> usize {
        self.tets.size()
    }

    fn vertex_count(&self) -> usize {
        self.vertices.size()
    }

    fn vertex_degree(&self) -> Vec<usize> {
        let degrees = self
            .vertices
            .into_iter()
            .map(|label| self.vertices[label].degree)
            .collect::<Vec<_>>();
        histogram(degrees)
    }

    fn edge_degree(&self) -> Vec<usize> {
        let degrees = self
            .edges
            .into_iter()
            .map(|label| self.edges[label].degree)
            .collect::<Vec<_>>();
        histogram(degrees)
    }

    fn edge_middle(&self) -> Vec<usize> {
        let degrees = self
            .edges
            .into_iter()
            .map(|label| self.edges[label].middle_degree)
            .collect::<Vec<_>>();
        histogram(degrees)
    }

    fn triangle_middle(&self) -> Vec<usize> {
        let degrees = self
            .triangles
            .into_iter()
            .map(|label| self.triangles[label].middle_degree)
            .collect::<Vec<_>>();

        histogram(degrees)
    }

    fn edge_detour(&mut self) -> Vec<usize> {
        let mut detours = Vec::with_capacity(self.middle_edges.size());

        for edge in self.edges.into_iter() {
            if self.edges[edge].middle_degree > 0 {
                let vertex1 = self.edges[edge].mouse.head(self.tets());
                let vertex2 = self.edges[edge].mouse.tail(self.tets());

                let n1 = self.vertices[vertex1].node;
                let n2 = self.vertices[vertex2].node;

                self.forest.expose(n1);
                self.forest.evert(n1);
                self.forest.expose(n2);
                let detour = self.forest.depth(n2) / 2 - 1;
                detours.push(detour);
            }
        }
        histogram(detours)
    }

    fn triangle_detour(&mut self) -> Vec<usize> {
        let mut detours = Vec::with_capacity(self.middle_triangles.size());

        for triangle in self.triangles.into_iter() {
            if self.triangles[triangle].middle_degree > 0 {
                let tet1 = self.triangles[triangle].mouse.tet();
                let tet2 = self.triangles[triangle].mouse.adj_ext(self.tets()).tet();

                let n1 = self.tets[tet1].node;
                let n2 = self.tets[tet2].node;

                self.forest.expose(n1);
                self.forest.evert(n1);
                self.forest.expose(n2);
                let detour = self.forest.depth(n2) / 2 - 1;
                detours.push(detour);
            }
        }
        histogram(detours)
    }

    fn dual_graph(&self) -> Graph {
        let triangles = self.triangles.into_iter().collect::<Vec<_>>();

        const NEW_VEC_USIZE: Vec<usize> = Vec::new();
        let mut adj = vec![NEW_VEC_USIZE; triangles.len()];

        for triangle in &triangles {
            let mouse = self.triangles[*triangle].mouse;
            let tet_i = mouse.tet();
            let tet_j = mouse.adj_ext(&self.tets).tet();
            let i = self.tets.into_iter().position(|tet| tet == tet_i).unwrap();
            let j = self.tets.into_iter().position(|tet| tet == tet_j).unwrap();
            adj[i].push(j);
            adj[j].push(i);
        }

        Graph::new(adj)
    }

    fn dual_tree(&self) -> Graph {
        let triangles = self.triangles.into_iter().collect::<Vec<_>>();

        const NEW_VEC_USIZE: Vec<usize> = Vec::new();
        let mut adj = vec![NEW_VEC_USIZE; triangles.len()];

        for triangle in &triangles {
            if !self.is_middle_triangle(*triangle) {
                let mouse = self.triangles[*triangle].mouse;
                let tet_i = mouse.tet();
                let tet_j = mouse.adj_ext(&self.tets).tet();
                let i = self.tets.into_iter().position(|tet| tet == tet_i).unwrap();
                let j = self.tets.into_iter().position(|tet| tet == tet_j).unwrap();
                adj[i].push(j);
                adj[j].push(i);
            }
        }

        Graph::new(adj)
    }

    fn vertex_graph(&self) -> Graph {
        let edges = self.edges.into_iter().collect::<Vec<_>>();

        const NEW_VEC_USIZE: Vec<usize> = Vec::new();
        let mut adj = vec![NEW_VEC_USIZE; edges.len()];

        for edge in &edges {
            let mouse = self.edges[*edge].mouse;
            let vert_i = mouse.tail(&self.tets);
            let vert_j = mouse.head(&self.tets);
            let i = self.vertices.into_iter().position(|v| v == vert_i).unwrap();
            let j = self.vertices.into_iter().position(|v| v == vert_j).unwrap();
            adj[i].push(j);
            adj[j].push(i);
        }

        Graph::new(adj)
    }

    fn vertex_tree(&self) -> Graph {
        let edges = self.edges.into_iter().collect::<Vec<_>>();

        const NEW_VEC_USIZE: Vec<usize> = Vec::new();
        let mut adj = vec![NEW_VEC_USIZE; edges.len()];

        for edge in &edges {
            if !self.is_middle_edge(*edge) {
                let mouse = self.edges[*edge].mouse;
                let vert_i = mouse.tail(&self.tets);
                let vert_j = mouse.head(&self.tets);
                let i = self.vertices.into_iter().position(|v| v == vert_i).unwrap();
                let j = self.vertices.into_iter().position(|v| v == vert_j).unwrap();
                adj[i].push(j);
                adj[j].push(i);
            }
        }

        Graph::new(adj)
    }

    pub fn sample_mouse(&self) -> Mouse {
        let tet = self.tet_bag.sample();
        let half_edge: HalfEdge = HalfEdge::sample();
        Mouse::new(tet, half_edge)
    }

    pub fn sample_middle_edge(&self) -> Label<Edge> {
        self.middle_edges.sample()
    }

    pub fn sample_middle_triangle(&self) -> Label<Triangle> {
        self.middle_triangles.sample()
    }

    pub fn insert_vertex(&mut self, mouse: Mouse, degree: usize) -> Label<Vertex> {
        let vertex_label = self.vertices.next_label().unwrap();
        let node = Node::new(Simplex::Vertex(vertex_label));
        let node_label = self.forest.insert(node);
        self.vertices.insert(Vertex {
            degree,
            mouse,
            node: node_label,
        })
    }

    pub fn remove_vertex(&mut self, label: Label<Vertex>) {
        let node = self.vertices[label].node;
        self.forest.remove(node);
        self.vertices.remove(label);
    }

    pub fn insert_edge(&mut self, mouse: Mouse, degree: usize) -> Label<Edge> {
        let edge_label = self.edges.next_label().unwrap();
        let node = Node::new(Simplex::Edge(edge_label));
        let node_label = self.forest.insert(node);
        self.middle_edges.insert(edge_label);
        self.edges.insert(Edge {
            degree,
            middle_degree: 0,
            mouse,
            node: node_label,
        })
    }

    pub fn remove_edge(&mut self, label: Label<Edge>) {
        let node = self.edges[label].node;
        self.forest.remove(node);
        self.middle_edges.remove(label);
        self.edges.remove(label);
    }

    pub fn insert_triangle(&mut self, mouse: Mouse) -> Label<Triangle> {
        let triangle_label = self.triangles.next_label().unwrap();
        let node = Node::new(Simplex::Triangle(triangle_label));
        let node_label = self.forest.insert(node);
        self.middle_triangles.insert(triangle_label);
        self.triangles.insert(Triangle {
            middle_degree: 0,
            mouse,
            node: node_label,
        })
    }

    pub fn remove_triangle(&mut self, label: Label<Triangle>) {
        let node = self.triangles[label].node;
        self.forest.remove(node);
        self.middle_triangles.remove(label);
        self.triangles.remove(label);
    }

    pub fn insert_tet(
        &mut self,
        vertices: [Label<Vertex>; 4],
        edges: [Label<Edge>; 6],
        triangles: [Label<Triangle>; 4],
    ) -> Label<Tetrahedron> {
        let tet_label = self.tets.next_label().unwrap();
        let node = Node::new(Simplex::Tetrahedron(tet_label));
        let node_label = self.forest.insert(node);
        let adj = [
            HalfEdge::Three,
            HalfEdge::Five,
            HalfEdge::Four,
            HalfEdge::Zero,
            HalfEdge::Two,
            HalfEdge::One,
            HalfEdge::Nine,
            HalfEdge::Eleven,
            HalfEdge::Ten,
            HalfEdge::Six,
            HalfEdge::Eight,
            HalfEdge::Seven,
        ]
        .map(|half_edge| Mouse {
            tet: tet_label,
            half_edge,
        });
        self.tets.insert(Tetrahedron {
            adj,
            vertices,
            edges,
            triangles,
            node: node_label,
        })
    }

    pub fn remove_tet(&mut self, label: Label<Tetrahedron>) {
        let node = self.tets[label].node;
        self.forest.remove(node);
        self.tets.remove(label);
    }

    pub fn insert_tet_bag(&mut self, label: Label<Tetrahedron>) {
        self.tet_bag.insert(label);
    }

    pub fn remove_tet_bag(&mut self, label: Label<Tetrahedron>) {
        self.tet_bag.remove(label);
    }

    pub fn increment_vertex_degree(&mut self, label: Label<Vertex>) {
        self.vertices[label].degree += 1;
    }

    pub fn decrement_vertex_degree(&mut self, label: Label<Vertex>) {
        self.vertices[label].degree -= 1;
    }

    pub fn increment_edge_degree(&mut self, label: Label<Edge>) {
        self.edges[label].degree += 1;
    }

    pub fn decrement_edge_degree(&mut self, label: Label<Edge>) {
        self.edges[label].degree -= 1;
    }

    pub fn set_mouse_vertex(&mut self, label: Label<Vertex>, mouse: Mouse) {
        self.vertices[label].mouse = mouse;
    }

    pub fn set_mouse_edge(&mut self, label: Label<Edge>, mouse: Mouse) {
        self.edges[label].mouse = mouse;
    }

    pub fn set_mouse_triangle(&mut self, label: Label<Triangle>, mouse: Mouse) {
        self.triangles[label].mouse = mouse;
    }

    pub fn vertex_tree_move(
        &mut self,
        edge: Label<Edge>,
        seed: f32,
        ensemble: Ensemble,
    ) -> Outcome {
        if matches!(ensemble, Ensemble::Undecorated) {
            return Outcome::ValidVertexTree;
        }
        if self.edges[edge].middle_degree != 1 {
            return Outcome::LeafVertex;
        }
        let vertex1 = self.edges[edge].mouse.head(self.tets());
        let vertex2 = self.edges[edge].mouse.tail(self.tets());

        let n1 = self.vertices[vertex1].node;
        let n2 = self.vertices[vertex2].node;
        let e = self.edges[edge].node;

        // setup
        self.forest.expose(n1);
        self.forest.evert(n1);
        self.forest.expose(n2);
        let i_max = self.forest.depth(n2) / 2;
        let rng = (seed * i_max as f32).floor() as usize;
        let i = 2 * (rng + 1);
        let n_cut = self.forest.index_depth(n2, i);
        let e_cut = self.forest.index_depth(n2, i - 1);
        let Simplex::Edge(edge_cut) = *self.forest.value(e_cut) else {
            unreachable!();
        };
        let old_triangle = self.leaf_triangle(edge).unwrap();
        let Some(new_triangle) = self.leaf_triangle(edge_cut) else {
            return Outcome::LeafVertex;
        };

        // perform the actual move
        if matches!(ensemble, Ensemble::TripleTrees) {
            self.cut_middle_tree(old_triangle, edge);
            self.middle_edges.remove(edge);
        }

        // cut
        self.forest.expose(n_cut);
        self.forest.cut(n_cut);
        self.forest.splay(e_cut);
        self.forest.cut(e_cut);

        // evert
        self.forest.expose(n2);
        self.forest.evert(n2);
        self.forest.expose(n1);

        // link
        self.forest.link(n1, e);
        self.forest.expose(n2);
        self.forest.link(e, n2);

        if matches!(ensemble, Ensemble::TripleTrees) {
            self.middle_edges.insert(edge_cut);
            self.link_middle_tree(new_triangle, edge_cut);
        }

        Outcome::ValidVertexTree
    }

    pub fn dual_tree_move(
        &mut self,
        triangle: Label<Triangle>,
        seed: f32,
        ensemble: Ensemble,
    ) -> Outcome {
        if matches!(ensemble, Ensemble::Undecorated) {
            return Outcome::ValidDualTree;
        }
        if self.triangles[triangle].middle_degree != 1 {
            return Outcome::LeafDual;
        }
        let tet1 = self.triangles[triangle].mouse.tet();
        let tet2 = self.triangles[triangle].mouse.adj_ext(self.tets()).tet();
        let n1 = self.tets[tet1].node;
        let n2 = self.tets[tet2].node;
        let e = self.triangles[triangle].node;

        // setup
        self.forest.expose(n1);
        self.forest.evert(n1);
        self.forest.expose(n2);
        let i_max = self.forest.depth(n2) / 2;
        let rng = (seed * i_max as f32).floor() as usize;
        let i = 2 * (rng + 1);
        let n_cut = self.forest.index_depth(n2, i);
        let e_cut = self.forest.index_depth(n2, i - 1);
        let Simplex::Triangle(triangle_cut) = *self.forest.value(e_cut) else {
                unreachable!();
            };
        let old_edge = self.leaf_edge(triangle).unwrap();
        let Some(new_edge) = self.leaf_edge(triangle_cut) else {
                return Outcome::LeafDual;
            };

        // perform the actual move
        if matches!(ensemble, Ensemble::TripleTrees) {
            self.cut_middle_tree(triangle, old_edge);
            self.middle_triangles.remove(triangle);
        }

        // cut
        self.forest.expose(n_cut);
        self.forest.cut(n_cut);
        self.forest.splay(e_cut);
        self.forest.cut(e_cut);

        // evert
        self.forest.expose(n2);
        self.forest.evert(n2);
        self.forest.expose(n2);

        // link
        self.forest.expose(n1);
        self.forest.link(n1, e);
        self.forest.link(e, n2);

        if matches!(ensemble, Ensemble::TripleTrees) {
            self.middle_triangles.insert(triangle_cut);
            self.link_middle_tree(triangle_cut, new_edge);
        }

        Outcome::ValidDualTree
    }

    fn leaf_edge(&self, triangle: Label<Triangle>) -> Option<Label<Edge>> {
        let mouse = self.triangles[triangle].mouse;
        let mice = [mouse, mouse.next(), mouse.next().next()];
        let edges = mice.map(|m| m.edge(self.tets()));
        let mut leaf_edge = None;
        let mut middle_edges = 0;
        for edge in edges {
            if self.is_middle_edge(edge) {
                leaf_edge = Some(edge);
                middle_edges += 1;
                if middle_edges > 1 {
                    return None;
                }
            }
        }
        leaf_edge
    }

    fn leaf_triangle(&self, edge: Label<Edge>) -> Option<Label<Triangle>> {
        let mouse = self.edges[edge].mouse;
        let mut walker = mouse;
        let mut leaf = None;
        loop {
            let triangle = walker.triangle(self.tets());
            if self.is_middle_triangle(triangle) {
                if leaf.is_none() {
                    leaf = Some(triangle);
                } else {
                    return None;
                }
            }
            walker = walker.adj_int().adj_ext(self.tets());
            if walker == mouse {
                break leaf;
            }
        }
    }

    pub fn is_middle_edge(&self, label: Label<Edge>) -> bool {
        self.middle_edges.contains(label)
    }

    pub fn is_middle_triangle(&self, label: Label<Triangle>) -> bool {
        self.middle_triangles.contains(label)
    }

    pub fn link_vertex_tree(
        &mut self,
        node_a: Label<Vertex>,
        node_b: Label<Vertex>,
        edge: Label<Edge>,
    ) {
        self.middle_edges.remove(edge);

        let a = self.vertices[node_a].node;
        let b = self.vertices[node_b].node;
        let e = self.edges[edge].node;

        // evert
        self.forest.expose(b);
        self.forest.evert(b);
        self.forest.expose(b);

        // link
        self.forest.expose(a);
        self.forest.link(a, e);
        self.forest.link(e, b);
    }

    pub fn link_dual_tree(
        &mut self,
        node_a: Label<Tetrahedron>,
        node_b: Label<Tetrahedron>,
        edge: Label<Triangle>,
    ) {
        self.middle_triangles.remove(edge);

        let a = self.tets[node_a].node;
        let b = self.tets[node_b].node;
        let e = self.triangles[edge].node;

        // evert
        self.forest.expose(b);
        self.forest.evert(b);
        self.forest.expose(b);

        // link
        self.forest.expose(a);
        self.forest.link(a, e);
        self.forest.link(e, b);
    }

    pub fn link_middle_tree(&mut self, triangle: Label<Triangle>, edge: Label<Edge>) {
        let t = self.triangles[triangle].node;
        let e = self.edges[edge].node;

        self.triangles[triangle].middle_degree += 1;
        self.edges[edge].middle_degree += 1;

        // evert
        self.forest.expose(t);
        self.forest.evert(t);
        self.forest.expose(t);

        // link
        self.forest.expose(e);
        self.forest.link(e, t);
    }

    pub fn cut_vertex_tree(
        &mut self,
        node_a: Label<Vertex>,
        node_b: Label<Vertex>,
        edge: Label<Edge>,
    ) {
        let a = self.vertices[node_a].node;
        let b = self.vertices[node_b].node;
        let e = self.edges[edge].node;

        // evert
        self.forest.expose(a);
        self.forest.evert(a);

        // cut
        self.forest.expose(b);
        self.forest.cut(b);
        self.forest.splay(e);
        self.forest.cut(e);

        self.middle_edges.insert(edge);
    }

    pub fn cut_dual_tree(
        &mut self,
        node_a: Label<Tetrahedron>,
        node_b: Label<Tetrahedron>,
        edge: Label<Triangle>,
    ) {
        let a = self.tets[node_a].node;
        let b = self.tets[node_b].node;
        let e = self.triangles[edge].node;

        // evert
        self.forest.expose(a);
        self.forest.evert(a);

        // cut
        self.forest.expose(b);
        self.forest.cut(b);
        self.forest.splay(e);
        self.forest.cut(e);

        self.middle_triangles.insert(edge);
    }

    pub fn cut_middle_tree(&mut self, triangle: Label<Triangle>, edge: Label<Edge>) {
        let t = self.triangles[triangle].node;
        let e = self.edges[edge].node;

        self.triangles[triangle].middle_degree -= 1;
        self.edges[edge].middle_degree -= 1;

        // evert
        self.forest.expose(t);
        self.forest.evert(t);

        // cut
        self.forest.expose(e);
        self.forest.cut(e);
    }

    pub fn tets(&self) -> &Pool<Tetrahedron> {
        &self.tets
    }

    pub fn tets_mut(&mut self) -> &mut Pool<Tetrahedron> {
        &mut self.tets
    }

    #[cfg(test)]
    pub fn sanity_check(&mut self) {
        // check the euler characteristic
        // N0 - N1 + N2 - N3 = 0, where N2 = 2 * N3
        assert_eq!(self.vertex_count() + self.volume(), self.edge_count());

        // check trees
        assert_eq!(
            self.vertices.size(),
            self.edges.size() - self.middle_edges.size() + 1
        );
        assert_eq!(
            self.tets.size(),
            self.triangles.size() - self.middle_triangles.size() + 1
        );

        // check vertices
        for vertex in &self.vertices {
            // check if vertex is properly in forest
            let node = self.vertices[vertex].node;
            assert!(self.forest.contains(node));
            assert_eq!(self.forest.value(node), &Simplex::Vertex(vertex));

            // check if mouse is correct
            assert_eq!(self.vertices[vertex].mouse.tail(self.tets()), vertex);
        }

        // check edges
        for edge in &self.edges {
            // check if edge is properly in forest
            let node = self.edges[edge].node;
            assert!(self.forest.contains(node));
            assert_eq!(self.forest.value(node), &Simplex::Edge(edge));

            // check if mouse is correct
            assert_eq!(self.edges[edge].mouse.edge(self.tets()), edge);

            // check if vertices are distinct
            let vertex1 = self.edges[edge].mouse.head(self.tets());
            let vertex2 = self.edges[edge].mouse.tail(self.tets());
            assert_ne!(vertex1, vertex2);
        }

        // check triangles
        for triangle in &self.triangles {
            // check if triangle is properly in forest
            let node = self.triangles[triangle].node;
            assert!(self.forest.contains(node));
            assert_eq!(self.forest.value(node), &Simplex::Triangle(triangle));

            // check if mouse is correct
            assert_eq!(
                self.triangles[triangle].mouse.triangle(self.tets()),
                triangle
            );
            assert_eq!(
                self.triangles[triangle]
                    .mouse
                    .adj_ext(self.tets())
                    .triangle(self.tets()),
                triangle
            );

            // check middle tree
            if self.is_middle_triangle(triangle) {
                let t = self.triangles[triangle].node;
                self.forest.evert(t);
                let mouse = self.triangles[triangle].mouse;
                let edges = [
                    mouse.edge(self.tets()),
                    mouse.next().edge(self.tets()),
                    mouse.next().next().edge(self.tets()),
                ];
                for edge in edges {
                    if self.is_middle_edge(edge) {
                        let e = self.edges[edge].node;
                        let depth = self.forest.depth(e);
                        assert_eq!(
                            depth, 1,
                            "Edge {edge} should have a direct connection to triangle {triangle}."
                        );
                    }
                }
            }
        }

        // check tetrahedra
        for tet in &self.tets {
            // check if vertices are defined
            let vertices = self.tets[tet].vertices;
            for vertex in vertices {
                assert!(self.vertices.contains(vertex));
            }

            // check if edges are defined
            let edges = self.tets[tet].edges;
            for edge in edges {
                assert!(self.edges.contains(edge));
            }

            // check if triangles are defined
            let triangles = self.tets[tet].triangles;
            for triangle in triangles {
                assert!(self.triangles.contains(triangle));
            }

            // check if tetrahedron is properly in forest
            let node = self.tets[tet].node;
            assert!(self.forest.contains(node));
            assert_eq!(self.forest.value(node), &Simplex::Tetrahedron(tet));

            // check mice
            for mouse in self.tets[tet].adj {
                // check adjacencies
                assert_ne!(mouse.next(), mouse);
                assert_ne!(mouse.adj_int(), mouse);
                assert_ne!(mouse.adj_ext(self.tets()), mouse);
                assert_eq!(mouse.next().next().next(), mouse);
                assert_eq!(mouse.adj_int().adj_int(), mouse);
                assert_eq!(mouse.adj_ext(self.tets()).adj_ext(self.tets()), mouse);
                assert_eq!(
                    mouse.next().adj_ext(self.tets()).next(),
                    mouse.adj_ext(self.tets())
                );

                // check vertices
                assert_eq!(mouse.next().tail(self.tets()), mouse.head(self.tets()));
                assert_eq!(
                    mouse.adj_int().tail(self.tets()),
                    mouse.head(self.tets()),
                    "head of ({}) should be tail of adj_int ({})",
                    mouse,
                    mouse.adj_int()
                );
                assert_eq!(
                    mouse.adj_int().head(self.tets()),
                    mouse.tail(self.tets()),
                    "tail of ({}) should be head of adj_int ({})",
                    mouse,
                    mouse.adj_int()
                );
                assert_eq!(
                    mouse.adj_ext(self.tets()).tail(self.tets()),
                    mouse.head(self.tets()),
                    "head of ({}) should be tail of adj_ext ({})",
                    mouse,
                    mouse.adj_ext(self.tets())
                );
                assert_eq!(
                    mouse.adj_ext(self.tets()).head(self.tets()),
                    mouse.tail(self.tets()),
                    "tail of ({}) should be head of adj_ext ({})",
                    mouse,
                    mouse.adj_ext(self.tets())
                );

                // check subsimplices
                assert_eq!(mouse.adj_int().edge(self.tets()), mouse.edge(self.tets()));
                assert_eq!(
                    mouse.adj_ext(self.tets()).edge(self.tets()),
                    mouse.edge(self.tets())
                );
                let tail = mouse.tail(self.tets());
                let head = mouse.head(self.tets());

                assert!(self.vertices.contains(tail));
                assert!(self.vertices.contains(head));
            }
        }

        // check forest
        //self.forest.sanity_check();
        for node in self.forest.clone().into_iter() {
            match *self.forest.value(node) {
                Simplex::Edge(label) => {
                    if !self.is_middle_edge(label) {
                        let vertex1 = self.edges[label].mouse.head(self.tets());
                        let vertex2 = self.edges[label].mouse.tail(self.tets());
                        assert_ne!(vertex1, vertex2);
                        let v1 = self.vertices[vertex1].node;
                        let v2 = self.vertices[vertex2].node;
                        assert_ne!(v1, v2);
                        self.forest.evert(v1);
                        assert_eq!(self.forest.find_root(node), v1);
                        assert_eq!(self.forest.find_root(v2), v1);
                        let depth_e = self.forest.depth(node);
                        let depth_v1 = self.forest.depth(v1);
                        let depth_v2 = self.forest.depth(v2);
                        assert_eq!(depth_v1, 0);
                        assert_eq!(depth_e, 1);
                        assert_eq!(depth_v2, 2, "Vertex {vertex2} should have depth 2");
                    } else {
                        self.forest.evert(node);
                        let origin = self.edges[label].mouse;
                        let mut middle = self
                            .forest
                            .find_root(self.triangles[origin.triangle(self.tets())].node)
                            == node;
                        let mut walker = origin.adj_int().adj_ext(self.tets());
                        while walker != origin {
                            middle |= self
                                .forest
                                .find_root(self.triangles[walker.triangle(self.tets())].node)
                                == node;
                            walker = walker.adj_int().adj_ext(self.tets());
                        }
                        assert!(middle);
                    }
                }
                Simplex::Vertex(_) => assert_eq!(self.forest.depth(node) % 2, 0),
                Simplex::Tetrahedron(_) => assert_eq!(
                    self.forest.depth(node) % 2,
                    0,
                    "Depth is {}, should be even",
                    self.forest.depth(node)
                ),
                Simplex::Triangle(label) => {
                    if !self.is_middle_triangle(label) {
                        let tet1 = self.triangles[label].mouse.tet();
                        let tet2 = self.triangles[label].mouse.adj_ext(self.tets()).tet();
                        assert_ne!(tet1, tet2);
                        let t1 = self.tets[tet1].node;
                        let t2 = self.tets[tet2].node;
                        assert_ne!(t1, t2);
                        self.forest.evert(t1);
                        assert_eq!(self.forest.find_root(node), t1);
                        assert_eq!(self.forest.find_root(t2), t1);
                        let depth_t1 = self.forest.depth(t1);
                        let depth_e = self.forest.depth(node);
                        let depth_t2 = self.forest.depth(t2);
                        assert_eq!(depth_t1, 0);
                        assert_eq!(
                            depth_e, 1,
                            "Edge {tet1}-[{label}]-{tet2} should have depth 1"
                        );
                        assert_eq!(
                            depth_t2, 2,
                            "Tetrahedron {tet1}-{label}-[{tet2}] should have depth 2"
                        );
                    } else {
                        self.forest.evert(node);
                        let mouse = self.triangles[label].mouse;
                        let edge_nodes = [
                            self.edges[mouse.edge(self.tets())].node,
                            self.edges[mouse.next().edge(self.tets())].node,
                            self.edges[mouse.next().next().edge(self.tets())].node,
                        ];
                        assert!(
                            self.forest.find_root(edge_nodes[0]) == node
                                || self.forest.find_root(edge_nodes[1]) == node
                                || self.forest.find_root(edge_nodes[2]) == node
                        );
                    }
                }
            }
        }
    }

    #[cfg(test)]
    fn edge_count(&self) -> usize {
        self.edges.size()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum GraphType {
    DualGraph,
    VertexGraph,
    DualTree,
    VertexTree,
}

impl GraphType {
    pub fn name(&self) -> &str {
        match *self {
            GraphType::DualGraph => "dual_graph",
            GraphType::VertexGraph => "vertex_graph",
            GraphType::DualTree => "dual_tree",
            GraphType::VertexTree => "vertex_tree",
        }
    }

    pub fn generate(&self, triangulation: &Triangulation) -> Graph {
        match *self {
            GraphType::DualGraph => triangulation.dual_graph(),
            GraphType::VertexGraph => triangulation.vertex_graph(),
            GraphType::DualTree => triangulation.dual_tree(),
            GraphType::VertexTree => triangulation.vertex_tree(),
        }
    }
}

impl Label<Tetrahedron> {
    pub fn root(self) -> Mouse {
        Mouse {
            tet: self,
            half_edge: HalfEdge::Zero,
        }
    }
}

impl Edge {
    #[cfg(test)]
    pub fn from_label(i: usize) -> Label<Edge> {
        Label::<Edge>::from_value(i)
    }
}

impl Pool<Tetrahedron> {
    pub fn glue_triangles(&mut self, a: Mouse, b: Mouse) {
        self[a] = b;
        self[a.next()] = b.next().next();
        self[a.next().next()] = b.next();
        self[b] = a;
        self[b.next()] = a.next().next();
        self[b.next().next()] = a.next();
    }

    pub fn set_vertices(&mut self, mouse: Mouse, vertices: [Label<Vertex>; 4]) {
        let [v0, v1, v2, v3] = vertices;
        self[mouse.tet].vertices = match mouse.half_edge {
            HalfEdge::Zero => [v0, v1, v2, v3],
            HalfEdge::One => [v2, v0, v1, v3],
            HalfEdge::Two => [v1, v2, v0, v3],
            HalfEdge::Three => [v1, v0, v3, v2],
            HalfEdge::Four => [v0, v2, v3, v1],
            HalfEdge::Five => [v2, v1, v3, v0],
            HalfEdge::Six => [v3, v2, v1, v0],
            HalfEdge::Seven => [v3, v1, v0, v2],
            HalfEdge::Eight => [v3, v0, v2, v1],
            HalfEdge::Nine => [v2, v3, v0, v1],
            HalfEdge::Ten => [v1, v3, v2, v0],
            HalfEdge::Eleven => [v0, v3, v1, v2],
        };
    }

    pub fn set_edges(&mut self, mouse: Mouse, edges: [Label<Edge>; 6]) {
        let [e0, e1, e2, e3, e4, e5] = edges;
        self[mouse.tet].edges = match mouse.half_edge {
            HalfEdge::Zero => [e0, e1, e2, e3, e4, e5],
            HalfEdge::One => [e2, e0, e1, e5, e3, e4],
            HalfEdge::Two => [e1, e2, e0, e4, e5, e3],
            HalfEdge::Three => [e0, e3, e4, e1, e2, e5],
            HalfEdge::Four => [e2, e5, e3, e0, e1, e4],
            HalfEdge::Five => [e1, e4, e5, e2, e0, e3],
            HalfEdge::Six => [e5, e1, e4, e3, e2, e0],
            HalfEdge::Seven => [e4, e0, e3, e5, e1, e2],
            HalfEdge::Eight => [e3, e2, e5, e4, e0, e1],
            HalfEdge::Nine => [e5, e3, e2, e1, e4, e0],
            HalfEdge::Ten => [e4, e5, e1, e0, e3, e2],
            HalfEdge::Eleven => [e3, e4, e0, e2, e5, e1],
        };
    }

    pub fn set_triangles(&mut self, mouse: Mouse, triangles: [Label<Triangle>; 4]) {
        let [t0, t1, t2, t3] = triangles;
        self[mouse.tet].triangles = match mouse.half_edge {
            HalfEdge::Zero => [t0, t1, t2, t3],
            HalfEdge::One => [t0, t3, t1, t2],
            HalfEdge::Two => [t0, t2, t3, t1],
            HalfEdge::Three => [t1, t0, t3, t2],
            HalfEdge::Four => [t3, t0, t2, t1],
            HalfEdge::Five => [t2, t0, t1, t3],
            HalfEdge::Six => [t2, t3, t0, t1],
            HalfEdge::Seven => [t1, t2, t0, t3],
            HalfEdge::Eight => [t3, t1, t0, t2],
            HalfEdge::Nine => [t3, t2, t1, t0],
            HalfEdge::Ten => [t2, t1, t3, t0],
            HalfEdge::Eleven => [t1, t3, t2, t0],
        };
    }

    pub fn set_base_triangle(&mut self, mouse: Mouse, triangle: Label<Triangle>) {
        let i = match mouse.half_edge {
            HalfEdge::Zero => 0,
            HalfEdge::One => 0,
            HalfEdge::Two => 0,
            HalfEdge::Three => 1,
            HalfEdge::Four => 1,
            HalfEdge::Five => 1,
            HalfEdge::Six => 2,
            HalfEdge::Seven => 2,
            HalfEdge::Eight => 2,
            HalfEdge::Nine => 3,
            HalfEdge::Ten => 3,
            HalfEdge::Eleven => 3,
        };
        self[mouse.tet].triangles[i] = triangle;
    }
}

impl Mouse {
    fn new(tet: Label<Tetrahedron>, half_edge: HalfEdge) -> Self {
        Mouse { tet, half_edge }
    }

    #[cfg(test)]
    pub fn from_labels(tet: usize, half_edge: usize) -> Self {
        Mouse {
            tet: data_structs::Label::from_value(tet),
            half_edge: HalfEdge::from_value(half_edge),
        }
    }

    pub fn next(&self) -> Self {
        Mouse {
            tet: self.tet,
            half_edge: match self.half_edge {
                HalfEdge::Zero => HalfEdge::One,
                HalfEdge::One => HalfEdge::Two,
                HalfEdge::Two => HalfEdge::Zero,
                HalfEdge::Three => HalfEdge::Four,
                HalfEdge::Four => HalfEdge::Five,
                HalfEdge::Five => HalfEdge::Three,
                HalfEdge::Six => HalfEdge::Seven,
                HalfEdge::Seven => HalfEdge::Eight,
                HalfEdge::Eight => HalfEdge::Six,
                HalfEdge::Nine => HalfEdge::Ten,
                HalfEdge::Ten => HalfEdge::Eleven,
                HalfEdge::Eleven => HalfEdge::Nine,
            },
        }
    }

    pub fn adj_int(&self) -> Self {
        Mouse {
            tet: self.tet,
            half_edge: match self.half_edge {
                HalfEdge::Zero => HalfEdge::Three,
                HalfEdge::One => HalfEdge::Seven,
                HalfEdge::Two => HalfEdge::Eleven,
                HalfEdge::Three => HalfEdge::Zero,
                HalfEdge::Four => HalfEdge::Ten,
                HalfEdge::Five => HalfEdge::Eight,
                HalfEdge::Six => HalfEdge::Nine,
                HalfEdge::Seven => HalfEdge::One,
                HalfEdge::Eight => HalfEdge::Five,
                HalfEdge::Nine => HalfEdge::Six,
                HalfEdge::Ten => HalfEdge::Four,
                HalfEdge::Eleven => HalfEdge::Two,
            },
        }
    }

    pub fn adj_ext(&self, tets: &Pool<Tetrahedron>) -> Self {
        tets[*self]
    }

    pub fn tail(&self, tets: &Pool<Tetrahedron>) -> Label<Vertex> {
        match self.half_edge {
            HalfEdge::Zero => tets[self.tet].vertices[0],
            HalfEdge::One => tets[self.tet].vertices[1],
            HalfEdge::Two => tets[self.tet].vertices[2],
            HalfEdge::Three => tets[self.tet].vertices[1],
            HalfEdge::Four => tets[self.tet].vertices[0],
            HalfEdge::Five => tets[self.tet].vertices[3],
            HalfEdge::Six => tets[self.tet].vertices[3],
            HalfEdge::Seven => tets[self.tet].vertices[2],
            HalfEdge::Eight => tets[self.tet].vertices[1],
            HalfEdge::Nine => tets[self.tet].vertices[2],
            HalfEdge::Ten => tets[self.tet].vertices[3],
            HalfEdge::Eleven => tets[self.tet].vertices[0],
        }
    }

    pub fn head(&self, tets: &Pool<Tetrahedron>) -> Label<Vertex> {
        match self.half_edge {
            HalfEdge::Zero => tets[self.tet].vertices[1],
            HalfEdge::One => tets[self.tet].vertices[2],
            HalfEdge::Two => tets[self.tet].vertices[0],
            HalfEdge::Three => tets[self.tet].vertices[0],
            HalfEdge::Four => tets[self.tet].vertices[3],
            HalfEdge::Five => tets[self.tet].vertices[1],
            HalfEdge::Six => tets[self.tet].vertices[2],
            HalfEdge::Seven => tets[self.tet].vertices[1],
            HalfEdge::Eight => tets[self.tet].vertices[3],
            HalfEdge::Nine => tets[self.tet].vertices[3],
            HalfEdge::Ten => tets[self.tet].vertices[0],
            HalfEdge::Eleven => tets[self.tet].vertices[2],
        }
    }

    pub fn edge(&self, tets: &Pool<Tetrahedron>) -> Label<Edge> {
        match self.half_edge {
            HalfEdge::Zero => tets[self.tet].edges[0],
            HalfEdge::One => tets[self.tet].edges[1],
            HalfEdge::Two => tets[self.tet].edges[2],
            HalfEdge::Three => tets[self.tet].edges[0],
            HalfEdge::Four => tets[self.tet].edges[3],
            HalfEdge::Five => tets[self.tet].edges[4],
            HalfEdge::Six => tets[self.tet].edges[5],
            HalfEdge::Seven => tets[self.tet].edges[1],
            HalfEdge::Eight => tets[self.tet].edges[4],
            HalfEdge::Nine => tets[self.tet].edges[5],
            HalfEdge::Ten => tets[self.tet].edges[3],
            HalfEdge::Eleven => tets[self.tet].edges[2],
        }
    }

    pub fn triangle(&self, tets: &Pool<Tetrahedron>) -> Label<Triangle> {
        match self.half_edge {
            HalfEdge::Zero => tets[self.tet].triangles[0],
            HalfEdge::One => tets[self.tet].triangles[0],
            HalfEdge::Two => tets[self.tet].triangles[0],
            HalfEdge::Three => tets[self.tet].triangles[1],
            HalfEdge::Four => tets[self.tet].triangles[1],
            HalfEdge::Five => tets[self.tet].triangles[1],
            HalfEdge::Six => tets[self.tet].triangles[2],
            HalfEdge::Seven => tets[self.tet].triangles[2],
            HalfEdge::Eight => tets[self.tet].triangles[2],
            HalfEdge::Nine => tets[self.tet].triangles[3],
            HalfEdge::Ten => tets[self.tet].triangles[3],
            HalfEdge::Eleven => tets[self.tet].triangles[3],
        }
    }

    pub fn tet(&self) -> Label<Tetrahedron> {
        self.tet
    }
}

impl HalfEdge {
    fn sample() -> Self {
        let i = fastrand::usize(0..12);
        HalfEdge::from_value(i)
    }

    fn from_value(i: usize) -> Self {
        match i {
            0 => HalfEdge::Zero,
            1 => HalfEdge::One,
            2 => HalfEdge::Two,
            3 => HalfEdge::Three,
            4 => HalfEdge::Four,
            5 => HalfEdge::Five,
            6 => HalfEdge::Six,
            7 => HalfEdge::Seven,
            8 => HalfEdge::Eight,
            9 => HalfEdge::Nine,
            10 => HalfEdge::Ten,
            11 => HalfEdge::Eleven,
            _ => unreachable!(),
        }
    }
}

/// Specifies possible observables
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub enum Observable {
    Volume,
    DualGraphShell,
    DualTreeShell,
    VertexTreeShell,
    VertexCount,
    VertexDegree,
    EdgeDegree,
    EdgeFreq,
    EdgeMiddle,
    TriangleMiddle,
    TriangleFreq,
    EdgeDetour,
    TriangleDetour,
}

pub enum ObservableValue {
    Number(usize),
    List(Vec<usize>),
}

impl fmt::Display for ObservableValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ObservableValue::Number(num) => write!(f, "{num}"),
            ObservableValue::List(vec) => {
                if let Some((first, rest)) = vec.split_first() {
                    write!(f, "{first}")?;
                    for num in rest {
                        write!(f, ", {num}")?;
                    }
                }
                Ok(())
            }
        }
    }
}

impl Observable {
    pub fn name(&self) -> &str {
        match *self {
            Observable::Volume => "volume",
            Observable::DualGraphShell => "dual_graph_shell",
            Observable::DualTreeShell => "dual_tree_shell",
            Observable::VertexTreeShell => "vertex_tree_shell",
            Observable::VertexCount => "vertex_count",
            Observable::VertexDegree => "vertex_degree",
            Observable::EdgeDegree => "edge_degree",
            Observable::EdgeMiddle => "edge_middle",
            Observable::TriangleMiddle => "triangle_middle",
            Observable::EdgeDetour => "edge_detour",
            Observable::TriangleDetour => "triangle_detour",
            Observable::EdgeFreq => "edge_freq",
            Observable::TriangleFreq => "triangle_freq",
        }
    }

    pub fn observe(&self, triangulation: &mut Triangulation) -> ObservableValue {
        match self {
            Observable::Volume => ObservableValue::Number(triangulation.volume()),
            Observable::VertexCount => ObservableValue::Number(triangulation.vertex_count()),
            Observable::DualGraphShell => {
                let seed = fastrand::usize(0..(triangulation.volume()));
                let shells = triangulation.dual_graph().shells(seed);
                ObservableValue::List(shells)
            }
            Observable::DualTreeShell => {
                let seed = fastrand::usize(0..(triangulation.volume()));
                let shells = triangulation.dual_tree().shells(seed);
                ObservableValue::List(shells)
            }
            Observable::VertexTreeShell => {
                let seed = fastrand::usize(0..(triangulation.vertex_count()));
                let shells = triangulation.vertex_tree().shells(seed);
                ObservableValue::List(shells)
            }
            Observable::VertexDegree => ObservableValue::List(triangulation.vertex_degree()),
            Observable::EdgeDegree => ObservableValue::List(triangulation.edge_degree()),
            Observable::EdgeFreq => {
                let edges = triangulation.vertex_graph().edges();
                let freqdist = freqdist(edges);
                ObservableValue::List(freqdist)
            }
            Observable::EdgeMiddle => ObservableValue::List(triangulation.edge_middle()),
            Observable::TriangleFreq => {
                let edges = triangulation.dual_graph().edges();
                let freqdist = freqdist(edges);
                ObservableValue::List(freqdist)
            }
            Observable::TriangleMiddle => ObservableValue::List(triangulation.triangle_middle()),
            Observable::EdgeDetour => ObservableValue::List(triangulation.edge_detour()),
            Observable::TriangleDetour => ObservableValue::List(triangulation.triangle_detour()),
        }
    }
}

impl Index<Mouse> for Pool<Tetrahedron> {
    type Output = Mouse;
    fn index(&self, mouse: Mouse) -> &Self::Output {
        &self[mouse.tet].adj[mouse.half_edge as usize]
    }
}

impl IndexMut<Mouse> for Pool<Tetrahedron> {
    fn index_mut(&mut self, mouse: Mouse) -> &mut Self::Output {
        &mut self[mouse.tet].adj[mouse.half_edge as usize]
    }
}

impl fmt::Display for Triangulation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "triangulation:")?;
        write!(
            f,
            "tets:\n{}\nedges:\n{}\nvertices:\n{}\ntet_bag:\n{}\nmiddle_edges:\n{}\nmiddle_triangles:\n{}",
            self.tets, self.edges, self.vertices, self.tet_bag, self.middle_edges, self.middle_triangles
        )?;
        Ok(())
    }
}

impl fmt::Display for Tetrahedron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\nadj:")?;
        self.adj
            .iter()
            .enumerate()
            .try_for_each(|(i, mouse)| writeln!(f, "\t[{i}]:\t{mouse}"))?;
        writeln!(f, "vertices:")?;
        self.vertices
            .iter()
            .enumerate()
            .try_for_each(|(i, vertex)| writeln!(f, "\t[{i}]:\t{vertex}"))?;
        Ok(())
    }
}

impl fmt::Display for Vertex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mouse: {}, \t node: {}", self.mouse, self.node)
    }
}

impl fmt::Display for Edge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mouse: {}, \t node: {}", self.mouse, self.node)
    }
}

impl fmt::Display for Triangle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mouse: {}, \t node: {}", self.mouse, self.node)
    }
}

impl fmt::Display for Mouse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.tet, self.half_edge as usize)
    }
}
