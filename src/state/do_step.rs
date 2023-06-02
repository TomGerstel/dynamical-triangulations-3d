use crate::{
    state::{
        tables::{StateType, TreeState, TreeStateId, TreeTables},
        triangulation::Vertex,
        Edge, Label, Mouse, Outcome, Step, Triangle, Triangulation,
    },
    Ensemble,
};

use super::triangulation::Tetrahedron;

#[derive(Debug)]
struct ForestSimplices {
    triangles: Vec<Label<Triangle>>,
    tets_a: Vec<Label<Tetrahedron>>,
    tets_b: Vec<Label<Tetrahedron>>,
    edges: Vec<Label<Edge>>,
    verts_a: Vec<Label<Vertex>>,
    verts_b: Vec<Label<Vertex>>,
}

enum ForestOutcome {
    Valid,
    Invalid,
    Metropolis,
}

impl Triangulation {
    pub fn do_step(&mut self, step: Step, tables: &TreeTables, ensemble: Ensemble) -> Outcome {
        match step {
            Step::Move02 { mouse } => self.move_02(mouse, tables, ensemble),
            Step::Move20 { mouse } => self.move_20(mouse, tables, ensemble),
            Step::Move23 { mouse } => self.move_23(mouse, tables, ensemble),
            Step::Move32 { mouse } => self.move_32(mouse, tables, ensemble),
            Step::VertexTree { edge, seed } => self.vertex_tree_move(edge, seed, ensemble),
            Step::DualTree { edge, seed } => self.dual_tree_move(edge, seed, ensemble),
            Step::Rejected => Outcome::VolumeMH,
        }
    }

    fn move_02(&mut self, mouse: Mouse, tables: &TreeTables, ensemble: Ensemble) -> Outcome {
        // define mice
        let a = mouse;
        let b = a.adj_ext(self.tets());

        // define vertices
        let v0 = a.tail(self.tets());
        let v1 = a.head(self.tets());
        let v2 = a.next().head(self.tets());
        let v3 = self.insert_vertex(a, 3);

        // define edges
        let e0 = a.edge(self.tets());
        let e1 = a.next().edge(self.tets());
        let e2 = a.next().next().edge(self.tets());
        let e3 = self.insert_edge(a, 2);
        let e4 = self.insert_edge(a, 2);
        let e5 = self.insert_edge(a, 2);

        // define triangles
        let t0 = a.triangle(self.tets());
        let t1 = self.insert_triangle(a);
        let t2 = self.insert_triangle(a);
        let t3 = self.insert_triangle(a);
        let t4 = self.insert_triangle(a);

        // define tetrahedra
        let tet0 = a.tet();
        let tet1 = b.tet();
        let tet2 = self.insert_tet([v1, v0, v2, v3], [e0, e2, e1, e4, e3, e5], [t0, t2, t4, t3]);
        let tet3 = self.insert_tet([v0, v1, v2, v3], [e0, e1, e2, e3, e4, e5], [t1, t2, t3, t4]);

        let c = tet2.root();
        let d = tet3.root();

        // update forest
        let simplices_curr = ForestSimplices {
            triangles: vec![t0],
            tets_a: vec![tet0],
            tets_b: vec![tet1],
            edges: vec![e0, e1, e2],
            verts_a: vec![v0, v1, v0],
            verts_b: vec![v1, v2, v2],
        };
        let simplices_dest = ForestSimplices {
            triangles: vec![t0, t1, t2, t3, t4],
            tets_a: vec![tet0, tet1, tet2, tet2, tet2],
            tets_b: vec![tet2, tet3, tet3, tet3, tet3],
            edges: vec![e0, e1, e2, e3, e4, e5],
            verts_a: vec![v0, v1, v0, v0, v1, v2],
            verts_b: vec![v1, v2, v2, v3, v3, v3],
        };
        match ensemble {
            Ensemble::Undecorated => (),
            _ => {
                match self.forest_update(
                    simplices_curr,
                    simplices_dest,
                    tables,
                    StateType::Shard0,
                    ensemble,
                ) {
                    ForestOutcome::Valid => (),
                    _ => unreachable!(),
                }
            }
        }

        // glue triangles
        self.tets_mut().glue_triangles(a, c);
        self.tets_mut().glue_triangles(b, d);
        self.tets_mut().glue_triangles(c.adj_int(), d.adj_int());
        self.tets_mut()
            .glue_triangles(c.next().adj_int(), d.next().next().adj_int());
        self.tets_mut()
            .glue_triangles(c.next().next().adj_int(), d.next().adj_int());

        // split triangle
        self.tets_mut().set_base_triangle(b, t1);

        // insert simplices
        self.insert_tet_bag(tet2);
        self.insert_tet_bag(tet3);

        // update degrees
        self.increment_vertex_degree(v0);
        self.increment_vertex_degree(v1);
        self.increment_vertex_degree(v2);
        self.increment_edge_degree(e0);
        self.increment_edge_degree(e0);
        self.increment_edge_degree(e1);
        self.increment_edge_degree(e1);
        self.increment_edge_degree(e2);
        self.increment_edge_degree(e2);

        // update mice
        self.set_mouse_vertex(v3, c.adj_int().next().next());
        self.set_mouse_edge(e3, c.adj_int().next().next());
        self.set_mouse_edge(e4, c.adj_int().next());
        self.set_mouse_edge(e5, c.next().adj_int().next().next());
        self.set_mouse_triangle(t0, a);
        self.set_mouse_triangle(t1, b);
        self.set_mouse_triangle(t2, c.adj_int());
        self.set_mouse_triangle(t3, c.next().next().adj_int());
        self.set_mouse_triangle(t4, c.next().adj_int());
        Outcome::Valid02
    }

    fn move_20(&mut self, mouse: Mouse, tables: &TreeTables, ensemble: Ensemble) -> Outcome {
        // check geometry
        if !self.is_allow_20(mouse) {
            return Outcome::Geometry20;
        }

        // identify mice
        let a = mouse.adj_int().adj_ext(self.tets());
        let b = mouse.adj_ext(self.tets()).adj_int().adj_ext(self.tets());
        let c = mouse.adj_int();
        let d = mouse.adj_ext(self.tets()).adj_int();

        // identify simplices
        let v0 = a.tail(self.tets());
        let v1 = a.head(self.tets());
        let v2 = c.next().head(self.tets());
        let v3 = c.adj_int().next().head(self.tets());
        let e0 = a.edge(self.tets());
        let e1 = a.next().edge(self.tets());
        let e2 = a.next().next().edge(self.tets());
        let e3 = d.adj_int().next().edge(self.tets());
        let e4 = d.adj_int().next().next().edge(self.tets());
        let e5 = c.next().adj_int().next().next().edge(self.tets());
        let t0 = a.triangle(self.tets());
        let t1 = b.triangle(self.tets());
        let t2 = c.adj_int().triangle(self.tets());
        let t3 = d.next().adj_int().triangle(self.tets());
        let t4 = c.next().adj_int().triangle(self.tets());
        let tet0 = a.tet();
        let tet1 = b.tet();
        let tet2 = c.tet();
        let tet3 = d.tet();

        // update forest
        let simplices_curr = ForestSimplices {
            triangles: vec![t0, t1, t2, t3, t4],
            tets_a: vec![tet0, tet1, tet2, tet2, tet2],
            tets_b: vec![tet2, tet3, tet3, tet3, tet3],
            edges: vec![e0, e1, e2, e3, e4, e5],
            verts_a: vec![v0, v1, v0, v0, v1, v2],
            verts_b: vec![v1, v2, v2, v3, v3, v3],
        };
        let simplices_dest = ForestSimplices {
            triangles: vec![t0],
            tets_a: vec![tet0],
            tets_b: vec![tet1],
            edges: vec![e0, e1, e2],
            verts_a: vec![v0, v1, v0],
            verts_b: vec![v1, v2, v2],
        };

        match ensemble {
            Ensemble::Undecorated => (),
            _ => {
                match self.forest_update(
                    simplices_curr,
                    simplices_dest,
                    tables,
                    StateType::Shard2,
                    ensemble,
                ) {
                    ForestOutcome::Valid => (),
                    ForestOutcome::Invalid => {
                        return Outcome::Forest20;
                    }
                    ForestOutcome::Metropolis => {
                        return Outcome::ForestMH20;
                    }
                }
            }
        }

        // glue triangles
        self.tets_mut().glue_triangles(a, b);

        // merge triangles
        self.tets_mut().set_base_triangle(b, t0);

        // remove simplices
        self.remove_vertex(v3);
        self.remove_edge(e3);
        self.remove_edge(e4);
        self.remove_edge(e5);
        self.remove_triangle(t1);
        self.remove_triangle(t2);
        self.remove_triangle(t3);
        self.remove_triangle(t4);
        self.remove_tet(tet2);
        self.remove_tet(tet3);
        self.remove_tet_bag(tet2);
        self.remove_tet_bag(tet3);

        // update degrees
        self.decrement_vertex_degree(v0);
        self.decrement_vertex_degree(v1);
        self.decrement_vertex_degree(v2);
        self.decrement_edge_degree(e0);
        self.decrement_edge_degree(e0);
        self.decrement_edge_degree(e1);
        self.decrement_edge_degree(e1);
        self.decrement_edge_degree(e2);
        self.decrement_edge_degree(e2);

        // update mice
        self.set_mouse_vertex(v0, a);
        self.set_mouse_vertex(v1, a.next());
        self.set_mouse_vertex(v2, a.next().next());
        self.set_mouse_edge(e0, a);
        self.set_mouse_edge(e1, a.next());
        self.set_mouse_edge(e2, a.next().next());
        self.set_mouse_triangle(t0, a);
        Outcome::Valid20
    }

    fn move_23(&mut self, mouse: Mouse, tables: &TreeTables, ensemble: Ensemble) -> Outcome {
        // identify halfedges
        let a = mouse;
        let b = mouse.adj_ext(self.tets()).next();

        // identify vertices
        let v0 = a.adj_int().next().head(self.tets());
        let v1 = a.tail(self.tets());
        let v2 = a.head(self.tets());
        let v3 = b.head(self.tets());
        let v4 = b.adj_int().next().head(self.tets());

        if v0 == v4 {
            return Outcome::EnsembleRestriction;
        }

        // identify edges
        let e0 = a.edge(self.tets());
        let e1 = b.next().edge(self.tets());
        let e2 = b.edge(self.tets());
        let e3 = b.adj_int().next().edge(self.tets());
        let e4 = b.next().adj_int().next().next().edge(self.tets());
        let e5 = b.adj_int().next().next().edge(self.tets());
        let e6 = a.adj_int().next().edge(self.tets());
        let e7 = a.adj_int().next().next().edge(self.tets());
        let e8 = a.next().adj_int().next().next().edge(self.tets());
        let e9 = self.insert_edge(a, 3);

        // identify triangles
        let t0 = a.adj_int().triangle(self.tets());
        let t1 = b.next().adj_int().triangle(self.tets());
        let t2 = a.next().next().adj_int().triangle(self.tets());
        let t3 = b.next().next().adj_int().triangle(self.tets());
        let t4 = a.next().adj_int().triangle(self.tets());
        let t5 = b.adj_int().triangle(self.tets());
        let t6 = a.triangle(self.tets());
        let t7 = self.insert_triangle(a);
        let t8 = self.insert_triangle(a);

        // identify tetrahedra
        let tet0 = a.adj_int().adj_ext(self.tets()).tet();
        let tet1 = b.next().adj_int().adj_ext(self.tets()).tet();
        let tet2 = a.next().next().adj_int().adj_ext(self.tets()).tet();
        let tet3 = b.next().next().adj_int().adj_ext(self.tets()).tet();
        let tet4 = a.next().adj_int().adj_ext(self.tets()).tet();
        let tet5 = b.adj_int().adj_ext(self.tets()).tet();
        let tet6 = a.tet();
        let tet8 = self.insert_tet([v4, v0, v2, v3], [e9, e7, e4, e5, e8, e1], [t6, t8, t4, t1]);
        let tet7 = b.tet();

        // update forest
        let simplices_curr = ForestSimplices {
            triangles: vec![t0, t1, t2, t3, t4, t5, t6],
            tets_a: vec![tet0, tet1, tet2, tet3, tet4, tet5, tet6],
            tets_b: vec![tet6, tet7, tet6, tet7, tet6, tet7, tet7],
            edges: vec![e0, e1, e2, e3, e4, e5, e6, e7, e8],
            verts_a: vec![v1, v2, v1, v1, v2, v3, v0, v0, v0],
            verts_b: vec![v2, v3, v3, v4, v4, v4, v1, v2, v3],
        };
        let simplices_dest = ForestSimplices {
            triangles: vec![t0, t1, t2, t3, t4, t5, t6, t7, t8],
            tets_a: vec![tet0, tet1, tet2, tet3, tet4, tet5, tet6, tet6, tet7],
            tets_b: vec![tet6, tet8, tet7, tet6, tet8, tet7, tet8, tet7, tet8],
            edges: vec![e0, e1, e2, e3, e4, e5, e6, e7, e8, e9],
            verts_a: vec![v1, v2, v1, v1, v2, v3, v0, v0, v0, v0],
            verts_b: vec![v2, v3, v3, v4, v4, v4, v1, v2, v3, v4],
        };

        match ensemble {
            Ensemble::Undecorated => (),
            _ => {
                match self.forest_update(
                    simplices_curr,
                    simplices_dest,
                    tables,
                    StateType::Flip2,
                    ensemble,
                ) {
                    ForestOutcome::Valid => (),
                    ForestOutcome::Invalid => {
                        self.remove_edge(e9);
                        self.remove_triangle(t7);
                        self.remove_triangle(t8);
                        self.remove_tet(tet8);
                        return Outcome::Forest23;
                    }
                    ForestOutcome::Metropolis => {
                        self.remove_edge(e9);
                        self.remove_triangle(t7);
                        self.remove_triangle(t8);
                        self.remove_tet(tet8);
                        return Outcome::ForestMH23;
                    }
                }
            }
        }

        // identify old boundary
        let boundary2 = [
            a.next().adj_int(),
            a.next().next().adj_int(),
            b.next().adj_int(),
            b.next().next().adj_int(),
        ];

        // create dummy simplex
        let dummy_root = self
            .insert_tet([v0, v0, v0, v0], [e0, e0, e0, e0, e0, e0], [t0, t0, t0, t0])
            .root();
        let dummy_mice = [
            dummy_root,
            dummy_root.adj_int(),
            dummy_root.next().adj_int(),
            dummy_root.next().next().adj_int(),
        ];

        // glue dummy to old boundary
        for i in 0..4 {
            let boundary = boundary2[i].adj_ext(self.tets());
            self.tets_mut().glue_triangles(dummy_mice[i], boundary);
        }

        // insert new tetrahedron and update subsimplices
        let c = tet8.root();
        self.insert_tet_bag(c.tet());
        self.tets_mut().set_vertices(a, [v1, v2, v4, v0]);
        self.tets_mut().set_vertices(b, [v1, v3, v0, v4]);
        self.tets_mut().set_edges(a, [e0, e4, e3, e6, e7, e9]);
        self.tets_mut().set_edges(b, [e2, e8, e6, e3, e5, e9]);
        self.tets_mut().set_triangles(a, [t3, t0, t6, t7]);
        self.tets_mut().set_triangles(b, [t2, t5, t8, t7]);

        // update degrees
        self.increment_vertex_degree(v0);
        self.increment_vertex_degree(v4);
        self.decrement_edge_degree(e0);
        self.decrement_edge_degree(e1);
        self.decrement_edge_degree(e2);
        self.increment_edge_degree(e3);
        self.increment_edge_degree(e4);
        self.increment_edge_degree(e5);
        self.increment_edge_degree(e6);
        self.increment_edge_degree(e7);
        self.increment_edge_degree(e8);

        // update mice
        self.set_mouse_vertex(v0, c.next());
        self.set_mouse_vertex(v1, a);
        self.set_mouse_vertex(v2, a.next());
        self.set_mouse_vertex(v3, b.next());
        self.set_mouse_vertex(v4, c);
        self.set_mouse_edge(e0, a);
        self.set_mouse_edge(e1, c.next().adj_int().next().next());
        self.set_mouse_edge(e2, b);
        self.set_mouse_edge(e3, a.next().next());
        self.set_mouse_edge(e4, a.next());
        self.set_mouse_edge(e5, b.adj_int().next().next());
        self.set_mouse_edge(e6, b.next().next());
        self.set_mouse_edge(e7, c.next());
        self.set_mouse_edge(e8, b.next());
        self.set_mouse_edge(e9, c);
        self.set_mouse_triangle(t0, a.adj_int());
        self.set_mouse_triangle(t1, c.next().next().adj_int());
        self.set_mouse_triangle(t2, b);
        self.set_mouse_triangle(t3, a);
        self.set_mouse_triangle(t4, c.next().adj_int());
        self.set_mouse_triangle(t5, b.adj_int());
        self.set_mouse_triangle(t6, c);
        self.set_mouse_triangle(t7, a.next().next().adj_int());
        self.set_mouse_triangle(t8, c.adj_int());

        // identify new boundary
        let boundary3 = [
            c.adj_int().next().next().adj_int().next(),
            b,
            c.next().next().adj_int().next(),
            a,
        ];

        // replace dummy with new simplex
        for i in 0..4 {
            let dummy = dummy_mice[i].adj_ext(self.tets());
            self.tets_mut().glue_triangles(dummy, boundary3[i]);
        }

        // remove dummy
        self.remove_tet(dummy_root.tet());

        // internal gluing
        self.tets_mut()
            .glue_triangles(a.next().next().adj_int(), b.adj_int().next().adj_int());
        self.tets_mut()
            .glue_triangles(b.next().adj_int().next().next(), c.adj_int());
        self.tets_mut()
            .glue_triangles(c, a.next().adj_int().next().next());
        Outcome::Valid23
    }

    fn move_32(&mut self, mouse: Mouse, tables: &TreeTables, ensemble: Ensemble) -> Outcome {
        // check geometry
        if !self.is_allow_32(mouse) {
            return Outcome::Geometry32;
        }

        // identify halfedges
        let a = mouse
            .next()
            .next()
            .adj_ext(self.tets())
            .adj_int()
            .next()
            .next();
        let b = mouse
            .adj_int()
            .adj_ext(self.tets())
            .next()
            .adj_int()
            .next()
            .next();
        let c = mouse;

        // identify vertices
        let v0 = c.head(self.tets());
        let v1 = a.tail(self.tets());
        let v2 = a.head(self.tets());
        let v3 = b.head(self.tets());
        let v4 = c.tail(self.tets());

        // identify edges
        let e0 = a.edge(self.tets());
        let e1 = c.next().next().adj_int().next().edge(self.tets());
        let e2 = b.edge(self.tets());
        let e3 = a.next().next().edge(self.tets());
        let e4 = a.next().edge(self.tets());
        let e5 = b.adj_int().next().next().edge(self.tets());
        let e6 = b.next().next().edge(self.tets());
        let e7 = c.next().edge(self.tets());
        let e8 = b.next().edge(self.tets());
        let e9 = c.edge(self.tets());

        // identify triangles
        let t0 = a.adj_int().triangle(self.tets());
        let t1 = c.next().next().adj_int().triangle(self.tets());
        let t2 = b.triangle(self.tets());
        let t3 = a.triangle(self.tets());
        let t4 = c.next().adj_int().triangle(self.tets());
        let t5 = b.adj_int().triangle(self.tets());
        let t6 = c.triangle(self.tets());
        let t7 = a.next().next().adj_int().triangle(self.tets());
        let t8 = b.next().adj_int().triangle(self.tets());

        // identify tetrahedra
        let tet0 = a.adj_int().adj_ext(self.tets()).tet();
        let tet1 = c.next().next().adj_int().adj_ext(self.tets()).tet();
        let tet2 = b.adj_ext(self.tets()).tet();
        let tet3 = a.adj_ext(self.tets()).tet();
        let tet4 = c.next().adj_int().adj_ext(self.tets()).tet();
        let tet5 = b.adj_int().adj_ext(self.tets()).tet();
        let tet6 = a.tet();
        let tet7 = b.tet();
        let tet8 = c.tet();

        // update forest
        let simplices_curr = ForestSimplices {
            triangles: vec![t0, t1, t2, t3, t4, t5, t6, t7, t8],
            tets_a: vec![tet0, tet1, tet2, tet3, tet4, tet5, tet6, tet6, tet7],
            tets_b: vec![tet6, tet8, tet7, tet6, tet8, tet7, tet8, tet7, tet8],
            edges: vec![e0, e1, e2, e3, e4, e5, e6, e7, e8, e9],
            verts_a: vec![v1, v2, v1, v1, v2, v3, v0, v0, v0, v0],
            verts_b: vec![v2, v3, v3, v4, v4, v4, v1, v2, v3, v4],
        };
        let simplices_dest = ForestSimplices {
            triangles: vec![t0, t1, t2, t3, t4, t5, t6],
            tets_a: vec![tet0, tet1, tet2, tet3, tet4, tet5, tet6],
            tets_b: vec![tet6, tet7, tet6, tet7, tet6, tet7, tet7],
            edges: vec![e0, e1, e2, e3, e4, e5, e6, e7, e8],
            verts_a: vec![v1, v2, v1, v1, v2, v3, v0, v0, v0],
            verts_b: vec![v2, v3, v3, v4, v4, v4, v1, v2, v3],
        };

        match ensemble {
            Ensemble::Undecorated => (),
            _ => {
                match self.forest_update(
                    simplices_curr,
                    simplices_dest,
                    tables,
                    StateType::Flip3,
                    ensemble,
                ) {
                    ForestOutcome::Valid => (),
                    ForestOutcome::Invalid => {
                        return Outcome::Forest32;
                    }
                    ForestOutcome::Metropolis => {
                        return Outcome::ForestMH32;
                    }
                }
            }
        }

        // create dummy simplex
        let dummy_root = self
            .insert_tet([v0, v0, v0, v0], [e0, e0, e0, e0, e0, e0], [t6, t6, t6, t6])
            .root();
        let dummy_mice = [
            dummy_root,
            dummy_root.adj_int(),
            dummy_root.next().adj_int(),
            dummy_root.next().next().adj_int(),
        ];

        // find old boundary
        let bound3 = [
            c.adj_int().next().next().adj_int().next(),
            b,
            c.next().next().adj_int().next(),
            a,
        ];

        // glue dummy to external boundary
        for i in 0..4 {
            let boundary = bound3[i].adj_ext(self.tets());
            self.tets_mut().glue_triangles(dummy_mice[i], boundary);
        }

        // remove simplices
        self.remove_edge(e9);
        self.remove_triangle(t7);
        self.remove_triangle(t8);
        self.remove_tet(tet8);
        self.remove_tet_bag(tet8);

        // internal gluing
        self.tets_mut().glue_triangles(a, b.next().next());

        // find triangles that should go to new boundary
        let bound2 = [
            a.next().adj_int(),
            a.next().next().adj_int(),
            b.next().adj_int(),
            b.next().next().adj_int(),
        ];

        // glue new boundary to dummy neighbours
        for i in 0..4 {
            let dummy = dummy_mice[i].adj_ext(self.tets());
            self.tets_mut().glue_triangles(dummy, bound2[i]);
        }

        // remove dummy
        self.remove_tet(dummy_root.tet());

        // update simplices
        self.tets_mut().set_vertices(a, [v1, v2, v3, v0]);
        self.tets_mut().set_vertices(b, [v1, v3, v2, v4]);
        self.tets_mut().set_edges(a, [e0, e1, e2, e6, e7, e8]);
        self.tets_mut().set_edges(b, [e2, e1, e0, e3, e5, e4]);
        self.tets_mut().set_triangles(a, [t6, t0, t4, t2]);
        self.tets_mut().set_triangles(b, [t6, t5, t1, t3]);

        // update degrees
        self.decrement_vertex_degree(v0);
        self.decrement_vertex_degree(v4);
        self.increment_edge_degree(e0);
        self.increment_edge_degree(e1);
        self.increment_edge_degree(e2);
        self.decrement_edge_degree(e3);
        self.decrement_edge_degree(e4);
        self.decrement_edge_degree(e5);
        self.decrement_edge_degree(e6);
        self.decrement_edge_degree(e7);
        self.decrement_edge_degree(e8);

        // update mice
        self.set_mouse_vertex(v0, a.adj_int().next().next());
        self.set_mouse_vertex(v1, a);
        self.set_mouse_vertex(v2, a.next());
        self.set_mouse_vertex(v3, b.next());
        self.set_mouse_vertex(v4, b.adj_int().next().next());
        self.set_mouse_edge(e0, a);
        self.set_mouse_edge(e1, a.next());
        self.set_mouse_edge(e2, b);
        self.set_mouse_edge(e3, b.adj_int().next());
        self.set_mouse_edge(e4, b.next().adj_int().next().next());
        self.set_mouse_edge(e5, b.adj_int().next().next());
        self.set_mouse_edge(e6, a.adj_int().next());
        self.set_mouse_edge(e7, a.adj_int().next().next());
        self.set_mouse_edge(e8, a.next().adj_int().next().next());
        self.set_mouse_triangle(t0, a.adj_int());
        self.set_mouse_triangle(t1, b.next().adj_int());
        self.set_mouse_triangle(t2, a.next().next().adj_int());
        self.set_mouse_triangle(t3, b.next().next().adj_int());
        self.set_mouse_triangle(t4, a.next().adj_int());
        self.set_mouse_triangle(t5, b.adj_int());
        self.set_mouse_triangle(t6, a);
        Outcome::Valid32
    }

    fn is_allow_20(&mut self, mouse: Mouse) -> bool {
        let mice = [
            mouse.adj_int().next().adj_int(),
            mouse.adj_int().next().next().adj_int(),
            mouse
                .adj_ext(self.tets())
                .adj_int()
                .next()
                .next()
                .adj_int()
                .adj_ext(self.tets()),
            mouse
                .adj_ext(self.tets())
                .adj_int()
                .next()
                .adj_int()
                .adj_ext(self.tets()),
        ];
        mice[0] == mice[2] && mice[1] == mice[3]
    }

    fn is_allow_32(&mut self, mouse: Mouse) -> bool {
        mouse
            == mouse
                .adj_int()
                .adj_ext(self.tets())
                .adj_int()
                .adj_ext(self.tets())
                .adj_int()
                .adj_ext(self.tets())
            && mouse.adj_int().adj_ext(self.tets()).tet() != mouse.tet()
            && mouse
                .adj_int()
                .adj_ext(self.tets())
                .adj_int()
                .adj_ext(self.tets())
                .tet()
                != mouse.tet()
            && mouse
                .adj_int()
                .adj_ext(self.tets())
                .adj_int()
                .adj_ext(self.tets())
                .tet()
                != mouse.adj_int().adj_ext(self.tets()).tet()
    }

    fn forest_update(
        &mut self,
        simplices_curr: ForestSimplices,
        simplices_dest: ForestSimplices,
        tables: &TreeTables,
        state_type_curr: StateType,
        ensemble: Ensemble,
    ) -> ForestOutcome {
        let n_triangles_curr = simplices_curr.triangles.len();
        let n_edges_curr = simplices_curr.edges.len();
        let n_triangles_dest = simplices_dest.triangles.len();
        let n_edges_dest = simplices_dest.edges.len();
        let state_type_dest = match state_type_curr {
            StateType::Shard0 => StateType::Shard2,
            StateType::Shard2 => StateType::Shard0,
            StateType::Flip2 => StateType::Flip3,
            StateType::Flip3 => StateType::Flip2,
        };
        let middle_map_curr = state_type_curr.middle_map();
        let middle_map_dest = state_type_dest.middle_map();

        // find the ID of the current configuration
        let curr_state = TreeState::new(
            simplices_curr
                .triangles
                .iter()
                .map(|&triangle| !self.is_middle_triangle(triangle))
                .collect::<Vec<bool>>(),
            simplices_curr
                .edges
                .iter()
                .map(|&edge| !self.is_middle_edge(edge))
                .collect::<Vec<bool>>(),
        );

        // find the possible destinations
        let curr_id = curr_state.id(n_triangles_curr, n_edges_curr);
        let Some(destinations) = tables.destinations(state_type_curr, curr_id) else {
            return ForestOutcome::Invalid;
        };

        // choose a destination (if possible)
        let n_into = destinations.len();
        let dest_id: TreeStateId = {
            let index = fastrand::usize(0..n_into);
            destinations[index]
        };
        let dest_state = TreeState::from(dest_id);

        // maybe reject the step
        let n_from = tables.destinations(state_type_dest, dest_id).unwrap().len();
        if fastrand::f32() > p_accept_bipartite(n_from, n_into) {
            return ForestOutcome::Metropolis;
        }

        // cut dual tree
        for i in 0..n_triangles_curr {
            if curr_state.triangles()[i] {
                self.cut_dual_tree(
                    simplices_curr.tets_a[i],
                    simplices_curr.tets_b[i],
                    simplices_curr.triangles[i],
                );
            }
        }

        // cut vertex tree
        for i in 0..n_edges_curr {
            if curr_state.edges()[i] {
                self.cut_vertex_tree(
                    simplices_curr.verts_a[i],
                    simplices_curr.verts_b[i],
                    simplices_curr.edges[i],
                );
            }
        }

        match ensemble {
            Ensemble::Undecorated => unreachable!(),
            Ensemble::Spanning => (),
            Ensemble::TripleTrees => {
                // cut middle tree
                for (t, e) in middle_map_curr {
                    if !curr_state.triangles()[*t] && !curr_state.edges()[*e] {
                        self.cut_middle_tree(
                            simplices_curr.triangles[*t],
                            simplices_curr.edges[*e],
                        );
                    }
                }

                // link middle tree according to destination
                for (t, e) in middle_map_dest {
                    if !dest_state.triangles()[*t] && !dest_state.edges()[*e] {
                        self.link_middle_tree(
                            simplices_dest.triangles[*t],
                            simplices_dest.edges[*e],
                        )
                    }
                }
            }
        }

        // link dual tree according to destination
        for i in 0..n_triangles_dest {
            if dest_state.triangles()[i] {
                self.link_dual_tree(
                    simplices_dest.tets_a[i],
                    simplices_dest.tets_b[i],
                    simplices_dest.triangles[i],
                );
            }
        }

        // link vertex tree according to destination
        for i in 0..n_edges_dest {
            if dest_state.edges()[i] {
                self.link_vertex_tree(
                    simplices_dest.verts_a[i],
                    simplices_dest.verts_b[i],
                    simplices_dest.edges[i],
                );
            }
        }

        ForestOutcome::Valid
    }
}

fn p_accept_bipartite(n: usize, m: usize) -> f32 {
    if n >= m {
        m as f32 / n as f32 * bipartite_ratio(n, m)
    } else {
        bipartite_ratio(m, n)
    }
}

fn bipartite_ratio(n: usize, m: usize) -> f32 {
    1_f32.min((2 * n) as f32 / (n + 2 * m) as f32)
}
