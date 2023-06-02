mod choose_step;
mod do_step;
mod graph;
mod tables;
mod triangulation;

use serde_derive::{Deserialize, Serialize};
pub use tables::TreeTables;
pub use triangulation::{GraphType, Observable};

use crate::{Ensemble, Model, Weights};

use std::fmt;
use triangulation::{Edge, Label, Mouse, Triangle, Triangulation};

#[derive(Debug, Clone)]
pub struct State {
    pub triangulation: Triangulation,
    acc_weights: [usize; 4],
    mh_probs: MHProbs,
    outcome_count: OutcomeCount,
    tree_tables: TreeTables,
    ensemble: Ensemble,
    pub meta_data: MetaData,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct MetaData {
    therm_outcome: OutcomeCount,
    tune_outcome: OutcomeCount,
    meas_outcome: OutcomeCount,
    timing: Timing,
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
struct Timing {
    table: u64,
    therm: u64,
    tune: u64,
    meas: u64,
}

impl State {
    pub fn new(model: &Model) -> Self {
        let triangulation = Triangulation::new(model.volume, model.ensemble);
        let acc_weights = model.weights.accumulate();
        let mh_probs = MHProbs::new(model);
        let outcome_count = OutcomeCount::default();
        let now = std::time::Instant::now();
        let tree_tables = TreeTables::new(model.ensemble);
        let elapsed = now.elapsed().as_secs();
        let ensemble = model.ensemble;
        let mut meta_data = MetaData::default();
        meta_data.timing.table = elapsed;
        State {
            triangulation,
            acc_weights,
            mh_probs,
            outcome_count,
            tree_tables,
            ensemble,
            meta_data,
        }
    }

    pub fn mc_step(&mut self) {
        let step = self.choose_step();

        let outcome = self
            .triangulation
            .do_step(step, &self.tree_tables, self.ensemble);
        match outcome {
            Outcome::Valid02 => self.outcome_count.valid02 += 1,
            Outcome::Valid20 => self.outcome_count.valid20 += 1,
            Outcome::Valid23 => self.outcome_count.valid23 += 1,
            Outcome::Valid32 => self.outcome_count.valid32 += 1,
            Outcome::ValidVertexTree => self.outcome_count.valid_vertex_tree += 1,
            Outcome::ValidDualTree => self.outcome_count.valid_dual_tree += 1,
            Outcome::Geometry20 => self.outcome_count.geometry20 += 1,
            Outcome::Geometry32 => self.outcome_count.geometry32 += 1,
            Outcome::Forest20 => self.outcome_count.forest20 += 1,
            Outcome::Forest23 => self.outcome_count.forest23 += 1,
            Outcome::Forest32 => self.outcome_count.forest32 += 1,
            Outcome::ForestMH20 => self.outcome_count.forest_mh20 += 1,
            Outcome::ForestMH23 => self.outcome_count.forest_mh23 += 1,
            Outcome::ForestMH32 => self.outcome_count.forest_mh32 += 1,
            Outcome::LeafVertex => self.outcome_count.leaf_vertex += 1,
            Outcome::LeafDual => self.outcome_count.leaf_dual += 1,
            Outcome::EnsembleRestriction => self.outcome_count.ensemble_restriction += 1,
            Outcome::VolumeMH => self.outcome_count.volume_mh += 1,
        }
    }

    pub fn set_therm_outcome(&mut self) {
        self.meta_data.therm_outcome = self.outcome_count;
        self.outcome_count = OutcomeCount::default();
    }

    pub fn set_tune_outcome(&mut self) {
        self.meta_data.tune_outcome = self.outcome_count;
        self.outcome_count = OutcomeCount::default();
    }

    pub fn set_meas_outcome(&mut self) {
        self.meta_data.meas_outcome = self.outcome_count;
    }

    pub fn set_therm_elapsed(&mut self, elapsed: u64) {
        self.meta_data.timing.therm = elapsed;
    }

    pub fn set_tune_elapsed(&mut self, elapsed: u64) {
        self.meta_data.timing.tune = elapsed;
    }

    pub fn set_meas_elapsed(&mut self, elapsed: u64) {
        self.meta_data.timing.meas = elapsed;
    }

    pub fn set_mh_probs(&mut self, model: &Model) {
        self.mh_probs = MHProbs::new(model);
    }
}

#[derive(Debug)]
pub enum Outcome {
    Valid02,
    Valid20,
    Valid23,
    Valid32,
    ValidVertexTree,
    ValidDualTree,
    Geometry20,
    Geometry32,
    Forest20,
    Forest23,
    Forest32,
    ForestMH20,
    ForestMH23,
    ForestMH32,
    LeafVertex,
    LeafDual,
    EnsembleRestriction,
    VolumeMH,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Step {
    Move02 { mouse: Mouse },
    Move20 { mouse: Mouse },
    Move23 { mouse: Mouse },
    Move32 { mouse: Mouse },
    VertexTree { edge: Label<Edge>, seed: f32 },
    DualTree { edge: Label<Triangle>, seed: f32 },
    Rejected,
}

impl fmt::Display for Step {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Move02 { mouse } => writeln!(f, "move02: {mouse}"),
            Self::Move20 { mouse } => writeln!(f, "move20: {mouse}"),
            Self::Move23 { mouse } => writeln!(f, "move23: {mouse}"),
            Self::Move32 { mouse } => writeln!(f, "move32: {mouse}"),
            Self::VertexTree { edge, seed } => writeln!(f, "vertex_tree: {edge} \t({seed})"),
            Self::DualTree { edge, seed } => writeln!(f, "dual_tree: {edge} \t ({seed})"),
            Self::Rejected => writeln!(f, "rejected"),
        }
    }
}

impl Weights {
    fn accumulate(&self) -> [usize; 4] {
        [
            self.shard,
            self.shard + self.flip,
            self.shard + self.flip + self.vertex_tree,
            self.shard + self.flip + self.vertex_tree + self.dual_tree,
        ]
    }
}

#[derive(Clone, Debug)]
struct SelectAccept {
    select_grow: Box<[f32]>,
    accept_grow: Box<[f32]>,
    accept_shrink: Box<[f32]>,
}

#[derive(Clone, Debug)]
struct MHProbs {
    shard_select_grow: Box<[f32]>,
    shard_accept_grow: Box<[f32]>,
    shard_accept_shrink: Box<[f32]>,
    flip_select_grow: Box<[f32]>,
    flip_accept_grow: Box<[f32]>,
    flip_accept_shrink: Box<[f32]>,
}

impl MHProbs {
    fn new(model: &Model) -> MHProbs {
        let shard = MHProbs::mh_probs(model, 2, 1);
        let flip = MHProbs::mh_probs(model, 1, 0);
        MHProbs {
            shard_select_grow: shard.select_grow,
            shard_accept_grow: shard.accept_grow,
            shard_accept_shrink: shard.accept_shrink,
            flip_select_grow: flip.select_grow,
            flip_accept_grow: flip.accept_grow,
            flip_accept_shrink: flip.accept_shrink,
        }
    }

    fn mh_probs(model: &Model, dn3: usize, dn0: usize) -> SelectAccept {
        let ratio = MHProbs::ratio(model, dn3, dn0);
        let norm = MHProbs::norm(&ratio, model.volume, dn3);
        let select_grow = MHProbs::select_grow(&ratio, &norm, model.volume, dn3);
        let (accept_grow, accept_shrink) = MHProbs::accept(&norm, model.volume, dn3);

        SelectAccept {
            select_grow,
            accept_grow,
            accept_shrink,
        }
    }

    fn ratio(model: &Model, dn3: usize, dn0: usize) -> Box<[f32]> {
        let n = model.volume;
        let sigma = model.sigma;
        let eps = 0.5 / (sigma * sigma);
        let k0 = model.kappa_0;
        let k3 = model.kappa_3.unwrap();
        (0..=(2 * n))
            .map(|n3| {
                let prefactor = n3 as f32 / ((n3 + dn3) as f32);
                let potential = dn3 as isize * (dn3 as isize + 2 * (n3 as isize - n as isize));
                let exponent = -k3 * (dn3 as f32) + k0 * (dn0 as f32) - eps * (potential as f32);
                prefactor * exponent.exp()
            })
            .collect()
    }

    fn norm(ratio: &[f32], n: usize, dn3: usize) -> Box<[f32]> {
        (0..=(2 * n))
            .map(|n3| {
                if n3 < dn3 {
                    std::f32::NAN
                } else {
                    1_f32.min(ratio[n3]) + 1_f32.min(1.0 / ratio[n3 - dn3])
                }
            })
            .collect()
    }

    fn select_grow(ratio: &[f32], norm: &[f32], n: usize, dn3: usize) -> Box<[f32]> {
        (0..=(2 * n))
            .map(|n3| {
                if n3 < 1 + dn3 {
                    1_f32
                } else if n3 + dn3 > 2 * n {
                    0_f32
                } else {
                    1_f32.min(ratio[n3]) / norm[n3]
                }
            })
            .collect()
    }

    fn accept(norm: &[f32], n: usize, dn3: usize) -> (Box<[f32]>, Box<[f32]>) {
        let accept_grow = (0..=(2 * n))
            .map(|n3| {
                if n3 < 1 + dn3 {
                    1_f32
                } else if n3 + dn3 > 2 * n {
                    0_f32
                } else {
                    1_f32.min(norm[n3] / norm[n3 + dn3])
                }
            })
            .collect();
        let accept_shrink = (0..(norm.len()))
            .map(|n3| {
                if n3 < 1 + dn3 {
                    0_f32
                } else if n3 + dn3 > 2 * n {
                    1_f32
                } else {
                    1_f32.min(norm[n3] / norm[n3 - dn3])
                }
            })
            .collect();
        (accept_grow, accept_shrink)
    }
}

#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
pub struct OutcomeCount {
    valid02: usize,
    valid20: usize,
    valid23: usize,
    valid32: usize,
    valid_vertex_tree: usize,
    valid_dual_tree: usize,
    geometry20: usize,
    geometry32: usize,
    forest20: usize,
    forest23: usize,
    forest32: usize,
    forest_mh20: usize,
    forest_mh23: usize,
    forest_mh32: usize,
    leaf_vertex: usize,
    leaf_dual: usize,
    ensemble_restriction: usize,
    volume_mh: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let model = Model::new(Ensemble::TripleTrees, 2, 0.1, 0.0, 3.0);
        let mut triangulation = Triangulation::new(model.volume, model.ensemble);
        assert_eq!(triangulation.volume(), 2);
        triangulation.sanity_check();
    }

    #[test]
    fn random_moves() {
        // initialise
        let model = Model::new(Ensemble::TripleTrees, 4, 2.0, 0.0, 3.0);
        let mut state = State::new(&model);

        // perform a large number of steps and check each iteration
        (0..(100_000 * model.volume)).for_each(|_| {
            let step = state.choose_step();
            match step {
                Step::Rejected => (),
                _ => {
                    state
                        .triangulation
                        .do_step(step, &state.tree_tables, state.ensemble);
                    state.triangulation.sanity_check();
                }
            };
        });
    }
}
