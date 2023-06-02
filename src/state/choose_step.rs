use crate::state::{MHProbs, State, Step, Triangulation};

enum StepType {
    Shard,
    Flip,
    VertexTree,
    DualTree,
}

impl State {
    pub fn choose_step(&mut self) -> Step {
        // sample step
        let step = self
            .triangulation
            .sample_step(self.acc_weights, &self.mh_probs);

        // accept or reject step
        if self.triangulation.accept_step(step, &self.mh_probs) {
            step
        } else {
            Step::Rejected
        }
    }
}

impl Triangulation {
    fn sample_step(&self, acc_weights: [usize; 4], mh_probs: &MHProbs) -> Step {
        let step_type = Triangulation::sample_step_type(acc_weights);
        let seed = fastrand::f32();
        let mouse = self.sample_mouse();
        let vol = self.volume();

        match step_type {
            StepType::Shard => {
                let prob = mh_probs.shard_select_grow[vol];
                if seed < prob {
                    Step::Move02 { mouse }
                } else {
                    Step::Move20 { mouse }
                }
            }
            StepType::Flip => {
                let prob = mh_probs.flip_select_grow[vol];
                if seed < prob {
                    Step::Move23 { mouse }
                } else {
                    Step::Move32 { mouse }
                }
            }
            StepType::VertexTree => Step::VertexTree {
                edge: self.sample_middle_edge(),
                seed: fastrand::f32(),
            },
            StepType::DualTree => Step::DualTree {
                edge: self.sample_middle_triangle(),
                seed: fastrand::f32(),
            },
        }
    }

    fn sample_step_type(acc_weights: [usize; 4]) -> StepType {
        let weight_seed = fastrand::usize(0..*acc_weights.last().unwrap());

        if weight_seed < acc_weights[0] {
            StepType::Shard
        } else if weight_seed < acc_weights[1] {
            StepType::Flip
        } else if weight_seed < acc_weights[2] {
            StepType::VertexTree
        } else if weight_seed < acc_weights[3] {
            StepType::DualTree
        } else {
            unreachable!("Step type seed outside valid range");
        }
    }

    fn accept_step(&self, step: Step, mh_probs: &MHProbs) -> bool {
        let n3 = self.volume();
        let p_accept = match step {
            Step::Move02 { mouse: _ } => mh_probs.shard_accept_grow[n3],
            Step::Move20 { mouse: _ } => mh_probs.shard_accept_shrink[n3],
            Step::Move23 { mouse: _ } => mh_probs.flip_accept_grow[n3],
            Step::Move32 { mouse: _ } => mh_probs.flip_accept_shrink[n3],
            Step::VertexTree { edge: _, seed: _ } => return true,
            Step::DualTree { edge: _, seed: _ } => return true,
            Step::Rejected => unreachable!(),
        };

        let seed = fastrand::f32();
        seed < p_accept
    }
}
