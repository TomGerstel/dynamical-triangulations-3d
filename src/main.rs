mod state;

use clap::Parser;
use serde_derive::{Deserialize, Serialize};
use state::{GraphType, Observable, State};
use std::{error::Error, fs, io::Write, path::Path, time};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn main() -> Result<()> {
    std::env::set_var("RUST_BACKTRACE", "1");
    println!();
    println!("Running...");

    // parse commandline arguments
    let args = Args::parse();

    // load config
    let mut config = load_config(&args)?;

    // equilibration and tuning phase
    let mut state = tune(&mut config);

    // measurement phase
    measure(&config, &mut state)?;
    println!("Done!");

    Ok(())
}

fn load_config(args: &Args) -> Result<Config> {
    let path = Path::new("configs").join(format!("{}.toml", args.config));
    let config_toml = fs::read_to_string(path)?;
    let multi_config: MultiConfig = toml::from_str(&config_toml)?;
    let config = multi_config.get(args);
    Ok(config)
}

fn tune(config: &mut Config) -> State {
    let fit = match config.model.ensemble {
        Ensemble::Undecorated => [2.35652, 1.01641, 0.17],
        Ensemble::Spanning => [3.49503, 2.54614, 0.17],
        Ensemble::TripleTrees => [2.61778, 2.26785, 0.17],
    };
    let number = 0.25 * config.model.kappa_0 - 0.5 * fit[0] + 0.5 * fit[1];
    let initial = number + fit[0] + (fit[2] + number * number).sqrt();

    config.model.kappa_3 = Some(initial);
    let mut state = thermalise(config);

    println!("Tuning cosmological constant...");
    println!("\tinitial: {initial}");

    let now = std::time::Instant::now();
    let mut error = config.model.sigma;
    while 5.0 * error >= config.model.sigma {
        let mean = (0..250)
            .map(|_| {
                for _ in 0..(config.model.volume * config.markov_chain.wait) {
                    state.mc_step();
                }
                state.triangulation.volume()
            })
            .sum::<usize>() as f32
            / 250.0;

        let d_n = mean - config.model.volume as f32;
        let d_kappa_3: f32 = d_n / (config.model.sigma * config.model.sigma);
        error = d_n.abs() / (config.model.volume as f32);

        config.model.kappa_3 = Some(config.model.kappa_3.unwrap() + d_kappa_3);
        state.set_mh_probs(&config.model);

        println!("\tupdated: {}", config.model.kappa_3.unwrap());
    }
    let elapsed = now.elapsed().as_secs();
    state.set_tune_elapsed(elapsed);
    state.set_tune_outcome();
    state
}

fn thermalise(config: &Config) -> State {
    println!("Initialising lookup tables...");
    let mut state = State::new(&config.model);
    println!("Thermalising...");
    let now = std::time::Instant::now();
    for _ in 0..(config.model.volume * config.markov_chain.thermalisation) {
        state.mc_step();
    }
    let elapsed = now.elapsed().as_secs();
    state.set_therm_elapsed(elapsed);
    state.set_therm_outcome();
    state
}

fn measure(config: &Config, state: &mut State) -> Result<()> {
    // create output directory
    let timestamp = time::SystemTime::now()
        .duration_since(time::UNIX_EPOCH)?
        .as_secs();
    let dirname = format!("{}_{}", config.markov_chain.output, timestamp);
    fs::create_dir(Path::new("output").join(&dirname))?;

    // create output files
    let mut obs_files = config
        .markov_chain
        .observables
        .iter()
        .map(|obs| {
            let filename = format!("{}.csv", obs.name());
            let path = Path::new("output").join(&dirname).join(filename);
            Ok((*obs, fs::File::create(path)?))
        })
        .collect::<Result<Vec<_>>>()?;

    // execute measurement phase and write results to files
    let now = std::time::Instant::now();
    println!("Measuring...");
    for _ in 0..(config.markov_chain.amount) {
        for _ in 0..(config.model.volume * config.markov_chain.wait) {
            state.mc_step();
        }
        for (observable, file) in &mut obs_files {
            let observed = observable.observe(&mut state.triangulation);
            writeln!(file, "{observed}")?;
        }
    }
    let elapsed = now.elapsed().as_secs();
    state.set_meas_elapsed(elapsed);

    // store graph data
    fs::create_dir(Path::new("graphs").join(&dirname))?;
    for graph_type in [
        GraphType::DualGraph,
        GraphType::VertexGraph,
        GraphType::DualTree,
        GraphType::VertexTree,
    ] {
        let filename = format!("{}.csv", graph_type.name());
        let path = Path::new("graphs").join(&dirname).join(filename);
        let mut file = fs::File::create(path)?;
        let edges = graph_type.generate(&state.triangulation).edges();
        for edge in edges {
            writeln!(file, "{edge}")?;
        }
    }

    // write used config to file
    let toml = toml::to_string(config).unwrap();
    let path = Path::new("output").join(&dirname).join("config.toml");
    let mut config_file = fs::File::create(path)?;
    write!(config_file, "{toml}")?;

    // save metadata
    state.set_meas_outcome();
    let toml = toml::to_string(&state.meta_data).unwrap();
    let path = Path::new("output").join(&dirname).join("meta_data.toml");
    let mut meta_file = fs::File::create(path)?;
    write!(meta_file, "{toml}")?;

    Ok(())
}

#[derive(clap::Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    config: String,
    volume_index: usize,
    phase_index: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Config {
    markov_chain: MarkovChain,
    model: Model,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct MarkovChain {
    thermalisation: usize,
    wait: usize,
    amount: usize,
    observables: Vec<Observable>,
    output: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Model {
    ensemble: Ensemble,
    volume: usize,
    sigma: f32,
    kappa_0: f32,
    kappa_3: Option<f32>,
    weights: Weights,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct MultiConfig {
    markov_chain: MultiMarkovChain,
    model: MultiModel,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct MultiMarkovChain {
    thermalisation: Multi<usize>,
    wait: Multi<usize>,
    amount: usize,
    observables: Vec<Observable>,
    output: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MultiModel {
    ensemble: Ensemble,
    volume: Multi<usize>,
    sigma: f32,
    kappa_0: Multi<f32>,
    kappa_3: Option<f32>,
    weights: Weights,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
enum Multi<T> {
    Scalar(T),
    Vector(Vec<T>),
    Matrix(Vec<Vec<T>>),
}

impl MultiConfig {
    fn get(&self, args: &Args) -> Config {
        let MultiConfig {
            markov_chain,
            model,
        } = self;
        let MultiMarkovChain {
            thermalisation,
            wait,
            amount,
            observables,
            output,
        } = markov_chain;
        let MultiModel {
            ensemble,
            volume,
            sigma,
            kappa_0,
            kappa_3,
            weights,
        } = model;
        let markov_chain = MarkovChain {
            thermalisation: thermalisation.get(args.volume_index, args.phase_index),
            wait: wait.get(args.volume_index, args.phase_index),
            amount: *amount,
            observables: observables.clone(),
            output: match output {
                None => {
                    let mut output = args.config.to_string();
                    output.push('_');
                    output.push_str(args.volume_index.to_string().as_str());
                    output.push('_');
                    output.push_str(args.phase_index.to_string().as_str());
                    output
                }
                Some(output) => output.to_string(),
            },
        };
        let model = Model {
            ensemble: *ensemble,
            volume: volume.get(args.volume_index, args.phase_index),
            sigma: *sigma,
            kappa_0: kappa_0.get(args.volume_index, args.phase_index),
            kappa_3: *kappa_3,
            weights: *weights,
        };
        Config {
            markov_chain,
            model,
        }
    }
}

impl<T: Copy> Multi<T> {
    fn get(&self, i: usize, j: usize) -> T {
        match self {
            Multi::Scalar(value) => *value,
            Multi::Vector(vec) => vec[i],
            Multi::Matrix(mat) => mat[i][j],
        }
    }
}

#[cfg(test)]
impl Model {
    fn new(ensemble: Ensemble, volume: usize, sigma: f32, kappa_0: f32, kappa_3: f32) -> Self {
        Model {
            ensemble,
            volume,
            sigma,
            kappa_0,
            kappa_3: Some(kappa_3),
            weights: Weights::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Weights {
    shard: usize,
    flip: usize,
    vertex_tree: usize,
    dual_tree: usize,
}

impl Default for Weights {
    fn default() -> Self {
        Weights {
            shard: 1,
            flip: 1,
            vertex_tree: 1,
            dual_tree: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Ensemble {
    Undecorated,
    Spanning,
    TripleTrees,
}
