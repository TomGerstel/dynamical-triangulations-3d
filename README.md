# Dynamical Triangulations 3D

## Overview

This project contains a simulation of the [triple trees](https://arxiv.org/abs/2203.16105) model, which is a particular version of the dynamical triangulations approach in quantum gravity. The simulation uses Markov chain Monte Carlo methods, and it is written in the Rust programming language.

## Getting Started

In order to be able to run the simulation, you first need to install Rust. Instructions on how this is done can be found on [Rust's official website](https://www.rust-lang.org/). If Rust is installed, the project can be compiled using:

```bash
cargo build --release
```

To run a simulation, you must provide the name of a config file as stored in the `configs` directory. Additionally, two indices are required to index into the config file. Here is an example:

```bash
target/release/dynamical-triangulations-3d -- test 0 1
```
This should run a simulation using the `test.toml` config, indexed by `0` and `1`.

## Config Files

To run a simulation, you need to provide a config file in the `configs` directory. Several config files are provided as examples. Here is the `test.toml` file from the previous example:

```toml
[markov_chain]
thermalisation = [100_000, 200_000]
wait = [100, 200]
amount = 10_000
observables = ["Volume", "VertexCount", "VertexDegree", "EdgeDegree"]

[model]
ensemble = "TripleTrees"
volume = [100, 200]
sigma = 10.0
kappa_0 = [[-1.0, 1.0], [-2.0, 0.0]]

[model.weights]
shard = 1
flip = 1
vertex_tree = 1
dual_tree = 1
```

The `thermalisation` and `wait` parameters are both given in number of sweeps. Here one sweep means taking `N` Markov chain steps, where `N` is the volume of the triangulation. The `wait` value determines the number of steps between measurements, while `amount` determines the number of measurements. The `volume` is the desired number of tetrahedra in the triangulation. However, due to the nature of the simulation this value will fluctuate somewhat around the given value. These fluctuations will have a typical width `sigma`. The strength of the gravitational coupling is given by `kappa_0`. Note that the cosmological constant is automatically tuned to (approximately) its critical value. Finally, we can adjust the frequency of various types of moves by tuning the `weights` values. 

### Indexing Into Config Files

The options `thermalisation`, `wait`, `volume` and `kappa_0` can be provided as either a scalar or a 1D or 2D array. Where relevant, these arrays will be indexed using the indices provided as command-line arguments. This allows for running various simulations with similar settings using a single config file. 
### Observables

The possible `observables` are:
- `Volume`
- `DualGraphShell`
- `DualTreeShell`
- `VertexTreeShell`
- `VertexCount`
- `VertexDegree`
- `EdgeDegree`
- `EdgeFreq`
- `EdgeMiddle`
- `TriangleMiddle`
- `TriangleFreq`
- `EdgeDetour`
- `TriangleDetour`

The `Shell` observables provide a distance profile relative to a randomly sampled point on the corresponding graph. The `Degree` and `Middle` observables provide a histogram of all simplex degrees of a given type. In the case of `Middle` this degree is only considered within the middle tree. The `Freq` observables provide a histogram of duplicate simplex counts. Finally, the `Detour` observables give the distance within the tree between two nodes that are adjacent on the graph.

### Ensembles

The possible `ensemble`s are:
- `Undecorated` (no trees)
- `Spanning` (two spanning trees: vertex and dual)
- `TripleTrees` (three trees)