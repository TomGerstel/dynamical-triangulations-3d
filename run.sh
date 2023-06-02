#!/bin/bash
cargo clean
cargo build --release
sbatch jobs/$1.sh $1