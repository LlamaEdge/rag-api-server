#!/bin/bash

# Find unused dependencies in Cargo.toml
cargo +nightly udeps

# Sort dependencies in Cargo.toml alphabetically
cargo sort

# Format code
cargo +nightly fmt --all -- --check

# Clippy
cargo +nightly clippy --target wasm32-wasip1 --all-features -- -D warnings
