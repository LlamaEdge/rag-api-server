
# Clippy
cargo +nightly clippy --all-features -- -D warnings

# Format code
cargo fmt

# Find unused dependencies in Cargo.toml
cargo +nightly udeps

# Sort dependencies in Cargo.toml alphabetically
cargo sort
