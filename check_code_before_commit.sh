
# Find unused dependencies in Cargo.toml
cargo +nightly udeps

# Sort dependencies in Cargo.toml alphabetically
cargo sort

# Format code
cargo fmt

# Clippy
cargo +nightly clippy --all-features -- -D warnings
