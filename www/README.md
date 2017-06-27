# Source code of the vulkano website

To run the website, just do:

```rust
ADDR=0.0.0.0:8000 cargo run
```

The `deploy.pub` and `deploy.enc` files are the public and private keys used by travis to deploy
the website.

Note that this subdirectory is not part of the root workspace so that we can commit a `Cargo.lock`.
