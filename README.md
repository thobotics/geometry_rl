# IronLib

IronLib is a powerful framework that aims to "iron everything" by flattening all dependencies through a patching pattern. This means that it provides a seamless and efficient way to manage and integrate various software components, making development and deployment smoother and more efficient.

## Getting Started

To get started with IronLib, follow the steps below:

0. Preliminaries:
    Your workspace structure should look like this
    ```
    ├── ...
    ├── repos                  # Root workspace
    │   ├── ironlib            # ironlib repo
    │   ├── orbit              # orbit repo
    │   ├── others             # other repos
    └── ...
    ```
    Note: this structure is hard-coded as `docker-compose` will build an image with context points to the Root workspace.

1. Setup Docker:
    ```
    ./docker/container.sh start
    ```

2. Inside container (entering via `./docker/container.sh enter`):
    ```bash
    orbit -p examples/torchrl/train.py -cn ant_ppo_cfg simulator.headless=True
    ```

    ```bash
    orbit -p examples/torchrl/play.py -cn ant_ppo_cfg
    ```

3. Run the rsl_rl example with orbit:
    ```bash
    orbit -p examples/orbit/standalone/workflows/rsl_rl/train.py --task Isaac-Ant-v0 --headless
    ```

## Documentation

For detailed documentation and usage instructions, please refer to the [IronLib Documentation](https://ironlib-docs.com).

## Contributing

We welcome contributions from the community! If you have any bug reports, feature requests, or would like to contribute code, please refer to our [Contribution Guidelines](https://ironlib-docs.com/contributing).

## License

IronLib is licensed under the [MIT License](https://opensource.org/licenses/MIT).
