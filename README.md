# GeometryRL

## Getting Started

To get started with GeometryRL, follow the steps below:

0. Preliminaries:
    The default workspace structure looks like this
    ```
    ├── ...
    ├── repos                  # Root workspace
    │   ├── geometry_rl        # geometry_rl repo
    │   ├── orbit              # orbit repo
    │   ├── others             # other repos
    └── ...
    ```
    Note: you can change your workspace structure by modifying `docker/.env`, then `docker-compose` will build an image with context points to the Root workspace.

1. Setup Docker:
    ```
    ./docker/container.sh start
    ```

2. Inside container (entering via `./docker/container.sh enter`):
    ```bash
    orbit -p examples/torchrl/train.py -cn rigid_insertion_multi_hepi_trpl_cfg simulator.headless=True
    ```

    ```bash
    orbit -p examples/torchrl/play.py -cn rigid_insertion_multi_hepi_trpl_cfg
    ```

3. Run the rsl_rl example with orbit:
    ```bash
    orbit -p examples/orbit/standalone/workflows/rsl_rl/train.py --task Isaac-Ant-v0 --headless
    ```

## License

GeometryRL is licensed under the [MIT License](https://opensource.org/licenses/MIT).
