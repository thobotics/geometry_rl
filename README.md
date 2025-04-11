# Geometry-aware **RL** for Manipulation of Varying Shapes and Deformable Objects
⭐ **Accepted at [ICLR 2025 (Oral)]((https://openreview.net/forum?id=7BLXhmWvwF))** ⭐

## 🚀 Getting Started

To get started with ***GeometryRL***, follow the steps below:

0. 🛠️ **Preliminaries**:  
    The default workspace structure looks like this:
    ```
    ├── ...
    ├── repos                  # Root workspace
    │   ├── geometry_rl        # geometry_rl repo
    │   ├── geometry_orbit     # geometry_orbit repo
    │   ├── others             # other repos
    └── ...
    ```
    **Note**: You can change your workspace structure by modifying `docker/.env`. Then, `docker-compose` will build an image with the context pointing to the Root workspace.

1. 📂 **Clone the required repos**:  
    Clone `geometry_orbit` and `ITPAL`:
    ```
    git clone git@github.com:thobotics/geometry_orbit.git
    ```
    ```
    git clone git@github.com:ALRhub/ITPAL.git
    ```
    Place them into the correct workspace structure.

2. 🐳 **Setup Docker**:  
    Start the Docker container:
    ```
    ./docker/container.sh start
    ```

3. 🏗️ **Inside the container** (enter via `./docker/container.sh enter`):  
    Run the following commands:
    ```bash
    orbit -p examples/torchrl/train.py -cn rigid_insertion_multi_hepi_trpl_cfg simulator.headless=True
    ```

    ```bash
    orbit -p examples/torchrl/play.py -cn rigid_insertion_multi_hepi_trpl_cfg --checkpoint_name model_checkpoint_best.pth
    ```

## 📖 Citation

```
@inproceedings{
     hoang2025geometryaware,
     title={Geometry-aware {RL} for Manipulation of Varying Shapes and Deformable Objects},
     author={Tai Hoang and Huy Le and Philipp Becker and Vien Anh Ngo and Gerhard Neumann},
     booktitle={The Thirteenth International Conference on Learning Representations},
     year={2025},
     url={https://openreview.net/forum?id=7BLXhmWvwF}
}
```