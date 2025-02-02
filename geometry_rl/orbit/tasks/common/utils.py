import torch
import numpy as np
from scipy.spatial import KDTree

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ClothObject, RigidObject
from pxr import UsdGeom
from omni.isaac.orbit.utils.math import transform_points
from omni.isaac.core.utils.prims import get_prim_at_path
from .sim_utils import assign_subarray_indices, MultiAssetCfg


def find_k_closest_boundary_nodes(points_position, hole, k=10):
    """
    Find k-closest nodes to each hole center using KDTree, for each batch.

    Args:
        points_position (Tensor): Tensor of shape (batch_size, n_points, 3) containing 3D coordinates of nodes.
        holes (list of tuples): List of tuples each containing (x_center, y_center, radius).
        k (int): Number of closest nodes to find.

    Returns:
        dict of dicts: Outer dict keys are batch indices, inner dicts contain tensors of k-closest nodes to each hole center.
    """

    batch_size = points_position.shape[0]
    k_indices_batched = torch.zeros(batch_size, k, dtype=torch.long)

    for batch_index in range(batch_size):
        points_position_np = points_position[batch_index, :, :2].cpu().numpy()  # Convert to numpy for KDTree

        tree = KDTree(points_position_np)
        center = np.array([hole[0], hole[1]])
        distances, indices = tree.query(center, k=k)
        sorted_indices = indices[np.argsort(distances)]  # Sort indices by distances

        k_indices_batched[batch_index] = torch.tensor(sorted_indices)

    return k_indices_batched


def extract_cloth_geometry_positions(cloth_object: ClothObject, indices: torch.Tensor):
    """
    Extract the positions of the nodes of the cloth object using USD API.
    Copied from ClothPrimView.get_world_positions.

    Args:
        cloth_object (ClothObject): The cloth object.
        indices (Tensor): The indices of the cloth object.
    """
    cloth_view = cloth_object.root_view

    positions = cloth_view._backend_utils.create_zeros_tensor(
        [indices.shape[0], cloth_view.max_particles_per_cloth, 3],
        dtype="float32",
        device=cloth_view._device,
    )
    write_idx = 0
    for i in indices:
        cloth_view._apply_cloth_auto_api(i.tolist())
        points = cloth_view._prims[i.tolist()].GetAttribute("points").Get()
        if points is None:
            raise Exception(f"The prim {cloth_view.name} does not have points attribute.")
        positions[write_idx] = cloth_view._backend_utils.create_tensor_from_list(
            points, dtype="float32", device=cloth_view._device
        ).view(cloth_view.max_particles_per_cloth, 3)
        write_idx += 1

    return positions


def get_k_boundary_nodes_in_multi_asset(env, asset_cfg, k=10):
    spawn = env.scene[asset_cfg.name].cfg.spawn

    if hasattr(spawn, "assets_cfg"):
        id_map = spawn.geom_id_map(env.num_envs, len(spawn.assets_cfg))

        holes_boundary_nodes_indices = []

        for i in range(env.num_envs):
            asset = spawn.assets_cfg[id_map[i]]
            holes_boundary_nodes_indices.append(
                get_k_boundary_nodes(
                    env,
                    asset_cfg,
                    asset.size,
                    asset.holes,
                    k=k,
                    indices=torch.tensor([i]),
                )
            )

        holes_boundary_nodes_indices = torch.cat(holes_boundary_nodes_indices, dim=0)
    else:
        holes_boundary_nodes_indices = get_k_boundary_nodes(
            env,
            asset_cfg,
            env.scene[asset_cfg.name].cfg.spawn.size,
            env.scene[asset_cfg.name].cfg.spawn.holes,
        )

    return holes_boundary_nodes_indices


def get_k_boundary_nodes(env, object_cfg, object_size, holes, k=10, indices=None):
    """
    Get the k-closest boundary nodes to each hole in the cloth object.

    Args:
        env (RLTaskEnv): The RLTaskEnv.
        object_cfg (SceneEntityCfg): The object config.
        holes (list of tuples): List of tuples each containing (x_center, y_center, radius).
        k (int): Number of closest nodes to find.

    Returns:
        dict of dicts: Outer dict keys are batch indices, inner dicts contain tensors of k-closest nodes to each hole center.
    """
    if indices is None:
        indices = torch.arange(env.num_envs)

    cloth_geometry_positions = extract_cloth_geometry_positions(env.scene[object_cfg.name], indices)

    num_particles_x, num_particles_y = object_size

    # Compute the spacing between particles in each direction
    spacing_x = 1.0 / (num_particles_x + 1)
    spacing_y = 1.0 / (num_particles_y + 1)

    # For now, we assume same geometry for all cloths
    min_x = cloth_geometry_positions[:, :, 0].min(1).values[0]
    max_x = cloth_geometry_positions[:, :, 0].max(1).values[0]
    min_y = cloth_geometry_positions[:, :, 1].min(1).values[0]
    max_y = cloth_geometry_positions[:, :, 1].max(1).values[0]

    hole_coords = []
    for hole in holes:
        hole_center_x = hole[0]
        hole_center_y = hole[1]
        hole_radius = hole[2]

        hole_center_x = min_x + hole_center_x * spacing_x + spacing_x / 2
        hole_center_y = min_y + hole_center_y * spacing_y + spacing_y / 2
        hole_coords.append([hole_center_x.item(), hole_center_y.item()])

    k_indices_batched = find_k_closest_boundary_nodes(cloth_geometry_positions, hole_coords[0], k=k)

    return k_indices_batched


def get_geometry_from_rigid_object(env, name, return_edges=False, return_num=False):
    """
    Get the geometry positions of the rigid object.

    Args:
        env (RLTaskEnv): The RLTaskEnv.
        name (str): The name of the rigid object.

    Returns:
        torch.Tensor: The geometry positions of the rigid object.
    """
    device = env.device
    rigid_object = env.scene[name]
    prim_paths = rigid_object.root_physx_view.prim_paths
    geom_points = []
    geom_edges = []

    if isinstance(rigid_object.cfg.spawn, MultiAssetCfg):
        indices = assign_subarray_indices(len(prim_paths), len(rigid_object.cfg.spawn.assets_cfg))
        scale = [rigid_object.cfg.spawn.assets_cfg[i].scale for i in indices]
    else:
        scale = [rigid_object.cfg.spawn.scale] * len(prim_paths)

    for i, prim_path in enumerate(prim_paths):
        mesh_prim = sim_utils.get_all_matching_child_prims(prim_path, predicate=lambda pr: pr.GetTypeName() == "Mesh")
        mesh = UsdGeom.Mesh(mesh_prim[0])
        points = torch.tensor(mesh.GetPointsAttr().Get(), device=device)
        face_indices = torch.tensor(mesh.GetFaceVertexIndicesAttr().Get(), device=device)
        edges = extract_edges_from_faces(face_indices).to(device)

        # Scale the points
        points *= torch.tensor(scale[i], device=device)
        geom_points.append(points)
        geom_edges.append(edges)

    # Transform here to make sure the orientation is also correct
    transform = rigid_object.root_physx_view.get_transforms()
    translation = transform[:, :3]
    orientation = transform[:, 3:]  # xyzw -> wxyz

    num_points = [len(points) for points in geom_points]
    num_edges = [edges.shape[1] for edges in geom_edges]

    if isinstance(rigid_object.cfg.spawn, MultiAssetCfg):
        max_num_points = max(num_points)
        max_num_edges = max(num_edges)
        for i in range(len(geom_points)):
            geom_points[i] = transform_points(geom_points[i], translation[i], orientation[i])
            geom_points[i] -= env.scene.env_origins[i].unsqueeze(0)
            geom_points[i] = torch.cat(
                [
                    geom_points[i],
                    torch.zeros(max_num_points - len(geom_points[i]), 3, device=device),
                ]
            )
            geom_edges[i] = torch.cat(
                [
                    geom_edges[i],
                    -torch.ones(2, max_num_edges - geom_edges[i].shape[1], device=device),
                ],
                dim=1,
            )

        geometry_positions = torch.stack(geom_points, dim=0)
        geometry_edges = torch.stack(geom_edges, dim=0)
    else:
        geometry_positions = torch.stack(geom_points, dim=0)
        geometry_positions = transform_points(geometry_positions, translation, orientation)

        # After transformation, the geometry is in the world frame, so we need to subtract the environment origins
        geometry_positions -= env.scene.env_origins.unsqueeze(1)

    if return_num and return_edges:
        return (
            geometry_positions,
            torch.tensor(num_points, device=device).unsqueeze(1),
            geometry_edges.reshape(env.num_envs, -1),
            torch.tensor(num_edges, device=device).unsqueeze(1),
        )
    else:
        return geometry_positions


def extract_distance_matrix_from_two_sets(
    set_a: torch.Tensor,
    set_b: torch.Tensor,
    radius: float = 0.05,
) -> torch.Tensor:

    set_a = set_a.unsqueeze(2)
    set_b = set_b.unsqueeze(1)

    # Compute pairwise squared Euclidean distances
    distances_squared = ((set_a - set_b) ** 2).sum(-1)
    connections_mask = distances_squared < radius**2
    return distances_squared, connections_mask


def extract_boundary_nodes(indices):
    # Define tetrahedron faces
    tetra_faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]

    all_faces = []
    for i in range(0, len(indices), 4):
        tetra = indices[i : i + 4]
        for face in tetra_faces:
            all_faces.append(sorted([tetra[face[0]], tetra[face[1]], tetra[face[2]]]))

    # Find boundary faces
    idx_boundary_faces = []
    boundary_faces = []
    for i, face in enumerate(all_faces):
        if all_faces.count(face) == 1:  # if a face appears only once, it's a boundary face
            boundary_faces.append(face)
            idx_boundary_faces.append(i)

    all_faces = torch.tensor(all_faces).t()
    boundary_faces = torch.tensor(boundary_faces).t()
    idx_boundary_faces = torch.tensor(idx_boundary_faces)

    return boundary_faces.unique()


def extract_edges_from_faces(flattened_indices):
    # Extract unique edges from face vertex indices
    edges = set()

    for i in range(0, len(flattened_indices), 3):
        face = sorted(flattened_indices[i : i + 3])
        face_edges = [
            (face[0], face[1]),
            (face[1], face[0]),
            (face[1], face[2]),
            (face[2], face[1]),
            (face[0], face[2]),
            (face[2], face[0]),
        ]
        edges.update(face_edges)

    return torch.tensor(list(edges)).t()
