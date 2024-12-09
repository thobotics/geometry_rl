import random
import re
import traceback
from typing import Callable

from dataclasses import MISSING

import omni.isaac.core.utils.prims as prim_utils
import omni.usd
from pxr import Gf, Sdf, Semantics, Usd, UsdGeom, UsdUtils, Vt
from omni.physx import get_physx_replicator_interface

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.cloner import Cloner
from omni.isaac.orbit.assets import AssetBaseCfg, RigidObjectCfg
from omni.isaac.orbit.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.orbit.sim import SimulationContext
from omni.isaac.orbit.utils import Timer, configclass


def assign_subarray_indices(n, m):
    # Calculate the base size and the number of elements in larger subarrays
    base_size = n // m
    extras = n % m

    indices = []
    for i in range(m):
        # If there are extras, add one more to the current subarray
        size = base_size + (1 if i < extras else 0)
        # Extend the indices list with `size` number of current index `i`
        indices.extend([i] * size)

    return indices


def spawn_fixed_number_of_multi_object_sdf(
    prim_path: str,
    cfg: "MultiAssetCfg",
    translation: list[tuple[float, float, float]] | None = None,
    orientation: list[tuple[float, float, float, float]] | None = None,
    replicate_physics: bool = False,
) -> Usd.Prim:
    # resolve: {SPAWN_NS}/AssetName
    # note: this assumes that the spawn namespace already exists in the stage
    root_path, asset_path = prim_path.rsplit("/", 1)
    # check if input is a regex expression
    # note: a valid prim path can only contain alphanumeric characters, underscores, and forward slashes
    is_regex_expression = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None

    # resolve matching prims for source prim path expression
    if is_regex_expression and root_path != "":
        source_prim_paths = sim_utils.find_matching_prim_paths(root_path)
        # if no matching prims are found, raise an error
        if len(source_prim_paths) == 0:
            raise RuntimeError(
                f"Unable to find source prim path: '{root_path}'. Please create the prim before spawning."
            )
    else:
        source_prim_paths = [root_path]

    # spawn everything first in a "Dataset" prim
    prim_utils.create_prim("/World/Dataset", "Scope")
    proto_prim_paths = list()
    for index, asset_cfg in enumerate(cfg.assets_cfg):
        # spawn single instance
        proto_prim_path = f"/World/Dataset/Object_{index:02d}"
        prim = asset_cfg.func(proto_prim_path, asset_cfg, translation[index], orientation[index])
        # save the proto prim path
        proto_prim_paths.append(proto_prim_path)
        # set the prim visibility
        if hasattr(asset_cfg, "visible"):
            imageable = UsdGeom.Imageable(prim)
            if asset_cfg.visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
        # set the semantic annotations
        if hasattr(asset_cfg, "semantic_tags") and asset_cfg.semantic_tags is not None:
            # note: taken from replicator scripts.utils.utils.py
            for semantic_type, semantic_value in asset_cfg.semantic_tags:
                # deal with spaces by replacing them with underscores
                semantic_type_sanitized = semantic_type.replace(" ", "_")
                semantic_value_sanitized = semantic_value.replace(" ", "_")
                # set the semantic API for the instance
                instance_name = f"{semantic_type_sanitized}_{semantic_value_sanitized}"
                sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
                # create semantic type and data attributes
                sem.CreateSemanticTypeAttr()
                sem.CreateSemanticDataAttr()
                sem.GetSemanticTypeAttr().Set(semantic_type)
                sem.GetSemanticDataAttr().Set(semantic_value)
        # activate rigid body contact sensors
        if hasattr(asset_cfg, "activate_contact_sensors") and asset_cfg.activate_contact_sensors:
            sim_utils.activate_contact_sensors(proto_prim_path, asset_cfg.activate_contact_sensors)

    # resolve prim paths for spawning and cloning
    prim_paths = [f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths]

    indices = assign_subarray_indices(len(source_prim_paths), len(cfg.assets_cfg))

    # acquire stage
    stage = omni.usd.get_context().get_stage()
    # manually clone prims if the source prim path is a regex expression
    with Sdf.ChangeBlock():
        for i, prim_path in enumerate(prim_paths):
            # spawn single instance
            env_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
            proto_path = proto_prim_paths[indices[i]]
            # inherit the proto prim
            # env_spec.inheritPathList.Prepend(Sdf.Path(proto_path))
            Sdf.CopySpec(
                env_spec.layer,
                Sdf.Path(proto_path),
                env_spec.layer,
                Sdf.Path(prim_path),
            )
            # set the translation and orientation
            _ = UsdGeom.Xform(stage.GetPrimAtPath(proto_path)).GetPrim().GetPrimStack()

            translate_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOp:translate")
            if translate_spec is None:
                translate_spec = Sdf.AttributeSpec(env_spec, "xformOp:translate", Sdf.ValueTypeNames.Double3)
            translate_spec.default = Gf.Vec3d(*translation[indices[i]])

            orient_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOp:orient")
            if orient_spec is None:
                orient_spec = Sdf.AttributeSpec(env_spec, "xformOp:orient", Sdf.ValueTypeNames.Quatd)
            # convert orientation ordering (wxyz to xyzw)
            orientation_i = orientation[indices[i]]
            orientation_i = (
                orientation_i[1],
                orientation_i[2],
                orientation_i[3],
                orientation_i[0],
            )
            orient_spec.default = Gf.Quatd(*orientation_i)

            scale_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            if scale_spec is None:
                scale_spec = Sdf.AttributeSpec(env_spec, "xformOp:scale", Sdf.ValueTypeNames.Double3)

            op_order_spec = env_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
            if op_order_spec is None:
                op_order_spec = Sdf.AttributeSpec(env_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray)
            op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])

    # delete the dataset prim after spawning
    prim_utils.delete_prim("/World/Dataset")

    # return the prim
    return prim_utils.get_prim_at_path(prim_paths[0])


def spawn_multi_object_randomly(
    prim_path: str,
    cfg: "MultiAssetCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    # resolve: {SPAWN_NS}/AssetName
    # note: this assumes that the spawn namespace already exists in the stage
    root_path, asset_path = prim_path.rsplit("/", 1)
    # check if input is a regex expression
    # note: a valid prim path can only contain alphanumeric characters, underscores, and forward slashes
    is_regex_expression = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None

    # resolve matching prims for source prim path expression
    if is_regex_expression and root_path != "":
        source_prim_paths = sim_utils.find_matching_prim_paths(root_path)
        # if no matching prims are found, raise an error
        if len(source_prim_paths) == 0:
            raise RuntimeError(
                f"Unable to find source prim path: '{root_path}'. Please create the prim before spawning."
            )
    else:
        source_prim_paths = [root_path]

    # resolve prim paths for spawning and cloning
    prim_paths = [f"{source_prim_path}/{asset_path}" for source_prim_path in source_prim_paths]
    # manually clone prims if the source prim path is a regex expression
    for prim_path in prim_paths:
        # randomly select an asset configuration
        asset_cfg = random.choice(cfg.assets_cfg)
        # spawn single instance
        prim = asset_cfg.func(prim_path, asset_cfg, translation, orientation)
        # set the prim visibility
        if hasattr(asset_cfg, "visible"):
            imageable = UsdGeom.Imageable(prim)
            if asset_cfg.visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
        # set the semantic annotations
        if hasattr(asset_cfg, "semantic_tags") and asset_cfg.semantic_tags is not None:
            # note: taken from replicator scripts.utils.utils.py
            for semantic_type, semantic_value in asset_cfg.semantic_tags:
                # deal with spaces by replacing them with underscores
                semantic_type_sanitized = semantic_type.replace(" ", "_")
                semantic_value_sanitized = semantic_value.replace(" ", "_")
                # set the semantic API for the instance
                instance_name = f"{semantic_type_sanitized}_{semantic_value_sanitized}"
                sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
                # create semantic type and data attributes
                sem.CreateSemanticTypeAttr()
                sem.CreateSemanticDataAttr()
                sem.GetSemanticTypeAttr().Set(semantic_type)
                sem.GetSemanticDataAttr().Set(semantic_value)
        # activate rigid body contact sensors
        if hasattr(asset_cfg, "activate_contact_sensors") and asset_cfg.activate_contact_sensors:
            sim_utils.activate_contact_sensors(prim_path, asset_cfg.activate_contact_sensors)

    # return the prim
    return prim


@configclass
class MultiAssetCfg(sim_utils.SpawnerCfg):
    """Configuration parameters for loading multiple assets randomly."""

    # Uncomment this one: 45 seconds for 2048 envs
    # func: sim_utils.SpawnerCfg.func = spawn_multi_object_randomly
    # Uncomment this one: 2.15 seconds for 2048 envs
    func: Callable[..., Usd.Prim] = spawn_fixed_number_of_multi_object_sdf

    geom_id_map: Callable = assign_subarray_indices

    assets_cfg: list[sim_utils.SpawnerCfg] = MISSING
    """List of asset configurations to spawn."""
