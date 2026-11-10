# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

COBBLESTONE_ROAD_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)
# TLIBOT_Z1_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
#     size=(8.0, 8.0),
#     border_width=20.0,
#     num_rows=10,
#     num_cols=20,
#     horizontal_scale=0.1,
#     vertical_scale=0.005,
#     slope_threshold=0.75,
#     difficulty_range=(0.0, 1.0),
#     use_cache=False,
#     sub_terrains={
#         # easy start: perfectly flat patches
#         "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
#         # random bumps -- the main "rough road" case
#         "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
#             proportion=0.3, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
#         ),
#         "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
#             proportion=0.15,
#             step_height_range=(0.03, 0.12),
#             step_width=0.3,
#             platform_width=2.0,
#             border_width=1.0,
#             holes=False,
#         ),
#         "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
#             proportion=0.15,
#             step_height_range=(0.03, 0.12),
#             step_width=0.3,
#             platform_width=2.0,
#             border_width=1.0,
#             holes=False,
#         ),
#         "boxes": terrain_gen.MeshRandomGridTerrainCfg(
#             proportion=0.2, grid_width=0.35, grid_height_range=(0.02, 0.10), platform_width=2.0
#         ),
#         "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
#             proportion=0.1, slope_range=(0.0, 0.25), platform_width=2.0, border_width=0.25
#         ),
#     },
# )
TLIBOT_Z1_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # "flat": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.50,
        #     noise_range=(0.0, 0.0),
        #     noise_step=0.005,
        #     downsampled_scale=0.2,
        #     border_width=0.25,
        # ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(-0.02, 0.02),
            noise_step=0.005,
            downsampled_scale=0.2,
            border_width=0.25,
        ),
    },
)