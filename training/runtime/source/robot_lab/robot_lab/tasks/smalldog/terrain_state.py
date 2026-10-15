"""Persist SmallDog terrain assignments and the forward-speed curriculum."""
import torch


def initialize_stratified_terrain(env, max_level):
    terrain = env.scene.terrain
    rows, columns = terrain.terrain_origins.shape[:2]
    count = terrain.terrain_levels.numel()
    if not 0 <= max_level < rows:
        raise ValueError("Requested terrain level is outside the generated terrain grid")
    if count % columns:
        raise ValueError("Stratified terrain assignments require an environment count divisible by columns")
    ids = torch.arange(count, device=env.device)
    terrain.terrain_types.copy_(ids.remainder(columns))
    terrain.terrain_levels.copy_((ids // columns).remainder(max_level + 1))
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    env.episode_length_buf.zero_()
    return terrain.terrain_levels.clone()


def capture_terrain_state(env):
    terrain = env.scene.terrain
    command = env.command_manager.get_term("base_velocity")
    speed = env.curriculum_manager.cfg.forward_speed.func
    return {
        "terrain_shape": tuple(terrain.terrain_origins.shape[:2]),
        "terrain_levels": terrain.terrain_levels.detach().cpu().clone(),
        "terrain_types": terrain.terrain_types.detach().cpu().clone(),
        "forward_speed_range": tuple(command.cfg.ranges.lin_vel_x),
        "forward_speed_counters": {
            name: getattr(speed, name) for name in ("count", "passed", "error_sum", "completed")
        },
    }


def restore_terrain_state(env, state):
    terrain = env.scene.terrain
    levels = state["terrain_levels"].to(device=env.device, dtype=torch.long)
    types = state["terrain_types"].to(device=env.device, dtype=torch.long)
    if tuple(state["terrain_shape"]) != tuple(terrain.terrain_origins.shape[:2]):
        raise ValueError("Saved terrain grid differs from the active terrain grid")
    if levels.shape != terrain.terrain_levels.shape or types.shape != terrain.terrain_types.shape:
        raise ValueError("Saved terrain assignments require the same number of environments")
    rows, columns = state["terrain_shape"]
    if not bool(((levels >= 0) & (levels < rows) & (types >= 0) & (types < columns)).all()):
        raise ValueError("Saved terrain assignments are outside the active terrain grid")
    terrain.terrain_levels.copy_(levels)
    terrain.terrain_types.copy_(types)
    terrain.env_origins[:] = terrain.terrain_origins[levels, types]
    env.command_manager.get_term("base_velocity").cfg.ranges.lin_vel_x = tuple(state["forward_speed_range"])
    speed = env.curriculum_manager.cfg.forward_speed.func
    for name, value in state["forward_speed_counters"].items():
        setattr(speed, name, value)
    # Restart physical episodes without scoring their previous, unrelated poses.
    env.episode_length_buf.zero_()
    env.common_step_counter = 0


def terrain_state_summary(state):
    levels = state["terrain_levels"]
    rows = state["terrain_shape"][0]
    return {
        "mean_level": float(levels.float().mean()),
        "min_level": int(levels.min()),
        "max_level": int(levels.max()),
        "level_histogram": torch.bincount(levels, minlength=rows).tolist(),
        "forward_speed_ceiling_m_s": state["forward_speed_range"][1],
    }
