import numpy as np
import torch
from copy import deepcopy
from scipy.interpolate import interp1d
from collections.abc import Mapping
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory

# The model checkpoints contain the paths to the different modules within the future-motion repo.
# Therefore, we override these paths with those of the tuplan_garage repo.
MODEL_CONFIG_OVERRIDE = {
    "model": {
        "_target_": "tuplan_garage.planning.external_submodules.future_motion.src.models.ac_scene_motion.SceneMotion",
        "motion_decoder": {
            "_target_": "tuplan_garage.planning.external_submodules.future_motion.src.models.ac_wayformer.Decoder"
        },
    },
    "pre_processing": {
        "agent_centric": {
            "_target_": "tuplan_garage.planning.external_submodules.future_motion.src.external_submodules.hptr.src.data_modules.agent_centric.AgentCentricPreProcessing"
        },
        "ac_global": {
            "_target_": "tuplan_garage.planning.external_submodules.future_motion.src.data.ac_scene_motion.AgentCentricSceneMotion"
        },
    },
    "train_metric": {
        "_target_": "tuplan_garage.planning.external_submodules.future_motion.src.models.metrics.planning.EgoPlanningMetrics"
    },
    "waymo_metric": {
        "_target_": "tuplan_garage.planning.external_submodules.future_motion.src.external_submodules.hptr.src.models.metrics.waymo.WaymoMetrics"
    },
    "waymo_ego_metric": {
        "_target_": "tuplan_garage.planning.external_submodules.future_motion.src.models.metrics.waymo_ego.WaymoEgoMetrics"
    },
    "post_processing": {
        "to_dict": {
            "_target_": "tuplan_garage.planning.external_submodules.future_motion.src.external_submodules.hptr.src.data_modules.post_processing.ToDict"
        },
        "get_cov_mat": {
            "_target_": "tuplan_garage.planning.external_submodules.future_motion.src.external_submodules.hptr.src.data_modules.post_processing.GetCovMat"
        },
        "waymo": {
            "_target_": "tuplan_garage.planning.external_submodules.future_motion.src.external_submodules.hptr.src.data_modules.waymo_post_processing.WaymoPostProcessing"
        },
    },
    "sub_womd": {
        "_target_": "tuplan_garage.planning.external_submodules.future_motion.src.external_submodules.hptr.src.utils.submission.SubWOMD"
    },
    "sub_av2": {
        "_target_": "tuplan_garage.planning.external_submodules.future_motion.src.external_submodules.hptr.src.utils.submission.SubAV2"
    },
}


def convert_predictions_to_trajectory(
    prediction_dict: dict, resample: bool
) -> torch.Tensor:
    """
    Convert predictions tensor to Trajectory.data shape
    :param prediction_dict: dict with model output
    :return: data suitable for Trajectory
    """
    # conf: [n_scene, n_agent, k_pred]
    conf = prediction_dict["waymo_scores"]
    # trajs: [n_scene, n_step, n_agent, k_pred, 2]
    trajs = prediction_dict["waymo_trajs"]
    # yaw: [n_scene, n_step, n_agent, k_pred, 1]
    yaw = prediction_dict["waymo_yaw_bbox"]
    # pred_idx: [n_scene, n_target] (mapping from pred agents to all agents)
    pred_idx = prediction_dict["ref_idx"]
    # role: [n_scene, n_target, 3]
    role = prediction_dict["ref_role"]

    ego_traj = get_ego_traj_from_trajs(trajs, conf, pred_idx, role)
    if resample:
        ego_traj = resample_traj(ego_traj, 10)

    if yaw != None:
        ego_traj_with_yaw = torch.cat((ego_traj, yaw), dim=-1)
    else:
        yaw = polynomial_yaw_interpolation_from_points(ego_traj)
        ego_traj_with_yaw = torch.cat((ego_traj, yaw.unsqueeze(-1)), dim=-1)

    num_batches = trajs.shape[0]
    return ego_traj_with_yaw.view(num_batches, -1, Trajectory.state_size())


def get_ego_traj_from_trajs(trajs, conf, pred_idx, role):
    n_scene = trajs.shape[0]

    ego_idx = pred_idx[role[..., 0]]

    best_ego_pred_list = []
    for i in range(n_scene):
        ego_scene_idx = ego_idx[i]
        best_ego_pred_idx = torch.argmax(conf[i, ego_scene_idx], dim=-1)
        best_ego_pred_list.append(trajs[i, :, ego_scene_idx, best_ego_pred_idx, :])
    best_ego_pred = torch.stack(best_ego_pred_list, dim=0)

    return best_ego_pred


def resample_traj(traj, resample_step_size):
    batch_size, n_points, _ = traj.shape
    traj_resampled = torch.zeros_like(traj)
    traj = traj.numpy()

    t = np.linspace(0, n_points - 1, n_points)
    downsampled_t = t[::resample_step_size]
    downsampled_t = np.append(downsampled_t, t[-1])

    for b in range(batch_size):
        points_b = traj[b]
        x, y = points_b[:, 0], points_b[:, 1]

        downsampled_x = x[::resample_step_size]
        downsampled_x = np.append(downsampled_x, x[-1])
        downsampled_y = y[::resample_step_size]
        downsampled_y = np.append(downsampled_y, y[-1])

        interp_x = interp1d(downsampled_t, downsampled_x, kind="cubic")
        interp_y = interp1d(downsampled_t, downsampled_y, kind="cubic")

        resampled_x = interp_x(t)  # Interpolated x values
        resampled_y = interp_y(t)  # Interpolated y values

        traj_resampled[b] = torch.stack(
            (torch.tensor(resampled_x), torch.tensor(resampled_y)), dim=-1
        )

    return traj_resampled


def polynomial_yaw_interpolation_from_points(
    points: torch.Tensor, window_size: int = 5, poly_order: int = 2
) -> torch.Tensor:
    """
    Calculate precise yaw angles for batched trajectories using a local polynomial fit.

    Args:
        points: Tensor of shape (batch, N, 2/3), where each point is (x, y).
        window_size: Number of points to consider in the local fit (must be odd).
        poly_order: The order of the polynomial for fitting.

    Returns:
        yaw_angles: Tensor of shape (batch, N) with yaw angles at each point in radians.
    """
    assert window_size % 2 == 1, "Window size must be odd for symmetric fitting."
    assert points.shape[2] == 2, "Trajectories should have shape (batch, N, 2)."

    batch_size, n, _ = points.shape
    yaw_angles = torch.zeros((batch_size, n), dtype=torch.float32)
    half_window = window_size // 2

    for b in range(batch_size):
        # Extract the trajectory for the current batch
        points_b = points[b].numpy()
        x, y = points_b[:, 0], points_b[:, 1]

        for i in range(n):
            # Determine the window bounds
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)

            # Get the local points in the window
            local_x = x[start:end]
            local_y = y[start:end]

            # Fit a polynomial to the local points
            if len(local_x) > 1:  # Ensure there's enough points to fit
                coeffs = np.polyfit(local_x, local_y, poly_order)  # Fit y = f(x)
                # Compute the derivative at x[i]
                deriv = np.polyder(coeffs)
                slope = np.polyval(deriv, x[i])
                yaw_angles[b, i] = torch.atan(torch.tensor(slope, dtype=torch.float32))
            else:
                yaw_angles[b, i] = 0  # Default to 0 if not enough points (edge case)

    return yaw_angles


def deep_merge_dicts(base_config, override_config):
    """
    Recursively merges the `override_config` dictionary into `base_config`.
    Preserves keys in `base_config` that are not present in `override_config`.
    If a key exists in both and its value is a dictionary, it merges them recursively.
    Otherwise, it overwrites the value in `base_config` with the value from `override_config`.
    """
    for key, value in override_config.items():
        if (
            isinstance(value, Mapping)
            and key in base_config
            and isinstance(base_config[key], Mapping)
        ):
            # Recursively merge dictionaries
            deep_merge_dicts(base_config[key], value)
        else:
            # Overwrite or add key-value pairs
            base_config[key] = deepcopy(value)
