import wandb
from typing import List
from omegaconf import DictConfig
import torch
from copy import deepcopy

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
)
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import (
    AbstractTargetBuilder,
)
from tuplan_garage.planning.training.preprocessing.features.scene_motion.scene_motion_features import (
    SceneMotionFeatures,
)
from tuplan_garage.planning.external_submodules.future_motion.src.future_motion import (
    FutureMotion,
)
from tuplan_garage.planning.training.modeling.models.scene_motion.utils import (
    MODEL_CONFIG_OVERRIDE,
    convert_predictions_to_trajectory,
    deep_merge_dicts,
)


class SceneMotionModel(TorchModuleWrapper):
    def __init__(
        self,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        future_trajectory_sampling: TrajectorySampling,
        checkpoint: str,
        # model_config_override: DictConfig,
    ):
        """
        :param feature_builders: list of builders for features
        :param target_builders: list of builders for targets
        :param future_trajectory_sampling: parameters of predicted trajectory
        """
        super().__init__(
            feature_builders=feature_builders,
            target_builders=target_builders,
            future_trajectory_sampling=future_trajectory_sampling,
        )

        self.checkpoint = checkpoint
        self.model_config_override = MODEL_CONFIG_OVERRIDE

        # Load scene-motion model
        self.scene_motion_model = self._load_model_from_ckpt(
            self.checkpoint, self.model_config_override
        )
        self.scene_motion_model.eval()

    def forward(self, features: SceneMotionFeatures) -> TargetsType:
        """ """
        batch = self._features_to_model_input(features)

        pred_dict = self.scene_motion_model(batch)

        return {
            "trajectory": Trajectory(
                data=convert_predictions_to_trajectory(pred_dict, resample=False)
            )
        }

    def _load_model_from_ckpt(self, checkpoint: str, config: DictConfig):
        if "/" in self.checkpoint:
            print("Loading checkpoint from path")
            ckpt_path = checkpoint
        else:
            print("Downloading checkpoint from wandb")
            run = wandb.init(project="nuplan", entity="kit_mrt")
            artifact = run.use_artifact(checkpoint, type="model")
            artifact_path = artifact.download(
                f"/home/wiss/steiner/projects/tuplan_garage/ckpt/{checkpoint}"
            )
            ckpt_path = f"{artifact_path}/model.ckpt"

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        merged_hparams = deepcopy(checkpoint["hyper_parameters"])
        deep_merge_dicts(merged_hparams, config)

        return FutureMotion.load_from_checkpoint(
            ckpt_path,
            **merged_hparams,
            strict=False,
        )

    def _features_to_model_input(
        self, features: dict[str, dict[str, SceneMotionFeatures]]
    ):
        scene_motion_features = features["scene_motion_features"]
        return {
            "history/agent/valid": scene_motion_features["history_agent_valid"],
            "history/agent/pos": scene_motion_features["history_agent_pos"],
            "history/agent/vel": scene_motion_features["history_agent_vel"],
            "history/agent/spd": scene_motion_features["history_agent_spd"],
            "history/agent/acc": scene_motion_features["history_agent_acc"],
            "history/agent/yaw_bbox": scene_motion_features["history_agent_yaw_bbox"],
            "history/agent/yaw_rate": scene_motion_features["history_agent_yaw_rate"],
            "history/agent/type": scene_motion_features["history_agent_type"],
            "history/agent/role": scene_motion_features["history_agent_role"],
            "history/agent/size": scene_motion_features["history_agent_size"],
            "history/tl_stop/valid": scene_motion_features["history_tl_stop_valid"],
            "history/tl_stop/state": scene_motion_features["history_tl_stop_state"],
            "history/tl_stop/pos": scene_motion_features["history_tl_stop_pos"],
            "history/tl_stop/dir": scene_motion_features["history_tl_stop_dir"],
            "map/valid": scene_motion_features["map_valid"],
            "map/type": scene_motion_features["map_type"],
            "map/pos": scene_motion_features["map_pos"],
            "map/dir": scene_motion_features["map_dir"],
            "route/valid": scene_motion_features["route_valid"],
            "route/type": scene_motion_features["route_type"],
            "route/pos": scene_motion_features["route_pos"],
            "route/dir": scene_motion_features["route_dir"],
        }
