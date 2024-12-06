from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import torch

from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    FeatureDataType,
    to_tensor,
)


@dataclass
class SceneMotionFeatures(AbstractModelFeature):
    history_agent_valid: FeatureDataType
    history_agent_pos: FeatureDataType
    history_agent_vel: FeatureDataType
    history_agent_spd: FeatureDataType
    history_agent_acc: FeatureDataType
    history_agent_yaw_bbox: FeatureDataType
    history_agent_yaw_rate: FeatureDataType
    history_agent_type: FeatureDataType
    history_agent_role: FeatureDataType
    history_agent_size: FeatureDataType
    history_tl_stop_valid: FeatureDataType
    history_tl_stop_state: FeatureDataType
    history_tl_stop_pos: FeatureDataType
    history_tl_stop_dir: FeatureDataType
    map_valid: FeatureDataType
    map_type: FeatureDataType
    map_pos: FeatureDataType
    map_dir: FeatureDataType
    route_valid: FeatureDataType
    route_type: FeatureDataType
    route_pos: FeatureDataType
    route_dir: FeatureDataType
    route_goal: FeatureDataType

    def to_feature_tensor(self) -> SceneMotionFeatures:
        """
        :return object which will be collated into a batch
        """
        return SceneMotionFeatures(
            history_agent_valid=to_tensor(self.history_agent_valid),
            history_agent_pos=to_tensor(self.history_agent_pos),
            history_agent_vel=to_tensor(self.history_agent_vel),
            history_agent_spd=to_tensor(self.history_agent_spd),
            history_agent_acc=to_tensor(self.history_agent_acc),
            history_agent_yaw_bbox=to_tensor(self.history_agent_yaw_bbox),
            history_agent_yaw_rate=to_tensor(self.history_agent_yaw_rate),
            history_agent_type=to_tensor(self.history_agent_type),
            history_agent_role=to_tensor(self.history_agent_role),
            history_agent_size=to_tensor(self.history_agent_size),
            history_tl_stop_valid=to_tensor(self.history_tl_stop_valid),
            history_tl_stop_state=to_tensor(self.history_tl_stop_state),
            history_tl_stop_pos=to_tensor(self.history_tl_stop_pos),
            history_tl_stop_dir=to_tensor(self.history_tl_stop_dir),
            map_valid=to_tensor(self.map_valid),
            map_type=to_tensor(self.map_type),
            map_pos=to_tensor(self.map_pos),
            map_dir=to_tensor(self.map_dir),
            route_valid=to_tensor(self.route_valid),
            route_type=to_tensor(self.route_type),
            route_pos=to_tensor(self.route_pos),
            route_dir=to_tensor(self.route_dir),
            route_goal=to_tensor(self.route_goal),
        )

    def to_device(self, device: torch.device) -> SceneMotionFeatures:
        """
        :param device: desired device to move feature to
        :return feature type that was moved to a device
        """
        validate_type(self.history_agent_valid, torch.Tensor)
        validate_type(self.history_agent_pos, torch.Tensor)
        validate_type(self.history_agent_vel, torch.Tensor)
        validate_type(self.history_agent_spd, torch.Tensor)
        validate_type(self.history_agent_acc, torch.Tensor)
        validate_type(self.history_agent_yaw_bbox, torch.Tensor)
        validate_type(self.history_agent_yaw_rate, torch.Tensor)
        validate_type(self.history_agent_type, torch.Tensor)
        validate_type(self.history_agent_role, torch.Tensor)
        validate_type(self.history_agent_size, torch.Tensor)
        validate_type(self.history_tl_stop_valid, torch.Tensor)
        validate_type(self.history_tl_stop_state, torch.Tensor)
        validate_type(self.history_tl_stop_pos, torch.Tensor)
        validate_type(self.history_tl_stop_dir, torch.Tensor)
        validate_type(self.map_valid, torch.Tensor)
        validate_type(self.map_type, torch.Tensor)
        validate_type(self.map_pos, torch.Tensor)
        validate_type(self.map_dir, torch.Tensor)
        validate_type(self.route_valid, torch.Tensor)
        validate_type(self.route_type, torch.Tensor)
        validate_type(self.route_pos, torch.Tensor)
        validate_type(self.route_dir, torch.Tensor)
        validate_type(self.route_goal, torch.Tensor)
        return SceneMotionFeatures(
            history_agent_valid=self.history_agent_valid.to(device=device),
            history_agent_pos=self.history_agent_pos.to(device=device),
            history_agent_vel=self.history_agent_vel.to(device=device),
            history_agent_spd=self.history_agent_spd.to(device=device),
            history_agent_acc=self.history_agent_acc.to(device=device),
            history_agent_yaw_bbox=self.history_agent_yaw_bbox.to(device=device),
            history_agent_yaw_rate=self.history_agent_yaw_rate.to(device=device),
            history_agent_type=self.history_agent_type.to(device=device),
            history_agent_role=self.history_agent_role.to(device=device),
            history_agent_size=self.history_agent_size.to(device=device),
            history_tl_stop_valid=self.history_tl_stop_valid.to(device=device),
            history_tl_stop_state=self.history_tl_stop_state.to(device=device),
            history_tl_stop_pos=self.history_tl_stop_pos.to(device=device),
            history_tl_stop_dir=self.history_tl_stop_dir.to(device=device),
            map_valid=self.map_valid.to(device=device),
            map_type=self.map_type.to(device=device),
            map_pos=self.map_pos.to(device=device),
            map_dir=self.map_dir.to(device=device),
            route_valid=self.route_valid.to(device=device),
            route_type=self.route_type.to(device=device),
            route_pos=self.route_pos.to(device=device),
            route_dir=self.route_dir.to(device=device),
            route_goal=self.route_goal.to(device=device),
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> AbstractModelFeature:
        """
        :return: Return dictionary of data that can be serialized
        """
        return data

    def unpack(self) -> List[AbstractModelFeature]:
        """
        :return: Unpack a batched feature to a list of features.
        """
        pass
