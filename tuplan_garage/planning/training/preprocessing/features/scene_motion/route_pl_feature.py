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
class RoutePolylineFeature(AbstractModelFeature):
    sdc_id: FeatureDataType
    sdc_route_lane_id: FeatureDataType
    sdc_route_type: FeatureDataType
    sdc_route_xyz: FeatureDataType
    sdc_route_goal: FeatureDataType

    def to_feature_tensor(self) -> RoutePolylineFeature:
        """
        :return object which will be collated into a batch
        """
        return RoutePolylineFeature(
            sdc_id=to_tensor(self.sdc_id),
            sdc_route_lane_id=to_tensor(self.sdc_route_lane_id),
            sdc_route_type=to_tensor(self.sdc_route_type),
            sdc_route_xyz=to_tensor(self.sdc_route_xyz),
            sdc_route_goal=to_tensor(self.sdc_route_goal),
        )

    def to_device(self, device: torch.device) -> AbstractModelFeature:
        """
        :param device: desired device to move feature to
        :return feature type that was moved to a device
        """
        validate_type(self.sdc_id, torch.Tensor)
        validate_type(self.sdc_route_lane_id, torch.Tensor)
        validate_type(self.sdc_route_type, torch.Tensor)
        validate_type(self.sdc_route_xyz, torch.Tensor)
        validate_type(self.sdc_route_goal, torch.Tensor)
        return RoutePolylineFeature(
            sdc_id=self.sdc_id.to(device=device),
            sdc_route_lane_id=self.sdc_route_lane_id.to(device=device),
            sdc_route_type=self.sdc_route_type.to(device=device),
            sdc_route_xyz=self.sdc_route_xyz.to(device=device),
            sdc_route_goal=self.sdc_route_goal.to(device=device),
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
