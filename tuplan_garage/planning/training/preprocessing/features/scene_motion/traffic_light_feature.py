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
class TrafficLightFeature(AbstractModelFeature):
    tl_lane_state: FeatureDataType
    tl_lane_id: FeatureDataType
    tl_stop_point: FeatureDataType

    def to_feature_tensor(self) -> TrafficLightFeature:
        """
        :return object which will be collated into a batch
        """
        return TrafficLightFeature(
            tl_lane_state=to_tensor(self.tl_lane_state),
            tl_lane_id=to_tensor(self.tl_lane_id),
            tl_stop_point=to_tensor(self.tl_stop_point),
        )

    def to_device(self, device: torch.device) -> AbstractModelFeature:
        """
        :param device: desired device to move feature to
        :return feature type that was moved to a device
        """
        validate_type(self.tl_lane_state, torch.Tensor)
        validate_type(self.tl_lane_id, torch.Tensor)
        validate_type(self.tl_stop_point, torch.Tensor)
        return TrafficLightFeature(
            tl_lane_state=self.tl_lane_state.to(device=device),
            tl_lane_id=self.tl_lane_id.to(device=device),
            tl_stop_point=self.tl_stop_point.to(device=device),
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
