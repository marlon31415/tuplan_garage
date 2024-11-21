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
class MapPolylineFeature(AbstractModelFeature):
    mf_id: FeatureDataType
    mf_xyz: FeatureDataType
    mf_type: FeatureDataType
    mf_edge: FeatureDataType

    def to_feature_tensor(self) -> MapPolylineFeature:
        """
        :return object which will be collated into a batch
        """
        return MapPolylineFeature(
            mf_id=to_tensor(self.mf_id),
            mf_xyz=to_tensor(self.mf_xyz),
            mf_type=to_tensor(self.mf_type),
            mf_edge=to_tensor(self.mf_edge),
        )

    def to_device(self, device: torch.device) -> AbstractModelFeature:
        """
        :param device: desired device to move feature to
        :return feature type that was moved to a device
        """
        validate_type(self.mf_id, torch.Tensor)
        validate_type(self.mf_xyz, torch.Tensor)
        validate_type(self.mf_type, torch.Tensor)
        validate_type(self.mf_edge, torch.Tensor)
        return MapPolylineFeature(
            mf_id=self.mf_id.to(device=device),
            mf_xyz=self.mf_xyz.to(device=device),
            mf_type=self.mf_type.to(device=device),
            mf_edge=self.mf_edge.to(device=device),
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
