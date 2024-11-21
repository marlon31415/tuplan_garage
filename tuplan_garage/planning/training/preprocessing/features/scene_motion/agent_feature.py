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
class AgentFeature(AbstractModelFeature):
    agent_id: FeatureDataType
    agent_type: FeatureDataType
    agent_states: FeatureDataType
    agent_role: FeatureDataType

    def to_feature_tensor(self) -> AgentFeature:
        """
        :return object which will be collated into a batch
        """
        return AgentFeature(
            agent_id=to_tensor(self.agent_id),
            agent_type=to_tensor(self.agent_type),
            agent_states=to_tensor(self.agent_states),
            agent_role=to_tensor(self.agent_role),
        )

    def to_device(self, device: torch.device) -> AbstractModelFeature:
        """
        :param device: desired device to move feature to
        :return feature type that was moved to a device
        """
        validate_type(self.agent_id, torch.Tensor)
        validate_type(self.agent_type, torch.Tensor)
        validate_type(self.agent_states, torch.Tensor)
        validate_type(self.agent_role, torch.Tensor)
        return AgentFeature(
            agent_id=self.agent_id.to(device=device),
            agent_type=self.agent_type.to(device=device),
            agent_states=self.agent_states.to(device=device),
            agent_role=self.agent_role.to(device=device),
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
