from typing import Type, cast

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.scenario_builder.scenario_utils import (
    sample_indices_with_time_horizon,
)
from tuplan_garage.planning.training.preprocessing.features.scene_motion.agent_feature import (
    AgentFeature,
)
from tuplan_garage.planning.external_submodules.future_motion.src.external_submodules.hptr.src.pack_h5_nuplan import (
    collate_agent_features,
    N_AGENT_PRED_CHALLENGE,
    N_AGENT_INTERACT_CHALLENGE,
)


class AgentFeatureBuilder(AbstractFeatureBuilder):
    """Builder for constructing route polyline features in h5 format (see future-motion or HPTR)"""

    def __init__(
        self, trajectory_sampling: TrajectorySampling, only_agents: bool = True
    ):
        self.num_past_poses = trajectory_sampling.num_poses
        self.past_time_horizon = trajectory_sampling.time_horizon
        self.interval_length = trajectory_sampling.interval_length
        self.only_agents = only_agents

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Type of the built feature."""
        return AgentFeature

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Unique string identifier of the built feature."""
        return "route_polyline_feature"

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> AgentFeature:
        """
        Inherited, see superclass.
        """
        present_ego_state, present_observation = current_input.history.current_state
        past_observations = current_input.history.observations[:-1]
        past_ego_states = current_input.history.ego_states[:-1]
        center = present_ego_state.center.point

        indices = sample_indices_with_time_horizon(
            self.num_past_poses,
            self.past_time_horizon,
            current_input.history.sample_interval,
        )
        try:
            sampled_past_observations = [
                cast(
                    DetectionsTracks, past_observations[-idx]
                ).tracked_objects.get_agents()
                for idx in reversed(indices)
            ]
            sampled_past_ego_states = [
                past_ego_states[-idx].agent for idx in reversed(indices)
            ]
        except IndexError:
            raise RuntimeError(
                f"SimulationHistoryBuffer duration: {current_input.history.duration} is "
                f"too short for requested past_time_horizon: {self.past_time_horizon}. "
                f"Please increase the simulation_history_buffer_duration in default_simulation.yaml"
            )
        sampled_past_observations = sampled_past_observations + [
            cast(DetectionsTracks, present_observation).tracked_objects.get_agents()
        ]
        sampled_past_ego_states = sampled_past_ego_states + [present_ego_state.agent]

        assert (
            len(sampled_past_ego_states) == self.num_past_poses + 1
        ), "Number of sampled history steps does not match the expected number of steps"

        n_step = self.num_past_poses + 1

        return self._compute_feature(
            center,
            sampled_past_ego_states,
            sampled_past_observations,
            n_step,
        )

    def get_features_from_scenario(self, scenario: AbstractScenario) -> AgentFeature:
        """
        Inherited, see superclass.
        """
        pass

    def _compute_feature(
        self,
        scenario_center,
        ego_states,
        observation_states,
        n_step,
        only_agents=True,
        interval_length=0.1,
        n_agent_pred_challenge=N_AGENT_PRED_CHALLENGE,
        n_agent_interact_challange=N_AGENT_INTERACT_CHALLENGE,
    ) -> AgentFeature:
        agent_id, agent_type, agent_states, agent_role = collate_agent_features(
            scenario_center,
            ego_states,
            observation_states,
            n_step,
            only_agents,
            interval_length,
            n_agent_pred_challenge,
            n_agent_interact_challange,
        )

        return AgentFeature(
            agent_id=agent_id,
            agent_type=agent_type,
            agent_states=agent_states,
            agent_role=agent_role,
        )
