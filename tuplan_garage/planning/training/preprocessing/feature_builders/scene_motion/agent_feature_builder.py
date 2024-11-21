from typing import Type, cast
import numpy as np
import copy

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.scenario_builder.scenario_utils import (
    sample_indices_with_time_horizon,
)
from tuplan_garage.planning.training.preprocessing.features.scene_motion.agent_feature import (
    AgentFeature,
)
from tuplan_garage.planning.external_submodules.future_motion.src.external_submodules.hptr.src.utils.pack_h5_nuplan_utils import (
    parse_object_state,
)

AGENT_TYPES = {
    "VEHICLE": 0,  # Includes all four or more wheeled vehicles, as well as trailers.
    "PEDESTRIAN": 1,  # All types of pedestrians, incl. strollers and wheelchairs.
    "BICYCLE": 2,  # Includes bicycles, motorcycles and tricycles.
    "TRAFFIC_CONE": 3,  # Cones that are temporarily placed to control the flow of traffic.
    "BARRIER": 3,  # Solid barriers that can be either temporary or permanent.
    "CZONE_SIGN": 3,  # Temporary signs that indicate construction zones.
    "GENERIC_OBJECT": 3,  # Animals, debris, pushable/pullable objects, permanent poles.
    "EGO": 0,  # The ego vehicle.
}
N_AGENT_TYPE = len(set(AGENT_TYPES.values()))
N_SDC_AGENT = 1
N_AGENT_PRED_CHALLENGE = 8
N_AGENT_INTERACT_CHALLENGE = 2


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

        return self._compute_feature(
            center, sampled_past_ego_states, sampled_past_observations
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
    ) -> AgentFeature:
        N_STEP = self.num_past_poses + 1
        N_CURRENT = N_STEP - 1

        # Tuple instead of Point2d for compatibility with nuplan pack utils
        scenario_center_tuple = [scenario_center.x, scenario_center.y]

        common_states = [
            [ego] + other for ego, other in zip(ego_states, observation_states)
        ]

        agent_id = []
        agent_type = []
        agent_states = []
        agent_role = []

        default_track = {
            "type": "UNSET",
            "state": {
                "position_x": np.zeros(shape=(N_STEP,)),
                "position_y": np.zeros(shape=(N_STEP,)),
                "position_z": np.zeros(shape=(N_STEP,)),
                "length": np.zeros(shape=(N_STEP,)),
                "width": np.zeros(shape=(N_STEP,)),
                "height": np.zeros(shape=(N_STEP,)),
                "heading": np.zeros(shape=(N_STEP,)),
                "velocity_x": np.zeros(shape=(N_STEP,)),
                "velocity_y": np.zeros(shape=(N_STEP,)),
                "valid": np.zeros(shape=(N_STEP,)),
            },
            "metadata": {
                "object_id": None,  # itegers defined by the dataset
                "nuplan_id": None,  # hex ids
                "distance_to_ego": 1000,
            },
        }

        tracks = {}
        dists_to_ego = []
        for i in range(N_STEP):
            current_observation = common_states[i]
            for obj in current_observation:
                tracked_object_type = obj.tracked_object_type
                if tracked_object_type is None or (
                    self.only_agents and AGENT_TYPES[tracked_object_type.name] > 2
                ):
                    continue
                track_token = obj.metadata.track_token
                if track_token not in tracks:
                    # add new track with token as key
                    tracks[track_token] = copy.deepcopy(default_track)
                    tracks[track_token]["metadata"]["nuplan_id"] = track_token
                    tracks[track_token]["metadata"]["object_id"] = obj.metadata.track_id
                    tracks[track_token]["type"] = AGENT_TYPES[tracked_object_type.name]

                state = parse_object_state(obj, scenario_center_tuple)
                self._fill_track_with_state(tracks[track_token]["state"], state, i)

                if i == N_CURRENT and int(tracks[track_token]["type"]) <= 2:
                    dist_to_ego = np.linalg.norm(
                        np.array([obj.center.x, obj.center.y])
                        - np.array(scenario_center_tuple)
                    )
                    dists_to_ego.append(dist_to_ego)
                    tracks[track_token]["metadata"]["distance_to_ego"] = dist_to_ego

        # adapt ego velocity
        self._calc_velocity_from_positions(tracks["ego"]["state"], self.interval_length)

        predict_dist, interest_dist = self._get_max_distance_for_challenges(
            dists_to_ego
        )

        for nuplan_id, track in tracks.items():
            _dist_to_ego = track["metadata"]["distance_to_ego"]
            agent_role.append([False, False, False])
            if track["type"] in [0, 1, 2]:
                agent_role[-1][2] = True if _dist_to_ego <= predict_dist else False
                agent_role[-1][1] = True if _dist_to_ego <= interest_dist else False
            if nuplan_id == "ego":
                agent_role[-1] = [True, True, True]

            agent_id.append(track["metadata"]["object_id"])
            agent_type.append(track["type"])
            agent_states_list = np.vstack(list(track["state"].values())).T.tolist()
            agent_states.append(agent_states_list)

        return AgentFeature(
            agent_id=agent_id,
            agent_type=agent_type,
            agent_states=agent_states,
            agent_role=agent_role,
        )

    def _fill_track_with_state(
        self, track_state: dict, state: dict, current_timestep: int
    ) -> dict:
        """
        Fills a track with the information from the track_obj
        :param track: the track to fill
        :param track_obj: the track object
        :return: the filled track
        """
        track_state["position_x"][current_timestep] = state["position"][0]
        track_state["position_y"][current_timestep] = state["position"][1]
        track_state["heading"][current_timestep] = state["heading"]
        track_state["velocity_x"][current_timestep] = state["velocity"][0]
        track_state["velocity_y"][current_timestep] = state["velocity"][1]
        track_state["valid"][current_timestep] = state["valid"]
        track_state["length"][current_timestep] = state["length"]
        track_state["width"][current_timestep] = state["width"]
        track_state["height"][current_timestep] = state["height"]

    def _calc_velocity_from_positions(self, track_state: dict, dt: float) -> None:
        positions = np.hstack([track_state["position_x"], track_state["position_y"]])
        velocity = (positions[1:] - positions[:-1]) / dt
        track_state["velocity_x"][:-1] = velocity[..., 0]
        track_state["velocity_x"][-1] = track_state["velocity_x"][-2]
        track_state["velocity_y"][:-1] = velocity[..., 1]
        track_state["velocity_y"][-1] = track_state["velocity_y"][-2]

    def _get_max_distance_for_challenges(self, dists_to_ego: list) -> tuple:
        """
        Get the distance to the ego for the prediction and interaction challenges
        :param dists_to_ego: list of distances to the ego
        :return:
            predict_dist: distance to the ego for vehicles to be considered in the prediction challenge
            interest_dist: distance to the ego for vehicles to be considered in the interaction challenge
        """
        # dists_to_ego includes the ego vehicle (distance 0)
        dists_to_ego.sort()
        predict_dist = (
            dists_to_ego[N_AGENT_PRED_CHALLENGE - 1]
            if len(dists_to_ego) >= N_AGENT_PRED_CHALLENGE
            else dists_to_ego[-1]
        )
        interest_dist = (
            dists_to_ego[N_AGENT_INTERACT_CHALLENGE - 1]
            if len(dists_to_ego) >= N_AGENT_INTERACT_CHALLENGE
            else dists_to_ego[-1]
        )
        return predict_dist, interest_dist
