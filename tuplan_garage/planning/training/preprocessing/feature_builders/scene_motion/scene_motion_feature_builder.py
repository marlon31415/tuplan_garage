from typing import List, Tuple, Type
import torch
import numpy as np

from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
)
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
)
from tuplan_garage.planning.training.preprocessing.features.scene_motion.agent_feature import (
    AgentFeature,
)
from tuplan_garage.planning.training.preprocessing.features.scene_motion.map_pl_feature import (
    MapPolylineFeature,
)
from tuplan_garage.planning.training.preprocessing.features.scene_motion.route_pl_feature import (
    RoutePolylineFeature,
)
from tuplan_garage.planning.training.preprocessing.features.scene_motion.traffic_light_feature import (
    TrafficLightFeature,
)
from tuplan_garage.planning.training.preprocessing.feature_builders.scene_motion.agent_feature_builder import (
    AgentFeatureBuilder,
)
from tuplan_garage.planning.training.preprocessing.feature_builders.scene_motion.map_pl_feature_builder import (
    MapPolylineFeatureBuilder,
)
from tuplan_garage.planning.training.preprocessing.feature_builders.scene_motion.route_pl_feature_builder import (
    RoutePolylineFeatureBuilder,
)
from tuplan_garage.planning.training.preprocessing.feature_builders.scene_motion.traffic_light_feature_builder import (
    TrafficLightFeatureBuilder,
)
from tuplan_garage.planning.training.preprocessing.features.scene_motion.scene_motion_features import (
    SceneMotionFeatures,
)
import tuplan_garage.planning.external_submodules.future_motion.src.external_submodules.hptr.src.utils.pack_h5 as pack_utils

PL_TYPES = {
    "INTERSECTION": 0,
    "STOP_LINE": 1,
    "CROSSWALK": 2,
    "BOUNDARIES": 3,
    "WALKWAYS": 4,
    "CARPARK_AREA": 5,
    "LINE_BROKEN_SINGLE_WHITE": 6,
    "CENTERLINE": 7,
    "ROUTE": 8,
}
N_PL_TYPE = len(PL_TYPES)
DIM_VEH_LANES = [7]
DIM_CYC_LANES = [3, 7]
DIM_PED_LANES = [2, 3, 4]

TL_TYPES = {
    "GREEN": 3,
    "YELLOW": 2,
    "RED": 1,
    "UNKNOWN": 0,
}
N_TL_STATE = len(TL_TYPES)

AGENT_TYPES = {
    "VEHICLE": 0,  # Includes all four or more wheeled vehicles, as well as trailers.
    "PEDESTRIAN": 1,  # Includes bicycles, motorcycles and tricycles.
    "BICYCLE": 2,  # All types of pedestrians, incl. strollers and wheelchairs.
    "TRAFFIC_CONE": 3,  # Cones that are temporarily placed to control the flow of traffic.
    "BARRIER": 3,  # Solid barriers that can be either temporary or permanent.
    "CZONE_SIGN": 3,  # Temporary signs that indicate construction zones.
    "GENERIC_OBJECT": 3,  # Animals, debris, pushable/pullable objects, permanent poles.
    "EGO": 0,  # The ego vehicle.
}
N_AGENT_TYPE = len(set(AGENT_TYPES.values()))

N_PL_MAX = 2000
N_TL_MAX = 40
N_AGENT_MAX = 800
N_PL_ROUTE_MAX = 250

N_PL = 1024
N_TL = 200  # due to polyline splitting this value can be higher than N_TL_MAX
N_AGENT = 64
N_AGENT_NO_SIM = N_AGENT_MAX - N_AGENT
N_PL_ROUTE = N_PL_ROUTE_MAX

THRESH_MAP = 120
THRESH_AGENT = 120

N_SDC_AGENT = 1
N_AGENT_PRED_CHALLENGE = 8
N_AGENT_INTERACT_CHALLENGE = 2


class SceneMotionFeatureBuilder(AbstractFeatureBuilder):
    """Builder for constructing features in h5 format (see future-motion or HPTR)"""

    def __init__(
        self,
        agent_feature_builder: AgentFeatureBuilder,
        map_polyline_feature_builder: MapPolylineFeatureBuilder,
        route_polyline_feature_builder: RoutePolylineFeatureBuilder,
        traffic_light_feature_builder: TrafficLightFeatureBuilder,
        pack_all: bool,
        pack_history: bool,
        dest_no_pred: bool,
        rand_pos: float = -1,  # -1: disable
        rand_yaw: float = -1,  # -1: disable
    ):
        self.agent_feature_builder = agent_feature_builder
        self.map_polyline_feature_builder = map_polyline_feature_builder
        self.route_polyline_feature_builder = route_polyline_feature_builder
        self.traffic_light_feature_builder = traffic_light_feature_builder

        self.pack_all = pack_all
        self.pack_history = pack_history
        self.dest_no_pred = dest_no_pred
        self.rand_pos = rand_pos
        self.rand_yaw = rand_yaw

        self.num_past_poses = agent_feature_builder.num_past_poses

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Type of the built feature."""
        return SceneMotionFeatures

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Unique string identifier of the built feature."""
        return "scene_motion_features"

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> SceneMotionFeatures:
        """Inherited, see superclass."""
        agent_feature = self.agent_feature_builder.get_features_from_simulation(
            current_input, initialization
        )
        map_polyline_feature = (
            self.map_polyline_feature_builder.get_features_from_simulation(
                current_input, initialization
            )
        )
        route_polyline_feature = (
            self.route_polyline_feature_builder.get_features_from_simulation(
                current_input, initialization
            )
        )
        traffic_light_feature = (
            self.traffic_light_feature_builder.get_features_from_simulation(
                current_input, initialization
            )
        )
        return self._compute_feature(
            agent_feature=agent_feature,
            map_polyline_feature=map_polyline_feature,
            route_polyline_feature=route_polyline_feature,
            traffic_light_feature=traffic_light_feature,
        )

    def get_features_from_scenario(
        self, scenario: AbstractScenario
    ) -> AbstractModelFeature:
        """
        Constructs model input features from a database samples.
        :param scenario: Generic scenario
        :return: Constructed features
        """
        pass

    def _compute_feature(
        self,
        agent_feature: AgentFeature,
        map_polyline_feature: MapPolylineFeature,
        route_polyline_feature: RoutePolylineFeature,
        traffic_light_feature: TrafficLightFeature,
    ) -> SceneMotionFeatures:
        N_STEP = self.num_past_poses + 1
        STEP_CURRENT = N_STEP - 1

        episode = {}
        n_pl = pack_utils.pack_episode_map(
            episode=episode,
            mf_id=map_polyline_feature.mf_id,
            mf_xyz=map_polyline_feature.mf_xyz,
            mf_type=map_polyline_feature.mf_type,
            mf_edge=map_polyline_feature.mf_edge,
            n_pl_max=N_PL_MAX,
        )
        n_tl = pack_utils.pack_episode_traffic_lights(
            episode=episode,
            tl_lane_state=traffic_light_feature.tl_lane_state,
            tl_lane_id=traffic_light_feature.tl_lane_id,
            tl_stop_point=traffic_light_feature.tl_stop_point,
            pack_all=self.pack_all,
            pack_history=self.pack_history,
            n_tl_max=N_TL_MAX,
            step_current=STEP_CURRENT,
        )
        n_agent = pack_utils.pack_episode_agents(
            episode=episode,
            agent_id=agent_feature.agent_id,
            agent_type=agent_feature.agent_type,
            agent_states=agent_feature.agent_states,
            agent_role=agent_feature.agent_role,
            pack_all=self.pack_all,
            pack_history=self.pack_history,
            n_agent_max=N_AGENT_MAX,
            step_current=STEP_CURRENT,
        )
        n_route_pl = pack_utils.pack_episode_route(
            episode=episode,
            sdc_id=route_polyline_feature.sdc_id,
            sdc_route_id=route_polyline_feature.sdc_route_lane_id,
            sdc_route_type=route_polyline_feature.sdc_route_type,
            sdc_route_xyz=route_polyline_feature.sdc_route_xyz,
            n_route_pl_max=N_PL_ROUTE_MAX,
        )
        scenario_center, scenario_yaw = pack_utils.center_at_sdc(
            episode, self.rand_pos, self.rand_yaw
        )

        episode_reduced = {}
        pack_utils.filter_episode_map(episode, N_PL, THRESH_MAP, thresh_z=3)
        episode_with_map = episode["map/valid"].any(1).sum() > 0
        pack_utils.repack_episode_map(episode, episode_reduced, N_PL, N_PL_TYPE)

        pack_utils.repack_episode_route(episode, episode_reduced, N_PL_ROUTE, N_PL_TYPE)

        pack_utils.filter_episode_traffic_lights(episode)
        pack_utils.repack_episode_traffic_lights(
            episode, episode_reduced, N_TL, N_TL_STATE
        )

        # Analogous to testing split
        mask_sim, mask_no_sim = pack_utils.filter_episode_agents(
            episode=episode,
            episode_reduced=episode_reduced,
            n_agent=N_AGENT,
            prefix="history/",
            dim_veh_lanes=DIM_VEH_LANES,
            dist_thresh_agent=THRESH_AGENT,
            step_current=STEP_CURRENT,
        )
        pack_utils.repack_episode_agents(
            episode, episode_reduced, mask_sim, N_AGENT, "history/"
        )
        pack_utils.repack_episode_agents_no_sim(
            episode, episode_reduced, mask_no_sim, N_AGENT_NO_SIM, "history/"
        )

        return SceneMotionFeatures(
            history_agent_valid=episode_reduced["history/agent/valid"],
            history_agent_pos=episode_reduced["history/agent/pos"],
            history_agent_vel=episode_reduced["history/agent/vel"],
            history_agent_spd=episode_reduced["history/agent/spd"],
            history_agent_acc=episode_reduced["history/agent/acc"],
            history_agent_yaw_bbox=episode_reduced["history/agent/yaw_bbox"],
            history_agent_yaw_rate=episode_reduced["history/agent/yaw_rate"],
            history_agent_type=episode_reduced["history/agent/type"],
            history_agent_role=episode_reduced["history/agent/role"],
            history_agent_size=episode_reduced["history/agent/size"],
            history_tl_stop_valid=episode_reduced["history/tl_stop/valid"],
            history_tl_stop_state=episode_reduced["history/tl_stop/state"],
            history_tl_stop_pos=episode_reduced["history/tl_stop/pos"],
            history_tl_stop_dir=episode_reduced["history/tl_stop/dir"],
            map_valid=episode_reduced["map/valid"],
            map_type=episode_reduced["map/type"],
            map_pos=episode_reduced["map/pos"],
            map_dir=episode_reduced["map/dir"],
            route_valid=episode_reduced["route/valid"],
            route_type=episode_reduced["route/type"],
            route_pos=episode_reduced["route/pos"],
            route_dir=episode_reduced["route/dir"],
        )
