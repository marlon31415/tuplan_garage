from typing import Any, Type

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)
from tuplan_garage.planning.training.preprocessing.features.scene_motion.traffic_light_feature import (
    TrafficLightFeature,
)
from tuplan_garage.planning.external_submodules.future_motion.src.external_submodules.hptr.src.utils.pack_h5_nuplan_utils import (
    set_light_position,
    mock_2d_to_3d_points,
)

TL_TYPES = {
    "GREEN": 3,
    "YELLOW": 2,
    "RED": 1,
    "UNKNOWN": 0,
}


class TrafficLightFeatureBuilder(AbstractFeatureBuilder):
    """Builder for constructing route polyline features in h5 format (see future-motion or HPTR)"""

    def __init__(self, num_history) -> None:
        self.num_history = num_history

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Type of the built feature."""
        return TrafficLightFeature

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Unique string identifier of the built feature."""
        return "route_polyline_feature"

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> TrafficLightFeature:
        """
        Inherited, see superclass.
        """
        map_api = initialization.map_api
        center = current_input.history.ego_states[-1].center.point
        traffic_light_data = current_input.traffic_light_data

        return self._compute_feature(map_api, center, traffic_light_data)

    def get_features_from_scenario(
        self, scenario: AbstractScenario
    ) -> AbstractModelFeature:
        """
        Inherited, see superclass.
        """
        pass

    def _compute_feature(
        self,
        map_api,
        scenario_center,
        traffic_light_data,
    ) -> TrafficLightFeature:
        N_STEP = self.num_history + 1
        scenario_center_tuple = [scenario_center.x, scenario_center.y]

        tl_lane_state = []
        tl_lane_id = []
        tl_stop_point = []

        # PlannerInput does not contain TL information for history
        # -> add empty list for all history steps
        # TODO: how to get past TL information?
        tl_lane_state.append([] * (N_STEP - 1))
        tl_lane_id.append([] * (N_STEP - 1))
        tl_stop_point.append([] * (N_STEP - 1))

        tl_lane_state.append(
            [TL_TYPES[tl_data.status.name] for tl_data in traffic_light_data]
        )
        tl_lane_id.append([tl_data.lane_connector_id for tl_data in traffic_light_data])
        tl_stop_point_2d = [
            set_light_position(map_api, lane_id, scenario_center_tuple)
            for lane_id in tl_lane_id[-1]
        ]
        tl_stop_point.append(mock_2d_to_3d_points(tl_stop_point_2d))

        return TrafficLightFeature(
            tl_lane_state=tl_lane_state,
            tl_lane_id=tl_lane_id,
            tl_stop_point=tl_stop_point,
        )
