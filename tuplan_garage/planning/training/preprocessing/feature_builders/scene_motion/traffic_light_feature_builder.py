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
from tuplan_garage.planning.external_submodules.future_motion.src.external_submodules.hptr.src.pack_h5_nuplan import (
    collate_tl_features,
)


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

        N_STEP = self.num_history + 1
        STEP_CURRENT = self.num_history

        return self._compute_feature(
            map_api, center, traffic_light_data, N_STEP, STEP_CURRENT
        )

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
        n_step,
        step_current,
    ) -> TrafficLightFeature:
        tl_lane_state, tl_lane_id, tl_stop_point = collate_tl_features(
            map_api, scenario_center, traffic_light_data, n_step, step_current
        )

        return TrafficLightFeature(
            tl_lane_state=tl_lane_state,
            tl_lane_id=tl_lane_id,
            tl_stop_point=tl_stop_point,
        )
