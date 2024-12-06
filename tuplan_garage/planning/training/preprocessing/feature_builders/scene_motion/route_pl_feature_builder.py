from typing import Type

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)
from tuplan_garage.planning.training.preprocessing.features.scene_motion.route_pl_feature import (
    RoutePolylineFeature,
)
from tuplan_garage.planning.external_submodules.future_motion.src.external_submodules.hptr.src.pack_h5_nuplan import (
    collate_route_features,
)


class RoutePolylineFeatureBuilder(AbstractFeatureBuilder):
    """Builder for constructing route polyline features in h5 format (see future-motion or HPTR)"""

    def __init__(self, radius: int = 200):
        self.radius = radius

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Type of the built feature."""
        return RoutePolylineFeature

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Unique string identifier of the built feature."""
        return "route_polyline_feature"

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> RoutePolylineFeature:
        """
        Inherited, see superclass.
        """
        map_api = initialization.map_api
        route_roadblock_ids = initialization.route_roadblock_ids
        center = current_input.history.ego_states[-1].center.point
        mission_goal = initialization.mission_goal

        return self._compute_feature(
            map_api, center, route_roadblock_ids, mission_goal, self.radius
        )

    def get_features_from_scenario(
        self, scenario: AbstractScenario
    ) -> AbstractModelFeature:
        """
        Inherited, see superclass.
        """
        pass

    def _compute_feature(
        self, map_api, scenario_center, route_roadblock_ids, mission_goal, radius=200
    ) -> RoutePolylineFeature:
        sdc_id, sdc_route_lane_id, sdc_route_type, sdc_route_xyz, sdc_route_goal = (
            collate_route_features(
                map_api, scenario_center, route_roadblock_ids, mission_goal, radius
            )
        )

        return RoutePolylineFeature(
            sdc_id=sdc_id,
            sdc_route_lane_id=sdc_route_lane_id,
            sdc_route_type=sdc_route_type,
            sdc_route_xyz=sdc_route_xyz,
            sdc_route_goal=sdc_route_goal,
        )
