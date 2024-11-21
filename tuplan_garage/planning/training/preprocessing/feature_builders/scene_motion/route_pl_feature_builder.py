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
from tuplan_garage.planning.external_submodules.future_motion.src.external_submodules.hptr.src.utils.pack_h5_nuplan_utils import (
    get_route_lane_polylines_from_roadblock_ids,
    nuplan_to_centered_vector,
    mock_2d_to_3d_points,
)

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

        return self._compute_feature(map_api, center, route_roadblock_ids)

    def get_features_from_scenario(
        self, scenario: AbstractScenario
    ) -> AbstractModelFeature:
        """
        Inherited, see superclass.
        """
        pass

    def _compute_feature(
        self, map_api, scenario_center, route_roadblock_ids
    ) -> RoutePolylineFeature:
        scenario_center_tuple = [scenario_center.x, scenario_center.y]

        # id=-1 is the default nuplan value for the ego; TODO: change this if needed
        sdc_id = [-1]
        sdc_route_type = []
        sdc_route_lane_id = []
        sdc_route_xyz = []

        polylines, route_lane_ids = get_route_lane_polylines_from_roadblock_ids(
            map_api, scenario_center, self.radius, route_roadblock_ids
        )
        route_lane_polylines = []
        pl_types = []
        for polyline in polylines:
            polyline_centered = nuplan_to_centered_vector(
                polyline, scenario_center_tuple
            )
            route_lane_polylines.append(mock_2d_to_3d_points(polyline_centered)[::10])
            pl_types.append(PL_TYPES["ROUTE"])
        sdc_route_xyz.append(route_lane_polylines)
        sdc_route_lane_id.append(route_lane_ids)
        sdc_route_type.append(pl_types)

        return RoutePolylineFeature(
            sdc_id=sdc_id,
            sdc_route_lane_id=sdc_route_lane_id,
            sdc_route_type=sdc_route_type,
            sdc_route_xyz=sdc_route_xyz,
        )
