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
from tuplan_garage.planning.training.preprocessing.features.scene_motion.map_pl_feature import (
    MapPolylineFeature,
)
from tuplan_garage.planning.external_submodules.future_motion.src.external_submodules.hptr.src.pack_h5_nuplan import (
    collate_map_features,
)


class MapPolylineFeatureBuilder(AbstractFeatureBuilder):
    """Builder for constructing route polyline features in h5 format (see future-motion or HPTR)"""

    def __init__(self, radius: int = 200):
        self.radius = radius

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Type of the built feature."""
        return MapPolylineFeature

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Unique string identifier of the built feature."""
        return "route_polyline_feature"

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> MapPolylineFeature:
        """
        Inherited, see superclass.
        """
        map_api = initialization.map_api
        center = current_input.history.ego_states[-1].center.point

        return self._compute_feature(map_api, center, self.radius)

    def get_features_from_scenario(
        self, scenario: AbstractScenario
    ) -> AbstractModelFeature:
        """
        Inherited, see superclass.
        """
        pass

    def _compute_feature(
        self, map_api, scenario_center, radius=200
    ) -> MapPolylineFeature:
        mf_id, mf_xyz, mf_type, mf_edge = collate_map_features(
            map_api, scenario_center, radius
        )

        return MapPolylineFeature(
            mf_id=mf_id, mf_xyz=mf_xyz, mf_type=mf_type, mf_edge=mf_edge
        )
