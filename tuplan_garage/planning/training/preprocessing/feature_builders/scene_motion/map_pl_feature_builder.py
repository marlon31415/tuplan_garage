from typing import Type
import numpy as np
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.common.maps.maps_datatypes import (
    SemanticMapLayer,
    StopLineType,
)
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
from tuplan_garage.planning.external_submodules.future_motion.src.external_submodules.hptr.src.utils.pack_h5_nuplan_utils import (
    nuplan_to_centered_vector,
    mock_2d_to_3d_points,
    extract_centerline,
    get_points_from_boundary,
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

        return self._compute_feature(map_api, center)

    def get_features_from_scenario(
        self, scenario: AbstractScenario
    ) -> AbstractModelFeature:
        """
        Inherited, see superclass.
        """
        pass

    def _compute_feature(self, map_api, scenario_center) -> MapPolylineFeature:
        # map features
        mf_id = []
        mf_xyz = []
        mf_type = []
        mf_edge = []

        # Center is Important !
        layer_names = [
            SemanticMapLayer.LANE_CONNECTOR,
            SemanticMapLayer.LANE,
            SemanticMapLayer.CROSSWALK,
            SemanticMapLayer.INTERSECTION,
            SemanticMapLayer.STOP_LINE,
            SemanticMapLayer.WALKWAYS,
            SemanticMapLayer.CARPARK_AREA,
            SemanticMapLayer.ROADBLOCK,
            SemanticMapLayer.ROADBLOCK_CONNECTOR,
            # unsupported yet
            # SemanticMapLayer.STOP_SIGN,
            # SemanticMapLayer.DRIVABLE_AREA,
        ]
        scenario_center_tuple = [scenario_center.x, scenario_center.y]
        nearest_vector_map = map_api.get_proximal_map_objects(
            scenario_center, self.radius, layer_names
        )

        # STOP LINES
        # Filter out stop polygons from type turn stop
        if SemanticMapLayer.STOP_LINE in nearest_vector_map:
            stop_polygons = nearest_vector_map[SemanticMapLayer.STOP_LINE]
            nearest_vector_map[SemanticMapLayer.STOP_LINE] = [
                stop_polygon
                for stop_polygon in stop_polygons
                if stop_polygon.stop_line_type != StopLineType.TURN_STOP
            ]
            for stop_line_polygon_obj in nearest_vector_map[SemanticMapLayer.STOP_LINE]:
                stop_line_polygon = stop_line_polygon_obj.polygon.exterior.coords
                mf_id.append(stop_line_polygon_obj.id)
                mf_type.append(PL_TYPES["STOP_LINE"])
                polygon_centered = nuplan_to_centered_vector(
                    np.array(stop_line_polygon), nuplan_center=scenario_center_tuple
                )
                mf_xyz.append(mock_2d_to_3d_points(polygon_centered)[::4])

        # LANES
        for layer in [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]:
            for lane in nearest_vector_map[layer]:
                if not hasattr(lane, "baseline_path"):
                    continue
                # Centerline (as polyline)
                centerline = extract_centerline(lane, scenario_center_tuple, True, 1)
                mf_id.append(int(lane.id))  # using lane ids for centerlines!
                mf_type.append(PL_TYPES["CENTERLINE"])
                mf_xyz.append(mock_2d_to_3d_points(centerline))
                if len(lane.outgoing_edges) > 0:
                    for _out_edge in lane.outgoing_edges:
                        mf_edge.append([int(lane.id), int(_out_edge.id)])
                else:
                    mf_edge.append([int(lane.id), -1])
                # Left boundary of centerline (as polyline)
                left = lane.left_boundary
                left_polyline = get_points_from_boundary(
                    left, scenario_center_tuple, True, 1
                )
                mf_id.append(left.id)
                mf_type.append(PL_TYPES["LINE_BROKEN_SINGLE_WHITE"])
                mf_xyz.append(mock_2d_to_3d_points(left_polyline))
                # right = lane.right_boundary
                # right_polyline = get_points_from_boundary(right, center)
                # mf_id.append(right.id)
                # mf_type.append(PL_TYPES["LINE_BROKEN_SINGLE_WHITE"])
                # mf_xyz.append(mock_2d_to_3d_points(right_polyline)[::2])

        # ROADBLOCKS (contain lanes)
        # Extract neighboring lanes and road boundaries
        block_polygons = []
        for layer in [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]:
            for block in nearest_vector_map[layer]:
                # roadblock_polygon = block.polygon.boundary.xy
                # polygon = nuplan_to_centered_vector(
                #     np.array(roadblock_polygon).T, scenario_center_tuple
                # )

                # According to the map attributes, lanes are numbered left to right with smaller indices being on the
                # left and larger indices being on the right.
                lanes = (
                    sorted(block.interior_edges, key=lambda lane: lane.index)
                    if layer == SemanticMapLayer.ROADBLOCK
                    else block.interior_edges
                )
                for i, lane in enumerate(lanes):
                    if not hasattr(lane, "baseline_path"):
                        continue
                    if layer == SemanticMapLayer.ROADBLOCK:
                        if i != 0:
                            left_neighbor = lanes[i - 1]
                            mf_edge.append([int(lane.id), int(left_neighbor.id)])
                        if i != len(lanes) - 1:
                            right_neighbor = lanes[i + 1]
                            mf_edge.append([int(lane.id), int(right_neighbor.id)])
                        # if i == 0:  # left most lane
                        #     left = lane.left_boundary
                        #     left_boundary = get_points_from_boundary(left, center, True, 1)
                        #     try:
                        #         idx = mf_id.index(left.id)
                        #         mf_id[idx] = left.id  # use roadblock ids for boundaries
                        #         mf_type[idx] = PL_TYPES["BOUNDARIES"]
                        #         mf_xyz[idx] = mock_2d_to_3d_points(left_boundary)
                        #     except:
                        #         mf_id.append(block.id)  # use roadblock ids for boundaries
                        #         mf_type.append(PL_TYPES["BOUNDARIES"])
                        #         mf_xyz.append(mock_2d_to_3d_points(right_boundary))
                        if i == len(lanes) - 1:  # right most lane
                            right = lane.right_boundary
                            right_boundary = get_points_from_boundary(
                                right, scenario_center_tuple, True, 1
                            )
                            try:
                                idx = mf_id.index(right.id)
                                mf_id[idx] = (
                                    right.id
                                )  # use roadblock ids for boundaries
                                mf_type[idx] = PL_TYPES["BOUNDARIES"]
                                mf_xyz[idx] = mock_2d_to_3d_points(right_boundary)
                            except:
                                mf_id.append(
                                    block.id
                                )  # use roadblock ids for boundaries
                                mf_type.append(PL_TYPES["BOUNDARIES"])
                                mf_xyz.append(mock_2d_to_3d_points(right_boundary))

                if layer == SemanticMapLayer.ROADBLOCK:
                    block_polygons.append(block.polygon)

        # WALKWAYS
        for area in nearest_vector_map[SemanticMapLayer.WALKWAYS]:
            if isinstance(area.polygon.exterior, MultiLineString):
                boundary = gpd.GeoSeries(area.polygon.exterior).explode(
                    index_parts=True
                )
                sizes = []
                for idx, polygon in enumerate(boundary[0]):
                    sizes.append(len(polygon.xy[1]))
                points = boundary[0][np.argmax(sizes)].xy
            elif isinstance(area.polygon.exterior, LineString):
                points = area.polygon.exterior.xy
            polygon = nuplan_to_centered_vector(
                np.array(points).T, scenario_center_tuple
            )
            mf_id.append(int(area.id))
            mf_type.append(PL_TYPES["WALKWAYS"])
            mf_xyz.append(mock_2d_to_3d_points(polygon)[::4])

        # CROSSWALK
        for area in nearest_vector_map[SemanticMapLayer.CROSSWALK]:
            if isinstance(area.polygon.exterior, MultiLineString):
                boundary = gpd.GeoSeries(area.polygon.exterior).explode(
                    index_parts=True
                )
                sizes = []
                for idx, polygon in enumerate(boundary[0]):
                    sizes.append(len(polygon.xy[1]))
                points = boundary[0][np.argmax(sizes)].xy
            elif isinstance(area.polygon.exterior, LineString):
                points = area.polygon.exterior.xy
            polygon = nuplan_to_centered_vector(
                np.array(points).T, scenario_center_tuple
            )
            mf_id.append(int(area.id))
            mf_type.append(PL_TYPES["CROSSWALK"])
            mf_xyz.append(mock_2d_to_3d_points(polygon)[::4])

        # INTERSECTION
        interpolygons = [
            block.polygon for block in nearest_vector_map[SemanticMapLayer.INTERSECTION]
        ]
        boundaries = gpd.GeoSeries(
            unary_union(interpolygons + block_polygons)
        ).boundary.explode(index_parts=True)
        # boundaries.plot()
        # plt.show()
        for idx, boundary in enumerate(boundaries[0]):
            block_points = np.array(
                list(i for i in zip(boundary.coords.xy[0], boundary.coords.xy[1]))
            )
            block_points = nuplan_to_centered_vector(
                block_points, scenario_center_tuple
            )
            mf_id.append(idx)
            mf_type.append(PL_TYPES["INTERSECTION"])
            mf_xyz.append(mock_2d_to_3d_points(block_points)[::4])

        return MapPolylineFeature(
            mf_id=mf_id, mf_xyz=mf_xyz, mf_type=mf_type, mf_edge=mf_edge
        )
