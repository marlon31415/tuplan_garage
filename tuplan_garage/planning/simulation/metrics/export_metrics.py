import pandas as pd
import glob
import os


def read_parquet_files(directory):
    all_files = glob.glob(os.path.join(directory, "*.parquet"))
    df_list = [pd.read_parquet(file) for file in all_files]
    return pd.concat(df_list, ignore_index=True)


def export_metrics(metrics, output_file):
    with open(output_file, "w") as f:
        f.write(metrics.to_string())


def main(aggregator_metric_dir, output_file):
    aggregator_metric_df = read_parquet_files(aggregator_metric_dir)
    aggregator_metric_df = aggregator_metric_df[
        ~aggregator_metric_df["scenario"].str.contains(r"\d")
    ]
    aggregator_metric_df.reset_index(drop=True, inplace=True)
    aggregator_metric_df = aggregator_metric_df.dropna(axis=1, how="all")
    export_metrics(aggregator_metric_df, output_file)


if __name__ == "__main__":
    challenge = "closed_loop_reactive_agents"
    sim_tag = "2024.11.29.19.54.03"
    aggregator_metric_dir = f"/home/steiner/nuplan/exp/exp/simulation/{challenge}/{sim_tag}/aggregator_metric"
    output_file_name = f"{challenge}_{sim_tag}_metrics"
    output_file = f"/home/steiner/projects/motion_prediction_and_planning/tuplan_garage/tuplan_garage/planning/simulation/metrics/{output_file_name}.txt"
    main(aggregator_metric_dir, output_file)


val14_scenarios = [
    "starting_right_turn",
    "high_magnitude_speed",
    "near_multiple_vehicles",
    "following_lane_with_lead",
    "traversing_pickup_dropoff",
    "starting_left_turn",
    "starting_straight_traffic_light_intersection_traversal",
    "stationary_in_traffic",
    "stopping_with_lead",
    "high_lateral_acceleration",
    "changing_lane",
    "waiting_for_pedestrian_to_cross",
    "low_magnitude_speed",
    "behind_long_vehicle",
    # "final_score",
]
