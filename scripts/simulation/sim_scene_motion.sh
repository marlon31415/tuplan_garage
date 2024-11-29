SPLIT=val14_split
CHALLENGE=closed_loop_reactive_agents # open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents
CHECKPOINT="/home/wiss/steiner/projects/tuplan_garage/ckpt/model.ckpt"
# CHECKPOINT=1e0x3595:v15

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
+simulation=$CHALLENGE \
planner=scene_motion_planner \
scenario_filter=$SPLIT \
scenario_builder=nuplan \
planner.scene_motion_planner.model='\${model}' \
model=scene_motion_model \
model.checkpoint=$CHECKPOINT \
hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments, pkg://tuplan_garage.planning.training.modeling.models.configs]"
