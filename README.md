![Black](https://img.shields.io/badge/code%20style-black-000000.svg)

# Motion Planning with nuPlan
This repository builds upon the work of [tuPlan Garage](https://github.com/autonomousvision/tuplan_garage) by [@autonomousvision](https://github.com/autonomousvision) and adds our [SceneMotion](https://www.arxiv.org/abs/2408.01537) prediction model (see [future-motion](https://github.com/KIT-MRT/future-motion) on Github). Here, the original model is adapted to the planning task. The modified implementation can be found [here](https://github.com/marlon31415/future-motion). We only use this repository to simulate our planner. Training takes place within our [future-motion repository](https://github.com/KIT-MRT/future-motion). 

## Results
Planning results on the proposed *Val14* benchmark. Please refer to the [paper](https://arxiv.org/abs/2306.07962) for more details.

| **Method**                                        | **Representation** | **CLS-R ↑** | **CLS-NR ↑** | **OLS ↑** | **Time (ms) ↓** |
| ------------------------------------------------- | ------------------ | ----------- | ------------ | --------- | --------------- |
| SceneMotion                                       | Polylines          |             |              |           |                 |
| [Urban Driver](https://arxiv.org/abs/2109.13333)* | Polygon            | 50          | 53           | 82        | 64              |
| [GC-PGP](https://arxiv.org/abs/2302.07753v1)      | Graph              | 55          | 59           | 83        | 100             |
| [PlanCNN](https://arxiv.org/abs/2210.14222)       | Raster             | 72          | 73           | 64        | 43              |
| [IDM](https://arxiv.org/abs/cond-mat/0002177)     | Centerline         | 77          | 76           | 38        | 27              |
| PDM-Open                                          | Centerline         | 54          | 50           | **86**    | **7**           |
| PDM-Closed                                        | Centerline         | **92**      | **93**       | 42        | 91              |
| PDM-Hybrid                                        | Centerline         | **92**      | **93**       | 84        | 96              |
| *Log Replay*                                      | *GT*               | *80*        | *94*         | *100*     | -               |

*Open-loop reimplementation of Urban Driver

## Getting started

### 1. Installation
To install tuPlan Garage, please follow these steps:
- setup the nuPlan dataset ([described here](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html)) and install the nuPlan devkit ([see here](https://nuplan-devkit.readthedocs.io/en/latest/installation.html))
- download tuPlan Garage and move inside the folder
```
git clone https://github.com/marlon31415/tuplan_garage && cd tuplan_garage
```
- make sure the environment you created when installing the nuplan-devkit is activated
```
conda activate nuplan
```
- install the local tuplan_garage as a pip package
```
pip install -e .
```
- add the following environment variable to your `~/.bashrc`
```
NUPLAN_DEVKIT_ROOT="$HOME/nuplan-devkit/"
```

### 2. Training
When running a training, you have to add the `hydra.searchpath` for the `tuplan_garage` correctly.
Note: since hydra does not yet support appending to lists ([see here](https://github.com/facebookresearch/hydra/issues/1547)), you have to add the original searchpaths in the override.
Training scripts can be run with the scripts found in `/scripts/training/`.
Before training from an already existing cache, please check [this](https://github.com/motional/nuplan-devkit/issues/128) issue.
You can find our trained models [here](https://drive.google.com/drive/folders/1LLdunqyvQQuBuknzmf7KMIJiA2grLYB2?usp=sharing).

### 3. Evaluation
Same as for the training, when running an evaluation, you have to add the `hydra.searchpath` for the `tuplan_garage` correctly.
The example below runs an evaluation of the `pdm_closed_planner` on the `val14_split`, both of which are part of the tuplan_garage
```
python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
+simulation=closed_loop_nonreactive_agents \
planner=pdm_closed_planner \
scenario_filter=val14_split \
scenario_builder=nuplan \
hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
```
You can find exemplary shells scripts in `/scripts/simulation/`

## Citation

## Other resources <a name="otherresources"></a>
- [SceneMotion](https://www.arxiv.org/abs/2408.01537) | [future-motion Github](https://github.com/KIT-MRT/future-motion)

## Acknowledgements
This repository builds upon the work [tuPlan Garage](https://github.com/autonomousvision/tuplan_garage)
