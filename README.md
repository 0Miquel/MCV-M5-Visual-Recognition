# MCV-M5-Visual-Recognition

Group 2 project of Visual Recognition at the Master in Computer Vision from UAB.
## Team members: Group 2
* [Josep Bravo](https://github.com/LeBrav) (Email: pepbravo11@gmail.com)
* [Miquel Romero](https://github.com/0Miquel) (Email: miquel.robla@gmail.com)
* [Guillem Martinez](https://github.com/guillem-ms) (Email: guillemmarsan@gmail.com)
## Overleaf report (no edit)
In the following link you can find the visualization latex report of our group 2:
https://www.overleaf.com/read/jhvhctdhpxkp

## Final Presentation
Link for the final presentation M5
https://docs.google.com/presentation/d/1NJRXlq7O_Mx9ZrNx00Bv-KhJWc3BFqJDpFG89rfuhj0/edit?usp=sharing


## Week 1
### Train
CLI command to run a training experiment.
```
cd week1
python main.py --config_name=<config_name> --wandb_name=<wandb_project_name>
```
- ``--config_name``: Name of .yaml file that specifies the config files 
that will be used to build the configuration (e.g. config)
- ``--wandb_name``: Name of the WandB project already created in order to track
the experiment

If no ``wandb_name`` is specified, then it will execute an offline training, which will
not be logged into WandB, in case you are not interested in tracking that run.

### Hyperparameter search (WandB sweep)
CLI command to run a hyperparameter search experiment.
```
cd week1
python main_sweep.py --sweep=<sweep_file> --sweep_count=<n_runs> --wandb_name=<wandb_project_name>
```
- ``--sweep``: Name of the sweep file inside the ``sweeps`` folder to be used (e.g. sweep.yaml)
- ``--sweep_count``: Number of runs to execute
- ``--wandb_name``: Name of the WandB project already created in order to track
the experiment

All the arguments are necessary.

### Power Point Presentation
In the following link you can find the read-only Google Slides presentation for this week:
https://docs.google.com/presentation/d/1whdoAJ6VHrrJ_RWEamB2IAcLW70gui0yw31goTlhJck/edit?usp=sharing

## Week 2 
In order to execute a certain task, change configs/task_NUMTASK.yaml with your desired configuration:
```
cd week2
python task_NUMTASK.py
```

### Power Point Presentation
In the following link you can find the read-only Google Slides presentation for this week:
https://docs.google.com/presentation/d/1NJRXlq7O_Mx9ZrNx00Bv-KhJWc3BFqJDpFG89rfuhj0/edit?usp=sharing

## Week 3
In order to execute a certain task, change configs/task_NUMTASK.yaml with your desired configuration:
```
cd week3
python task_NUMTASK.py
```
### Power Point Presentation
In the following link you can find the read-only Google Slides presentation for this week:
https://docs.google.com/presentation/d/1Wicc6TT8sVbltHl0nESWz1UxX_8PSeO7KW8COBSkgjk/edit?usp=sharing
## Week 4
In order to execute a certain task, change configs/task_NUMTASK.yaml with your desired configuration:
```
cd week4
python task_NUMTASK.py
```
### Power Point Presentation
In the following link you can find the read-only Google Slides presentation for this week:
https://docs.google.com/presentation/d/1NuAMZuHNXXfummQBfeOsSC9dsxs2dcrOulw_pFhopKU/edit?usp=sharing


## Week 5
In order to execute a certain task, change configs/task_NUMTASK.yaml with your desired configuration:
```
cd week5
python task_NUMTASK.py
```

### Power Point Presentation
In the following link you can find the read-only Google Slides presentation for this week:
https://docs.google.com/presentation/d/15Y6BcmU0pF1KHSJPR3W71xmezhZ812SjBFHPi4A4wb4/edit?usp=sharing
