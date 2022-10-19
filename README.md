# XAVI
Explainable Autonomous Vehicle Intelligence


## Configuration file
You can add new scenarios for testing by creating a new configuration file in the scenarios/configs folder. 
The file should be a well-formatted JSON file, that can be read using Python's built-in tools.
The configuration file has the following structure, fields, and types. 

Notes:
1. Fields with an exclamation mark (!) must always be included.
2. Angles are in radians according to the standard unit circle.

```
scenarios/configs/new_config.json
|____scenario: dict
|    |____!map_path: str    The path to the OpenDrive *.xodr file for the road layout
|    |____max_speed: float  The maximum speed allowed in the scenario
|    |____fps: int          Execution frequency of the simulation
|    |____seed: int         Random seed
|    |____max_steps: int    If given, stop executing simulation after this many steps
|____agents: list
|    |____agent*: dict     List of agents to spawn
|    |    |____!id: int        The numerical identifier of the agent
|    |    |____!type: str      The type of the agent. Currently, only MCTSAgent and TrafficAgent is supported
|    |    |____!spawn: dict
|    |    |   |____!box: dict                The area to spawn the vehicle in
|    |    |   |   |____!center: list[float]      The center of the spawn area
|    |    |   |   |____!length: float            The length of the spawn area
|    |    |   |   |____!width: float             The width of the spawn area
|    |    |   |   |____!heading: float           The heading of the spawn area
|    |    |   |____velocity: list[float]     The spawn velocity range for random sampling
|    |    |____!goal: dict
|    |    |   |____!box: dict                The area for the vehicle to reach
|    |    |   |   |____!center: list[float]      The center of the goal area
|    |    |   |   |____!length: float            The length of the goal area
|    |    |   |   |____!width: float             The width of the goal area
|    |    |   |   |____!heading: float           The heading of the goal area
 **The following is for MCTSAgents only**
|    |    |____cost_factors: dict           The cost weighing factors for IGP2 goal recognition
|    |    |    |____time: float                  Time to goal
|    |    |    |____velocity: float              Average velocity
|    |    |    |____acceleration: float          Average acceleration
|    |    |    |____jerk: float                  Average jerk
|    |    |    |____heading: float               Average heading
|    |    |    |____angular_velocity: float      Average angular velocity
|    |    |    |____angular_acceleration: float  Average angular acceleration
|    |    |    |____curvature: float             Average curvature
|    |    |    |____safety: float                Safety of trajectory
|    |    |____mcts: dict                  MCTS parameters
|    |    |    |____t_update: float                Runtime period of MCTS in seconds
|    |    |    |____n_simulations: int             Number of rollouts in MCTS
|    |    |    |____store_results: str             Either 'all' or 'final'. If absent, results are not stored
|    |    |    |____reward_factors: dict  MCTS reward factors
|    |    |    |    |____time: float                    Time to goal
|    |    |    |    |____jerk: float                    Average jerk
|    |    |    |    |____angular_velocity: float        Average angular velocity
|    |    |    |    |____curvature: float               Average curvature
|    |    |    |    |____safety: float                  Safety term. Currently, not used.
|    |    |____view_radius: float         Radius of circle centered on the vehicle, in which it can observe the environment
|    |    |____kinematic: bool            Whether to use a bicycle-model vehicle for simulations or a trajectory state-following vehicle
                                               The latter must be used for CARLA. 
```
