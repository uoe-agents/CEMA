""" You can run this file to generate explanations for the first scenario from IGP2.
 It does not support interactive explanation generation, rather it creates an explanations for every possible
 counterfactual right after a planning step has been completed. The explanations will be printed to the screen.

 Should you wish to modify how explanation generation works, the main method responsible for generating
 the explanations is XAVIAgent.explain_all_actions()
 """

import random

import matplotlib.pyplot as plt
import numpy as np
import igp2 as ip
import xavi

from scenarios.scripts.util import setup_xavi_logging, generate_random_frame, parse_args

if __name__ == '__main__':
    setup_xavi_logging()

    args = parse_args()

    # Set run parameters here
    seed = args.seed
    max_speed = args.max_speed
    ego_id = 0
    n_simulations = args.n_sim
    fps = args.fps  # Simulator frequency
    T = args.period  # MCTS update period

    random.seed(seed)
    np.random.seed(seed)
    np.seterr(divide="ignore")
    ip.Maneuver.MAX_SPEED = max_speed

    # Set randomised spawn parameters here
    ego_spawn_box = ip.Box(np.array([28.25, -30.0]), 3.5, 10, 0.0)
    ego_vel_range = (5.0, max_speed)
    veh1_spawn_box = ip.Box(np.array([-20.0, -5.25]), 10, 3.5, 0.0)
    veh1_vel_range = (5.0, max_speed)
    veh2_spawn_box = ip.Box(np.array([100.0, -1.75]), 10, 3.5, 0.0)
    veh2_vel_range = (5.0, max_speed)

    # Vehicle goals
    goals = {
        ego_id: ip.BoxGoal(ip.Box(np.array([65, -5.25]), 5, 3.5, 0.0)),
        ego_id + 1: ip.BoxGoal(ip.Box(np.array([28.25, 10.0]), 3.5, 5, 0.0)),
        ego_id + 2: ip.BoxGoal(ip.Box(np.array([5.0, -1.75]), 5, 3.5, 0.0))
    }

    scenario_path = "scenarios/maps/scenario2.xodr"
    scenario_map = ip.Map.parse_from_opendrive(scenario_path)

    frame = generate_random_frame(ego_id,
                                  scenario_map,
                                  [(ego_spawn_box, ego_vel_range),
                                   (veh1_spawn_box, veh1_vel_range),
                                   (veh2_spawn_box, veh2_vel_range)])

    cost_factors = {"time": 3., "velocity": 0.0, "acceleration": 0.1, "jerk": 0., "heading": 0.0,
                    "angular_velocity": 0.5, "angular_acceleration": 0.1, "curvature": 0.0, "safety": 0.}
    reward_factors = {"time": 1.0, "jerk": -1.0, "angular_velocity": -0.1, "curvature": 0.0}
    simulation = xavi.Simulation(scenario_map, fps)

    agents = {}
    agents_meta = ip.AgentMetadata.default_meta_frame(frame)
    for aid in frame.keys():
        goal = goals[aid]

        if aid == ego_id:
            agents[aid] = ip.MCTSAgent(agent_id=aid,
                                       initial_state=frame[aid],
                                       t_update=T,
                                       scenario_map=scenario_map,
                                       goal=goal,
                                       cost_factors=cost_factors,
                                       reward_factors=reward_factors,
                                       fps=fps,
                                       n_simulations=n_simulations,
                                       view_radius=100,
                                       store_results="all",
                                       kinematic=True)
            simulation.add_agent(agents[aid])
        else:
            agents[aid] = ip.TrafficAgent(aid, frame[aid], goal, fps)
            simulation.add_agent(agents[aid])

    for t in range(500):
        simulation.step()
        if t > 0 and t % 40 == 0:
            simulation.plot(debug=True)
            plt.show()
