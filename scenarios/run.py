import logging
import os.path
import pickle

import igp2 as ip
import xavi
import numpy as np
import random
import matplotlib.pyplot as plt

from util import generate_random_frame, setup_xavi_logging, parse_args, load_config, parse_query

logger = logging.Logger(__name__)

if __name__ == '__main__':
    setup_xavi_logging()

    args = parse_args()
    logger.debug(args)
    config = load_config(args)
    queries = parse_query(args)

    if not os.path.exists("output"):
        os.mkdir("output")
    output_path = os.path.join("output", f"scenario_{args.scenario}")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    scenario_map = ip.Map.parse_from_opendrive(config["scenario"]["map_path"])

    # Get run parameters
    seed = args.seed if args.seed else config["scenario"]["seed"] if "seed" in config["scenario"] else 21
    fps = args.fps if args.fps else config["scenario"]["fps"] if "fps" in config["scenario"] else 20

    random.seed(seed)
    np.random.seed(seed)
    np.seterr(divide="ignore")

    ip.Maneuver.MAX_SPEED = config["scenario"].get("max_speed", 10)
    ip.Trajectory.VELOCITY_STOP = config["scenario"].get("velocity_stop", 0.1)
    ip.SwitchLane.TARGET_SWITCH_LENGTH = config["scenario"].get("target_switch_length", 20)
    ip.GiveWay.MAX_ONCOMING_VEHICLE_DIST = config["scenario"].get("max_oncoming_vehicle_dist", 100)
    xavi.XAVITree.STOP_CHANCE = config["scenario"].get("stop_chance", 1.0)

    frame = generate_random_frame(scenario_map, config)

    simulation = xavi.Simulation(scenario_map, fps)

    agents = {}

    xavi_agent = None
    for agent in config["agents"]:
        base_agent = {"agent_id": agent["id"], "initial_state": frame[agent["id"]],
                      "goal": ip.BoxGoal(ip.Box(**agent["goal"]["box"])), "fps": fps}
        if agent["type"] == "MCTSAgent":
            agent = ip.MCTSAgent(scenario_map=scenario_map,
                                 cost_factors=agent["cost_factors"],
                                 view_radius=agent["view_radius"],
                                 kinematic=agent["kinematic"],
                                 **base_agent,
                                 **agent["mcts"])
        elif agent["type"] == "XAVIAgent":
            agent = xavi.XAVIAgent(scenario_map=scenario_map,
                                   cost_factors=agent["cost_factors"],
                                   view_radius=agent["view_radius"],
                                   kinematic=agent["kinematic"],
                                   **base_agent,
                                   **agent["explainer"],
                                   **agent["mcts"])
            xavi_agent = agent
        elif agent["type"] == "TrafficAgent":
            agent = ip.TrafficAgent(**base_agent)

        simulation.add_agent(agent)

    # Execute simulation for fixed number of time steps
    for t in range(config["scenario"]["max_steps"]):
        simulation.step()
        if t % 20 == 0 and args.plot:
            xavi.plot_simulation(simulation, debug=False)
            plt.show()
        for query in queries:
            if t > 0 and t == query.t_query:
                if args.plot:
                    xavi.plot_simulation(simulation, debug=False)
                    plt.show()

                _, causes = xavi_agent.explain_actions(query)

                if args.save_causes:
                    file_path = os.path.join(output_path, f"q_t{t}_m{query.type}.pkl")
                    pickle.dump(causes, open(file_path, "wb"))
