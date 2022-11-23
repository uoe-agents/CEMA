import logging
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

    scenario_map = ip.Map.parse_from_opendrive(config.scenario.map_path)

    # Get run parameters
    seed = args.seed if args.seed else config.scenario.seed if "seed" in config.scenario else 21
    fps = args.fps if args.fps else config.scenario.fps if "fps" in config.scenario else 20

    random.seed(seed)
    np.random.seed(seed)
    np.seterr(divide="ignore")

    ip.Maneuver.MAX_SPEED = config.scenario.max_speed

    frame = generate_random_frame(scenario_map, config)

    simulation = xavi.Simulation(scenario_map, fps)

    agents = {}

    xavi_agent = None
    for agent in config.agents:
        base_agent = {"agent_id": agent.id, "initial_state": frame[agent.id],
                      "goal": ip.BoxGoal(ip.Box(**agent.goal.box)), "fps": fps}
        if agent.type == "MCTSAgent":
            agent = ip.MCTSAgent(scenario_map=scenario_map,
                                 cost_factors=agent.cost_factors,
                                 view_radius=agent.view_radius,
                                 kinematic=agent.kinematic,
                                 **base_agent,
                                 **agent.mcts)
        elif agent.type == "XAVIAgent":
            agent = xavi.XAVIAgent(scenario_map=scenario_map,
                                   cost_factors=agent.cost_factors,
                                   view_radius=agent.view_radius,
                                   kinematic=agent.kinematic,
                                   **base_agent,
                                   **agent.explainer,
                                   **agent.mcts)
            xavi_agent = agent
        elif agent.type == "TrafficAgent":
            agent = ip.TrafficAgent(**base_agent)

        simulation.add_agent(agent)

    # Execute simulation for fixed number of time steps
    explanation_generated = False
    for t in range(config.scenario.max_steps):
        simulation.step()
        # if t % 20 == 0:
        #     xavi.plot_simulation(simulation, debug=False)
        #     plt.show()
        for query in queries:
            if t > 0 and t % query.time == 0:  # Use 60 for S1; 75 for S2
                xavi_agent.explain_actions(query)
                explanation_generated = True

        if explanation_generated:
            break

