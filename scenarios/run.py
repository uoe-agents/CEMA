import logging
import igp2 as ip
import xavi
import numpy as np
import random

import matplotlib.pyplot as plt
from util import generate_random_frame, setup_xavi_logging, parse_args, load_config

logger = logging.Logger(__name__)

if __name__ == '__main__':
    setup_xavi_logging()

    args = parse_args()
    logger.debug(args)
    config = load_config(args)

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

    rollout = None
    for agent in config.agents:
        base_agent = {"agent_id": agent.id, "initial_state": frame[agent.id],
                      "goal": ip.BoxGoal(ip.Box(**agent.goal.box)), "fps": fps}
        if agent.type == "MCTSAgent":
            simulation.add_agent(ip.MCTSAgent(scenario_map=scenario_map,
                                              cost_factors=agent.cost_factors,
                                              view_radius=agent.view_radius,
                                              kinematic=agent.kinematic,
                                              **base_agent,
                                              **agent.mcts))
            rollout = xavi.rollout_generation(scenario_map=scenario_map,
                                              cost_factors=agent.cost_factors,
                                              view_radius=agent.view_radius,
                                              **base_agent,
                                              **agent.mcts)
        elif agent.type == "TrafficAgent":
            simulation.add_agent(ip.TrafficAgent(**base_agent))

    for t in range(config.scenario.max_steps):
        rollout.state_buffer(simulation.state)
        simulation.step()
        if t % 20 == 0:
            simulation.plot(debug=False)
            plt.show()
