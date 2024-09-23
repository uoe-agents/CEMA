import logging
import os.path
import pickle
import sys
from typing import List

import igp2 as ip
from igp2.core.config import Configuration
import xavi
import oxavi
import gofi
import numpy as np
import random
import matplotlib.pyplot as plt

from util import generate_random_frame, setup_xavi_logging, parse_args, \
    load_config, parse_query, to_ma_list

logger = logging.Logger(__name__)


def main():

    args = parse_args()

    if not os.path.exists("output"):
        os.mkdir("output")
    output_path = os.path.join("output", f"scenario_{args.scenario}")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    setup_xavi_logging(log_dir=os.path.join(output_path, "logs"), log_name=f"run")

    logger.debug(args)

    config = load_config(args)
    queries = parse_query(args)


    logger.info(args)

    scenario_map = gofi.OMap.parse_from_opendrive(config["scenario"]["map_path"])

    # Get run parameters
    seed = args.seed if args.seed else config["scenario"]["seed"] if "seed" in config["scenario"] else 21

    random.seed(seed)
    np.random.seed(seed)
    np.seterr(divide="ignore")

    ip_config = Configuration()
    ip_config.set_properties(**config["scenario"])
    xavi.XAVITree.STOP_CHANCE = config["scenario"].get("stop_chance", 1.0)
    oxavi.OXAVITree.STOP_CHANCE = config["scenario"].get("stop_chance", 1.0)
    oxavi.OFollowLaneCL.IGNORE_VEHICLE_IN_FRONT_CHANCE = config["scenario"].get("ignore_vehicle_in_front_chance", 0.0)

    frame = generate_random_frame(scenario_map, config)

    fps = args.fps if args.fps else config["scenario"]["fps"] if "fps" in config["scenario"] else 20
    try:
        simulation = gofi.OSimulation(scenario_map, fps)

        for agent_config in config["agents"]:
            agent, rolename = create_agent(agent_config, scenario_map, frame, fps, args)
            simulation.add_agent(agent, rolename=rolename)

        if args.plot:
            xavi.plot_simulation(simulation, debug=False)
            plt.show()

        simulation.step()

        ego = simulation.agents[0]
        ego.mcts.n = 15

        # Run MCTS search for counterfactual simulations while storing run results
        if not os.path.exists(f"output/scenario_{args.scenario}/distribution.pkl"):
            agents_metadata = {aid: state.metadata for aid, state in frame.items()}
            all_deterministic_trajectories = xavi.util.get_deterministic_trajectories(ego.goal_probabilities)
            distribution = xavi.Distribution(ego.goal_probabilities)
            for i, deterministic_trajectories in enumerate(all_deterministic_trajectories):
                logger.info(f"Running deterministic simulations {i + 1}/{len(all_deterministic_trajectories)}")
                ego.mcts.search(
                    agent_id=ego.agent_id,
                    goal=ego.goal,
                    frame=frame,
                    meta=agents_metadata,
                    predictions=deterministic_trajectories)
                goal_trajectories = {aid: (gp.goals_and_types[0], gp.all_trajectories[gp.goals_and_types[0]][0]) 
                                    for aid, gp in deterministic_trajectories.items()}
                probabilities, data, reward_data = xavi.util.get_visit_probabilities(ego.mcts.results)
                distribution.add_distribution(goal_trajectories, probabilities, data, reward_data)
            pickle.dump(distribution, open(f"output/scenario_{args.scenario}/distribution.pkl", "wb"))
        else:
            distribution = pickle.load(open(f"output/scenario_{args.scenario}/distribution.pkl", "rb"))
        dataset = distribution.sample_dataset(30)

        

    except Exception as e:
        logger.exception(msg=str(e), exc_info=e)
        result = False
    finally:
        del simulation
    return result


def create_agent(agent_config, scenario_map, frame, fps, args):
    base_agent = {"agent_id": agent_config["id"], "initial_state": frame[agent_config["id"]],
                  "goal": ip.BoxGoal(ip.Box(**agent_config["goal"]["box"])), "fps": fps}

    mcts_agent = {"scenario_map": scenario_map,
                  "cost_factors": agent_config.get("cost_factors", None),
                  "view_radius": agent_config.get("view_radius", None),
                  "kinematic": not args.carla,
                  "velocity_smoother": agent_config.get("velocity_smoother", None),
                  "goal_recognition": agent_config.get("goal_recognition", None),
                  "stop_goals": agent_config.get("stop_goals", False)}

    agent_type = agent_config["type"]

    if agent_type == "OXAVIAgent":
        mcts_agent["occluded_factors_prior"] = agent_config.get("occluded_factors_prior", 0.1)
        agent = oxavi.OXAVIAgent(**base_agent, **mcts_agent, **agent_config["explainer"], **agent_config["mcts"])
        rolename = "ego"
    elif agent_type == "XAVIAgent":
        agent = xavi.XAVIAgent(**base_agent, **mcts_agent, **agent_config["explainer"], **agent_config["mcts"])
        rolename = "ego"
    elif agent_type in "TrafficAgent":
        if "macro_actions" in agent_config and agent_config["macro_actions"]:
            base_agent["macro_actions"] = to_ma_list(
                agent_config["macro_actions"], agent_config["id"], frame, scenario_map)
        rolename = agent_config.get("rolename", "car")
        agent = ip.TrafficAgent(**base_agent)
    elif agent_type == "OccludedAgent":
        if "macro_actions" in agent_config and agent_config["macro_actions"]:
            base_agent["macro_actions"] = to_ma_list(
                agent_config["macro_actions"], agent_config["id"], frame, scenario_map)
        agent = gofi.OccludedAgent(occlusions=agent_config["occlusions"], **base_agent)
        rolename = agent_config.get("rolename", "occluded")
    elif agent_type == "StaticObject":
        agent = gofi.StaticObject(**base_agent)
        rolename = agent_config.get("rolename", "object")
    else:
        raise ValueError(f"Unsupported agent type {agent_config['type']}")
    return agent, rolename



if __name__ == '__main__':
    sys.exit(main())
