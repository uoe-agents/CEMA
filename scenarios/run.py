import logging
import os.path
import pickle
import sys
from typing import List

import igp2 as ip
import pygame
import xavi
import numpy as np
import random
import matplotlib.pyplot as plt

from util import generate_random_frame, setup_xavi_logging, parse_args, \
    load_config, parse_query, to_ma_list

logger = logging.Logger(__name__)


def main():
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

    random.seed(seed)
    np.random.seed(seed)
    np.seterr(divide="ignore")

    ip.Maneuver.MAX_SPEED = config["scenario"].get("max_speed", 10)
    ip.Trajectory.VELOCITY_STOP = config["scenario"].get("velocity_stop", 0.1)
    ip.SwitchLane.TARGET_SWITCH_LENGTH = config["scenario"].get("target_switch_length", 20)
    ip.GiveWay.MAX_ONCOMING_VEHICLE_DIST = config["scenario"].get("max_oncoming_vehicle_dist", 100)
    xavi.XAVITree.STOP_CHANCE = config["scenario"].get("stop_chance", 1.0)

    frame = generate_random_frame(scenario_map, config)

    fps = args.fps if args.fps else config["scenario"]["fps"] if "fps" in config["scenario"] else 20
    try:
        if args.carla:
            map_name = os.path.split(config["scenario"]["map_path"])[1][:-5]
            simulation = ip.carla.CarlaSim(xodr=config["scenario"]["map_path"], map_name=map_name)
        else:
            simulation = xavi.Simulation(scenario_map, fps)

        xavi_agent = None
        for agent in config["agents"]:
            base_agent = {"agent_id": agent["id"], "initial_state": frame[agent["id"]],
                          "goal": ip.BoxGoal(ip.Box(**agent["goal"]["box"])), "fps": fps}
            mcts_agent = {"scenario_map": scenario_map,
                          "cost_factors": agent.get("cost_factors", None),
                          "view_radius": agent.get("view_radius", None),
                          "kinematic": not args.carla,
                          "velocity_smoother": agent.get("velocity_smoother", None),
                          "goal_recognition": agent.get("goal_recognition", None)}
            if agent["type"] == "MCTSAgent":
                agent = ip.MCTSAgent(**base_agent, **mcts_agent, **agent["mcts"])
                rolename = "ego"
            elif agent["type"] == "XAVIAgent":
                agent = xavi.XAVIAgent(**base_agent, **mcts_agent, **agent["explainer"], **agent["mcts"])
                xavi_agent = agent
                rolename = "ego"
            elif agent["type"] == "TrafficAgent":
                if "macro_actions" in agent:
                    base_agent["macro_actions"] = to_ma_list(agent["macro_actions"], agent["id"], frame, scenario_map)
                agent = ip.TrafficAgent(**base_agent)
                rolename = agent.get("rolename", "car")
            else:
                raise ValueError(f"Unsupported agent type {agent['type']}")

            simulation.add_agent(agent, rolename=rolename)

        if args.carla:
            result = run_carla_simulation(xavi_agent, simulation, args, queries, config, output_path)
        else:
            result = run_simple_simulation(xavi_agent, simulation, args, queries, config, output_path)
    except Exception as e:
        logger.exception(msg=str(e), exc_info=e)
        result = False
    finally:
        del simulation
    return result


def run_carla_simulation(xavi_agent, simulation, args, queries, config, output_path) -> bool:
    visualiser = ip.carla.Visualiser(simulation)
    world = None
    try:
        clock, world, display, controller = visualiser.initialize()
        for t in range(config["scenario"]["max_steps"]):
            done = visualiser.step(clock, world, display, controller)
            if not args.sim_only:
                explain(queries, xavi_agent, t, output_path, args)
            if done:
                break
    finally:
        if world is not None:
            world.destroy()
            del world
        pygame.quit()
    return True


def run_simple_simulation(xavi_agent, simulation, args, queries, config, output_path) -> bool:
    for t in range(config["scenario"]["max_steps"]):
        simulation.step()
        if t % 20 == 0 and args.plot:
            xavi.plot_simulation(simulation, debug=False)
            plt.show()
        if not args.sim_only:
            explain(queries, xavi_agent, t, output_path, args)
    return True


def explain(queries: List[xavi.Query], xavi_agent: xavi.XAVIAgent, t: int, output_path: str, args):
    for query in queries:
        if t > 0 and t == query.t_query:
            _, causes = xavi_agent.explain_actions(query)

            if args.save_agent:
                file_path = os.path.join(output_path, f"agent_t{t}_m{query.type}.pkl")
                pickle.dump(xavi_agent, open(file_path, "wb"))

            if args.save_causes:
                file_path = os.path.join(output_path, f"q_t{t}_m{query.type}.pkl")
                pickle.dump(causes, open(file_path, "wb"))


if __name__ == '__main__':
    sys.exit(main())
