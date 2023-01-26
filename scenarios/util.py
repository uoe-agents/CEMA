import logging
import sys
import os
from typing import Dict, List

import logging
import igp2 as ip
import numpy as np
import argparse
import json
from shapely.geometry import Polygon

from xavi import Query

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", metavar="S", type=int, help="The ID of the scenario to execute.")
    parser.add_argument("--seed", type=int, help="Random seed of the simulation.")
    parser.add_argument("--fps", type=int, help="Framerate of the simulation.")
    parser.add_argument("--config_path", type=str, help="Path to a scenario configuration file.")
    parser.add_argument("--query_path", type=str, help="Path to load a query.")
    parser.add_argument("--save_causes", action="store_true", default=False,
                        help="Whether to pickle the causes for each query.")
    parser.add_argument("--save_agent", action="store_true", default=False,
                        help="Whether to pickle the agent for each query.")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Whether to display plots of the simulation.")
    return parser.parse_args()


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", metavar="S", type=int, default=1,
                        help="The number of the scenario to execute.")
    parser.add_argument("query", metavar="Q", type=int, default=0,
                        help="The index of the query to evaluate in the given scenario.")
    parser.add_argument("--size", action="store_true", help="Whether to run a size robustness evaluation.")
    parser.add_argument("--sampling", action="store_true", help="Whether to run a sampling robustness evaluation.")
    return parser.parse_args()


def setup_xavi_logging():
    # Add %(asctime)s  for time
    log_formatter = logging.Formatter("[%(threadName)-10.10s:%(name)-20.20s] [%(levelname)-6.6s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    logging.getLogger("igp2.velocitysmoother").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("PIL").setLevel(logging.INFO)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)


def load_config(args):
    if "scenario" in args:
        path = os.path.join("scenarios", "configs", f"scenario{args.scenario}.json")
    elif "config_path" in args:
        path = args.config_path
    else:
        raise ValueError("No scenario was specified!")
    return json.load(open(path, "r"))


def parse_query(args) -> List[Query]:
    """ Returns a list of parsed queries. """
    if args.query_path is not None:
        path = args.query_path
    elif args.scenario is not None:
        path = os.path.join("scenarios", "queries", f"query_scenario{args.scenario}.json")
    else:
        raise ValueError("No query was specified!")
    try:
        queries = json.load(open(path, "r"))
        if isinstance(queries, dict):
            queries = queries[f"s{args.scenario}"]
        return [Query(**query_dict) for query_dict in queries]
    except FileNotFoundError as e:
        logger.exception(str(e), exc_info=e)
        return []


def generate_random_frame(layout: ip.Map, config) -> Dict[int, ip.AgentState]:
    """ Generate a new frame with randomised spawns and velocities for each vehicle.

    Args:
        layout: The current road layout
        config: Dictionary of properties describing agent spawns.

    Returns:
        A new randomly generated frame
    """
    ret = {}
    for agent in config["agents"]:
        spawn_box = ip.Box(**agent["spawn"]["box"])
        spawn_vel = agent["spawn"]["velocity"]

        poly = Polygon(spawn_box.boundary)
        best_lane = None
        max_overlap = 0.0
        for road in layout.roads.values():
            for lane_section in road.lanes.lane_sections:
                for lane in lane_section.all_lanes:
                    overlap = lane.boundary.intersection(poly).area
                    if overlap > max_overlap:
                        best_lane = lane
                        max_overlap = overlap

        intersections = list(best_lane.midline.intersection(poly).coords)
        start_d = best_lane.distance_at(intersections[0])
        end_d = best_lane.distance_at(intersections[1])
        if start_d > end_d:
            start_d, end_d = end_d, start_d
        position_d = (end_d - start_d) * np.random.random() + start_d

        spawn_position = np.array(best_lane.point_at(position_d))
        spawn_heading = best_lane.get_heading_at(position_d)

        vel = (spawn_vel[1] - spawn_vel[0]) * np.random.random() + spawn_vel[0]
        vel = min(vel, ip.Maneuver.MAX_SPEED)
        spawn_velocity = vel * np.array([np.cos(spawn_heading), np.sin(spawn_heading)])

        agent_metadata = ip.AgentMetadata(**agent["metadata"]) if "metadata" in agent \
            else ip.AgentMetadata(**ip.AgentMetadata.CAR_DEFAULT)

        ret[agent["id"]] = ip.AgentState(time=0,
                                         position=spawn_position,
                                         velocity=spawn_velocity,
                                         acceleration=np.array([0.0, 0.0]),
                                         heading=spawn_heading,
                                         metadata=agent_metadata)

    return ret
