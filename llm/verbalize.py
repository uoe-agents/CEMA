""" This module contains functions to verbalize the simulation state,
including the road layout and the agent positions. """

import logging
from typing import Dict, Tuple

import re
import numpy as np
import igp2 as ip
from igp2.opendrive.elements.geometry import ramer_douglas

import xavi
import llm.util as util


logger = logging.getLogger(__name__)

ROAD_LAYOUT_PRETEXT = """The following is metadata to parse the elements of the road layout.

Coordinate system:
  We are using a planar 2D coordinate system.
  Coordinates are in units of meters and written as (x, y).
  Angles are in radians in the range [-pi, pi].

Roads and Lanes:
  The road layout consists of roads which are given a unique numeric ID.
  Roads are made up of lanes which are identified as 'road ID:lane ID'.
  Lanes are oriented in the direction of the road midline.
  Lanes are divided into left and right lanes.
  Right lanes have a negative ID and left lanes have a positive ID.
  Lanes are 3.5 meters wide.

Intersections:
  Roads are connected at intersections.
  Intersections are made up of connections between incoming and connecting roads.
"""


def scenario(
        config: Dict,
        scenario_map: ip.Map,
        observations: Dict[int, Tuple[ip.StateTrajectory, ip.AgentState]],
        query_obj: xavi.Query,
        **kwargs) -> str:
    """ Utility function to verbalize the entire scenario.

    Args:
        config: The scenario configuration dictionary.
        scenario_map: The road layout of the scenario.
        observations: The ego agent's observations of the scenario.
        query_obj: The user query to prompt.

    Keyword Args:
        add_rules: Whether to add the scenario rules ([True]/False).
        add_road_layout: Whether to add the road layout description ([True]/False).
        add_agents: Whether to add the agent descriptions ([True]/False).
        add_query: Whether to add the user query ([True]/False).
        add_metadata: Whether to add metadata before descriptions ([False]/True).
        lanes: Whether to add lane descriptions ([True]/False).
        intersections: Whether to add intersection descriptions ([True]/False).
        intersection_links: Whether to add intersection lane links (True/[False]).
        resolution: The resolution of the road midline (default=0.01).

    Returns:
        A string describing the scenario configuration, road layout, and simulation state.
    """
    ret = ""
    if kwargs.get("add_rules", True):
        ret += rules(config) + "\n\n"
    if kwargs.get("add_road_layout", True):
        ret += road_layout(scenario_map, **kwargs) + "\n\n"
    if kwargs.get("add_agents", True):
        ret += agents(observations, **kwargs) + "\n\n"
    if kwargs.get("add_query", True):
        ret += query(query_obj, **kwargs) + "\n\n"

    return ret[:-2]  # Remove the last two newlines


def rules(config: Dict) -> str:
    """ Verbalize the scenario rules.

    Args:
        config: The scenario configuration dictionary.
    """
    ret =   "The following are road rules and regulations:\n"
    ret += f"  Maximum speed limit: {config['scenario']['max_speed']} m/s\n"
    ret +=  "  Driving side: right-hand traffic"
    return ret


def road_layout(scenario_map: ip.Map, **kwargs) -> str:
    """ Verbalize the road layout.

    Args:
        scenario_map: The road layout of the scenario.

    Keyword Args:
        lanes: Whether to add lane descriptions ([True]/False).
        intersections: Whether to add intersection descriptions ([True]/False).
        intersection_links: Whether to add intersection lane links (True/[False]).
        resolution: The resolution of the road midline (default=0.01).
        add_metadata: Whether to add metadata before the road layout description ([False]/True).

    Returns:
        A string describing the road layout.
    """
    ret = ""

    add_metadata = kwargs.get("add_metadata", False)
    if add_metadata:
        ret += ROAD_LAYOUT_PRETEXT
        lane_links = kwargs.get("intersection_links", False)
        if not lane_links:
            ret += "  Connections are written as 'incoming road id->connecting road id'."
        else:
            ret += "  Connections are written as 'incoming road id:lane id->connecting road id:lane id'."
        ret += "\n\n"


    ret += "The road layout consists of the following elements:"
    ret += "\n\n"

    # Describe roads
    for rid, road in scenario_map.roads.items():
        if not road.drivable:
            continue

        ret += f"Road {rid}:\n"
        ret += f"  Length: {road.length} m\n"

        midline = ramer_douglas(np.array(road.midline.coords), dist=kwargs.get("resolution", 0.02))
        midline = [(x, y) for x, y in np.round(midline, 2)]
        ret += f"  Midline coordinates: {midline}\n"

        left_lanes = [lane for lane in road.lanes.lane_sections[0].left_lanes
                      if lane.type == ip.LaneTypes.DRIVING]
        right_lanes = [lane for lane in road.lanes.lane_sections[0].right_lanes
                       if lane.type == ip.LaneTypes.DRIVING]

        # Describe lanes
        if kwargs.get("lanes", True):
            if left_lanes:
                ret +=  "  Left lanes:\n"
                for lane in left_lanes:
                    ret += f"    Lane {rid}.{lane.id}.\n"
            if right_lanes:
                ret +=  "  Right lanes:\n"
                for lane in right_lanes:
                    ret += f"    Lane {rid}.{lane.id}.\n"
        ret += "\n"


    # Describe intersections
    if kwargs.get("intersections", True):
        for jid, junction in scenario_map.junctions.items():
            ret += f"Intersection {jid} connections:\n"
            for connection in junction.connections:
                if kwargs.get("intersection_links", False):
                    for lane_link in connection.lane_links:
                        ret += f"  {connection.incoming_road.id}.{lane_link.from_id}"
                        ret += f"->{connection.connecting_road.id}.{lane_link.to_id}\n"
                else:
                    ret += f"  {connection.incoming_road.id}->{connection.connecting_road.id}\n"

    if ret[-1] == "\n":
        ret = ret[:-1]
    return ret


def agents(observations: Dict[int, Tuple[ip.StateTrajectory, ip.AgentState]], **kwargs) -> str:
    """ Verbalize a frame of the simulation state.

    Args:
        observations: The ego agent's observations of the scenario.

    Keyword Args:
        f_subsample: Frequency of subsampling observations. Use this to decrease
             the complexity of the verbalization. ([1]/int).
        rounding: Number of decimal places to round the values to ([2]/int).
        control_signals: List of control signals to include in the verbalization.
            Possible values: ["time", "timesteps", "position", "speed", "acceleration", "heading"].
            Default is all control signals except time.

    Returns:
        A string describing the frame of the simulation state.
    """
    n_agents = len([ss for _, ss in observations.values()])
    ret = f"""The scenario has {n_agents} agents. Agent 0 is the autonomous vehicle (ego vehicle).

The ego vehicle observed the following control signals for each agent:

"""

    f_subsample = kwargs.get("f_subsample", 1)
    rounding = kwargs.get("rounding", 2)
    control_signals = kwargs.get("control_signals",
                                 ["timesteps", "position", "speed", "acceleration", "heading"])
    for agent_id, (trajectory, _) in observations.items():
        ret += f"Agent {agent_id}:\n"

        if f_subsample > 1:
            trajectory = util.subsample_trajectory(trajectory, f_subsample)

        for signal in control_signals:
            if signal == "time":
                ret += f"  Time: {util.ndarray2str(trajectory.times, rounding)}\n"
            elif signal == "timesteps":
                timesteps = np.array([s.time for s in trajectory.states])
                ret += f"  Timesteps: {util.ndarray2str(timesteps)}\n"
            elif signal == "position":
                ret += f"  Position: {util.ndarray2str(trajectory.path, rounding)}\n"
            elif signal == "speed":
                ret += f"  Speed: {util.ndarray2str(trajectory.velocity, rounding)}\n"
            elif signal == "acceleration":
                ret += f"  Acceleration: {util.ndarray2str(trajectory.acceleration, rounding)}\n"
            elif signal == "heading":
                ret += f"  Heading: {util.ndarray2str(trajectory.heading, rounding)}\n"
        ret += "\n"

    return ret[:-1]  # Remove the last newline


def query(query_obj: xavi.Query, **kwargs) -> str:
    """ Verbalize a query for prompting.

    Args:
        query_obj: The query object to verbalize.

    Keyword Args:
        ego_ref: To refer to the ego vehicle as "ego vehicle" or "agent 0" ([True]/False).
        include_t_query: Whether to include the time of the query (True/[False]).
        include_factual: Whether to include factual action if present ([True]/False).

    Returns:
        A string describing the  user query.
    """

    if query_obj.agent_id == 0:
        subject = "the ego vehicle" if kwargs.get("ego_ref", True) else "agent 0"
    else:
        subject = f"agent {query_obj.agent_id}"

    rex = re.compile(r'(?<=[a-z])(?=[A-Z])')
    words = [w.lower() for w in rex.split(query_obj.action)]
    if words[-1] == "junction":
        words = words[:-1]
    action = " ".join(words).lower()

    ret = ""
    if query_obj.type in [xavi.QueryType.WHY, xavi.QueryType.WHY_NOT]:
        if query_obj.tense == "past":
            ret += f"Why did {subject} {'not ' if query_obj.negative else ''}{action}"
        elif query_obj.tense == "present":
            action = util.to_gerund(words)
            ret += f"Why is {subject} {'not ' if query_obj.negative else ''}{action}"
        else:
            ret += f"Why will {subject} {'not ' if query_obj.negative else ''}{action}"

    elif query_obj.type == xavi.QueryType.WHAT_IF:  # What if question
        if query_obj.tense == "past":
            action = util.to_past(words, participle=True)
            ret += f"What if {subject} had {'not ' if query_obj.negative else ''}{action}"
        elif query_obj.tense == "present":
            if not query_obj.negative:
                action = util.to_past(words)
            ret += f"What if {subject} {'did not ' if query_obj.negative else ''}{action}"
        else:
            if not query_obj.negative:
                action = util.to_3rd_person(words)
            ret += f"What if {subject} {'does not ' if query_obj.negative else ''}{action}"

    else:  # What question
        if query_obj.tense == "past":
            ret += f"What did {subject} do"
        elif query_obj.tense == "present":
            ret += f"What is {subject} doing"
        else:
            ret += f"What will {subject} do"

    if kwargs.get("include_factual", False) and query_obj.factual:
        factual = rex.split(query_obj.factual)
        factual[0] = util.to_gerund(factual[0])
        factual = " ".join(factual).lower()
        ret += f" instead of {factual}"

    if kwargs.get("include_t_query", False):
        ret += f" at timestep {query_obj.t_query}"

    return ret + "?"


def causes(cause) -> str:
    """ Verbalize a collection of CEMA causes. """
    raise NotImplementedError
