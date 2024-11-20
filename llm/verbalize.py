""" This module contains functions to verbalize the simulation state, 
including the road layout and the agent positions. """

import logging
from typing import Dict
import numpy as np
import igp2 as ip
from igp2.opendrive.elements.geometry import ramer_douglas


logger = logging.getLogger(__name__)


def scenario(config: Dict) -> str:
    """ Verbalize the scenario configuration. 
    
    Args:
        config: The scenario configuration dictionary.
    """
    ret =   "The following are road rules and regulations:\n"
    ret += f"  Maximum speed limit: {config['scenario']['max_speed']} m/s\n"
    ret +=  "  Driving side: right-hand traffic\n"
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
        metadata: Whether to add metadata before the road layout description ([False]/True).

    Returns:
        A string describing the road layout.         
    """

    if kwargs.get("metadata", False):
        ret = """The following is metadata to parse the elements of the road layout.

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
    else:
        ret = ""

    lane_links = kwargs.get("intersection_links", False)
    if not lane_links:
        ret += "  Connections are written as 'incoming road id->connecting road id'."
    else:
        ret += "  Connections are written as 'incoming road id:lane id->connecting road id:lane id'."
    
    ret += "\n\n"
    ret += "The road layout consists of the following elements:\n\n"

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

    return ret


def frame(scenario_map: ip.Map, frame: Dict[int, ip.AgentState], **kwargs) -> str:
    """ Verbalize a frame of the simulation state. 
    
    Args:
        scenario_map: The road layout of the scenario.
        frame: Dictionary mapping agent IDs to agent states

    Keyword Args:
        TODO

    Returns:
        A string describing the frame of the simulation state.
    """
    raise NotImplementedError


def causes(cause) -> str:
    """ Verbalize a collection of CEMA causes. """
    raise NotImplementedError
