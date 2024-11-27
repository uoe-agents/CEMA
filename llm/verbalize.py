""" This module contains functions to verbalize the simulation state,
including the road layout and the agent positions. """

import logging
from typing import Dict, Tuple

import re
import numpy as np
import simplenlg as nlg
import igp2 as ip
from igp2.opendrive.elements.geometry import ramer_douglas

import xavi


logger = logging.getLogger(__name__)
TENSES = {
    "past": nlg.Tense.PAST,
    "present": nlg.Tense.PRESENT,
    "future": nlg.Tense.FUTURE
}

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
        **kwargs) -> str:
    """ Utility function to verbalize the entire scenario.

    Args:
        config: The scenario configuration dictionary.
        scenario_map: The road layout of the scenario.
        observations: The ego agent's observations of the scenario.

    Keyword Args:
        rules: Whether to add the scenario rules ([True]/False).
        road_layout: Whether to add the road layout description ([True]/False).
        agents: Whether to add the agent descriptions ([True]/False).
        lanes: Whether to add lane descriptions ([True]/False).
        intersections: Whether to add intersection descriptions ([True]/False).
        intersection_links: Whether to add intersection lane links (True/[False]).
        resolution: The resolution of the road midline (default=0.01).
        metadata: Whether to add metadata before the road layout description ([False]/True).

    Returns:
        A string describing the scenario configuration, road layout, and simulation state.
    """
    ret = ""
    if kwargs.get("rules", True):
        ret += rules(config) + "\n\n"
    if kwargs.get("road_layout", True):
        ret += road_layout(scenario_map, **kwargs) + "\n\n"
    if kwargs.get("agents", True):
        ret += agents(scenario_map, observations, **kwargs)

    return ret


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
        metadata: Whether to add metadata before the road layout description ([False]/True).

    Returns:
        A string describing the road layout.
    """

    ret = ROAD_LAYOUT_PRETEXT if kwargs.get("metadata", False) else ""

    lane_links = kwargs.get("intersection_links", False)
    if ret:
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


def agents(scenario_map: ip.Map, observations: Dict[int, Tuple[ip.StateTrajectory, ip.AgentState]], **kwargs) -> str:
    """ Verbalize a frame of the simulation state.

    Args:
        scenario_map: The road layout of the scenario.
        observations: The ego agent's observations of the scenario.

    Keyword Args:
        TODO

    Returns:
        A string describing the frame of the simulation state.
    """
    for t, (trajectory, start_state) in observations.items():
        pass


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
    lexicon = nlg.Lexicon().getDefaultLexicon()
    factory = nlg.NLGFactory(lexicon)
    realiser = nlg.Realiser(lexicon)

    # Create an interrogative sentence
    sentence = factory.createClause()
    sentence.setFeature(nlg.Feature.INTERROGATIVE_TYPE, nlg.InterrogativeType.WHY)

    # Get the agent and use it as subject
    if query_obj.agent_id == 0:
        subject = "the ego vehicle" if kwargs.get("ego_ref", True) else "agent 0"
    else:
        subject = f"agent {query_obj.agent_id}"
    sentence.setSubject(subject)

    # Regex to split camel case actions into words
    rex = re.compile(r'(?<=[a-z])(?=[A-Z])')
    action = " ".join(rex.split(query_obj.action)).lower()
    vp = factory.createVerbPhrase()
    vp.setHead(action)
    vp.setNegated(query_obj.negative)
    sentence.addComplement(vp)

    # Add factual action if present and needed
    if kwargs.get("include_factual", True) and query_obj.factual:
        factual = " ".join(rex.split(query_obj.factual)).lower()
        pp = factory.createPrepositionPhrase("instead of")
        factual = factory.createVerbPhrase(factual)
        factual.setFeature(nlg.Feature.FORM, nlg.Form.GERUND)
        pp.addComplement(factual)
        sentence.addComplement(pp)

    # Add query timing information if needed
    if kwargs.get("include_t_query", False):
        sentence.addComplement(f"at timestep {query_obj.t_query}")

    # Set the tense of the sentence
    sentence.setTense(TENSES[query_obj.tense])

    return realiser.realiseSentence(sentence)


def causes(cause) -> str:
    """ Verbalize a collection of CEMA causes. """
    raise NotImplementedError
