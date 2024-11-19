""" This module contains functions to verbalize the simulation state, 
including the road layout and the agent positions. """

import logging
import igp2 as ip


logger = logging.getLogger(__name__)


def road_layout(scenario_map: ip.Map, **kwargs) -> str:
    """ Verbalize the road layout.

    Args:
        scenario_map: The map of the scenario.

    Keyword Args:
        TODO

    Returns:
        A string describing the road layout.         
    """
    raise NotImplementedError


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
