import itertools
from copy import copy
from typing import List, Tuple, Dict
from collections import Counter
import igp2 as ip
import numpy as np
import logging

logger = logging.getLogger(__name__)

Observations = Dict[int, Tuple[ip.StateTrajectory, ip.AgentState]]


def to_state_trajectory(
        trajectory: ip.VelocityTrajectory,
        plan: List[ip.MacroAction] = None,
        fps: int = 20) -> ip.StateTrajectory:
    """ Convert a velocity trajectory to a state trajectory.

     Args:
         trajectory: The trajectory to convert
         plan: Optional plan to fill in missing action and maneuver information
         fps: Optional frame rate of the trajectory
     """
    trajectory = ip.StateTrajectory.from_velocity_trajectory(trajectory, fps)
    fill_missing_actions(trajectory, plan)
    return trajectory


def truncate_observations(observations: Observations, tau: int) -> (Observations, Dict[int, ip.AgentState]):
    """ Truncate all observations from the end with tau timesteps.

     Args:
         observations: The observations to truncate.
         tau: The number of steps to remove from the end.
     """
    truncated_obs = {}
    previous_frame = {}
    for agent_id, observation in observations.items():
        frame = observation[1]
        len_states = len(observation[0].states)
        if len_states > tau:
            truncated_obs[agent_id] = (observation[0].slice(0, len_states - tau), frame)
            previous_frame[agent_id] = observation[0].states[len_states - tau - 1]
        else:
            logger.warning(f"Agent {agent_id}: tau({tau}) >= number of observations({len_states}.)")
    return truncated_obs, previous_frame


def fill_missing_actions(trajectory: ip.StateTrajectory, plan: List[ip.MacroAction]):
    """ Infer maneuver and macro action data for each point in the trajectory and modify trajectory in-place. """
    ma_man_list = []
    points = None
    for ma in plan:
        for man in ma.maneuvers:
            ma_man_list.extend([(ma.__repr__(), man.__repr__())] * len(man.trajectory))
            points = man.trajectory.path if points is None else \
                np.append(points, man.trajectory.path, axis=0)
    for state in trajectory:
        nearest_idx = np.argmin(np.linalg.norm(points - state.position, axis=1))
        state.macro_action, state.maneuver = ma_man_list[nearest_idx]


def fix_initial_state(trajectory: ip.StateTrajectory):
    """ The initial frame is often missing macro and maneuver information due to the planning flow of IGP2.
    This function fills in the missing information using the second state. """
    if len(trajectory.states) > 1 and \
            trajectory.states[0].time == 0 and \
            trajectory.states[0].macro_action == trajectory.states[0].maneuver is None:
        trajectory.states[0].macro_action = copy(trajectory.states[1].macro_action)
        trajectory.states[0].maneuver = copy(trajectory.states[1].maneuver)


def most_common(lst: list):
    """ Return the most common element in a list. """
    data = Counter(lst)
    return max(lst, key=data.get) if lst else None


def product_dict(**kwargs):
    """ Take a cross-product of a dictionary of lists. """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))
