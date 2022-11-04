from typing import List
from collections import Counter
import igp2 as ip
import numpy as np


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


def most_common(lst: list):
    """ Return the most common element in a list. """
    data = Counter(lst)
    return max(lst, key=data.get) if lst else None
