import itertools
from copy import copy
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict
import igp2 as ip
import numpy as np
import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold, cross_validate

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
    """ Truncate all observations from the end down to timestep tau.

     Args:
         observations: The observations to truncate.
         tau: The number of steps to remove from the end.
     """
    truncated_obs = {}
    previous_frame = {}
    for agent_id, observation in observations.items():
        frame = observation[1]
        if tau > 0:
            truncated_obs[agent_id] = (observation[0].slice(0, tau), frame)
            previous_frame[agent_id] = observation[0].states[min(len(observation[0]), tau) - 1]
        else:
            raise ValueError(f"Agent {agent_id}: tau({tau}) <= 0.")
    return truncated_obs, previous_frame


def fill_missing_actions(trajectory: ip.StateTrajectory, plan: List[ip.MacroAction]):
    """ Infer maneuver and macro action data for each point in the trajectory and modify trajectory in-place.

    Args:
        trajectory: The StateTrajectory with missing macro and maneuver information.
        plan: The macro action plan that generated the StateTrajectory.
    """
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
    This function fills in the missing information using the second state.

    Args:
        trajectory: The StateTrajectory whose first state is missing macro action or maneuver information.
    """
    if len(trajectory.states) > 1 and \
            trajectory.states[0].time == 0 and \
            trajectory.states[0].macro_action is None and \
            trajectory.states[0].maneuver is None:
        trajectory.states[0].macro_action = copy(trajectory.states[1].macro_action)
        trajectory.states[0].maneuver = copy(trajectory.states[1].maneuver)


def find_join_index(
        scenario_map: ip.Map,
        init_trajectory: ip.StateTrajectory,
        joining_trajectory: ip.StateTrajectory) -> int:
    """ Determine the best point to join two trajectories.

    Args:
        scenario_map: The road layout.
        init_trajectory: The starting trajectory.
        joining_trajectory: The trajectory to join with the starting trajectory.

    Returns:
        The index at which to join the two trajectories.
    """
    last_point = init_trajectory.states[-1].position
    last_heading = init_trajectory.states[-1].heading
    last_point_lane = scenario_map.best_lane_at(last_point, last_heading)
    closest_idx = np.argmin(np.linalg.norm(joining_trajectory.path - last_point, axis=1))
    closest_point = joining_trajectory.path[closest_idx]
    closest_heading = joining_trajectory.heading[closest_idx]
    closest_point_lane = scenario_map.best_lane_at(closest_point, closest_heading)
    if last_point_lane != closest_point_lane:
        logger.warning(f"Last observed point is on different lane then closest predicted point.")
    else:
        while closest_idx < len(joining_trajectory):
            closest_point = joining_trajectory.path[closest_idx]
            closest_heading = joining_trajectory.heading[closest_idx]
            closest_point_lane = scenario_map.best_lane_at(closest_point, closest_heading)
            d_last = last_point_lane.distance_at(last_point)
            d_closest = closest_point_lane.distance_at(closest_point)
            if last_point_lane != closest_point_lane:
                d_closest += last_point_lane.length

            diff = d_last - d_closest
            if diff < -ip.Maneuver.POINT_SPACING:
                break
            elif diff < 0 and joining_trajectory.velocity[closest_idx] < ip.Trajectory.VELOCITY_STOP:
                break
            closest_idx += 1
        else:
            raise ValueError(f"Predicted trajectory has no valid point!")
    return int(closest_idx)


def get_coefficient_significance(data: pd.DataFrame,
                                 labels: np.ndarray,
                                 model: LogisticRegression,
                                 folds: int = 5,
                                 repeats: int = 7) -> pd.DataFrame:
    """ Run K-fold cross validation on the model to
    retrieve an error bound on the feature significance on the coefficients.

    Args:
        data: The dataset underlying the model.
        labels: The target labels of the data.
        model: The model validate.
        folds: K, the number of folds
        repeats: Number of random repeats for K-fold CV

    Returns: A Pandas dataframe with validate feature significance.
    """
    cv = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=0)
    cv_model = cross_validate(model, data, labels,
                              cv=cv, return_estimator=True, n_jobs=2)
    coefs = pd.DataFrame(
        [
            np.squeeze(est.coef_) * data.iloc[train_idx].std(axis=0)
            for est, (train_idx, _) in zip(cv_model["estimator"], cv.split(data, labels))
            if hasattr(est, "coef_")
        ],
        columns=data.columns,
    )
    return coefs


def find_optimal_rollout_in_subset(subset: List["Item"],
                                   reward_factors: Dict[str, float]) -> "Item":
    """ Find the most optimal action from a subset of MTCS rollouts based on average rewards.

    Returns: The rollout with the maximum average reward.
    """
    rollouts = defaultdict(list)
    for m, item in enumerate(subset):
        rollout = item.rollout
        sum_reward = 0.0
        for component, factor in reward_factors.items():
            reward = item.reward.reward_components[component] \
                if item.reward.reward_components[component] is not None else 0.0
            sum_reward += factor * reward
        rollouts[rollout.trace].append((rollout, sum_reward))
    means = {trace: np.mean(list(zip(*items))[1]) for trace, items in rollouts.items()}
    max_mean_trace = max(means, key=means.get)
    return max(rollouts[max_mean_trace], key=lambda x: x[1])[0]


def split_by_query(dataset: List["Item"]) -> (List["Item"], List["Item"]):
    """ Split a dataset by the presence of a the query.

    Returns: A pair as (query_present, query_not_present).
    """
    query_present = []
    query_not_present = []
    for item in dataset:
        if item.query_present:
            query_present.append(item)
        else:
            query_not_present.append(item)
    return query_present, query_not_present


def most_common(lst: list, **kwargs):
    """ Return the most common element in a list. """
    in_roundabout = kwargs.get("in_roundabout", None)
    data = list(sorted(Counter(lst).items(), key=lambda x: -x[1])) if lst else None
    if data is None:
        raise ValueError(f"No list given.")
    if in_roundabout and any(["continue" not in k.lower() for k, cnt in data]):
        filtered = [k for k, cnt in data if "continue" not in k.lower()]
        return filtered[0]
    else:
        return data[0][0]


def list_startswith(list1: list, list2: list) -> bool:
    """ Compare two lists. If the lengths are equal simply return equality using ==.
    If lengths are unequal, then check whether the first one has the same element as the second one. """
    len1, len2 = len(list1), len(list2)
    if len1 >= len2:
        return list1[:len2] == list2
    return False


def find_matching_rollout(rollouts: List[ip.MCTSResult], samples: Dict[int, ip.GoalsProbabilities]) \
        -> Optional[ip.MCTSResult]:
    for rollout in rollouts:
        for aid, prediction in samples.items():
            if rollout.samples[aid] != prediction:
                break
        else:
            return rollout


def product_dict(**kwargs):
    """ Take a cross-product of a dictionary of lists. """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))
