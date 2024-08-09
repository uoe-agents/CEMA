from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import igp2 as ip
import xavi
import gofi


class OXAVITree(gofi.OTree):
    """ Overwrite the original MCTS tree to disable give-way with some chance. """
    STOP_CHANCE = 1.0

    def select_action(self, node: ip.Node) -> ip.MCTSAction:
        action = super(OXAVITree, self).select_action(node)
        if action.macro_action_type == ip.Exit:
            give_way_stop = np.random.random() >= 1.0 - OXAVITree.STOP_CHANCE
            action.ma_args["stop"] = give_way_stop
        return action


@dataclass
class OItem(xavi.Item):
    """ Extend the original item class to include occluded factors. """
    occluded_factor: gofi.OccludedFactor


def get_occluded_trajectory(
        agent: ip.Agent,
        start_observation: ip.Observation,
        end_state: ip.AgentState) -> Tuple[ip.StateTrajectory, List[ip.MacroAction]]:
    """ Find the trajectory of the occluded agent by using A* to generate an open-loop trajectory.

    Args:
        agent: The occluded agent.
        start_observation: The observation at the start of the occlusion.
        end_state: The state at the end of the occlusion.
    """
    start_state = start_observation.frame[agent.agent_id]

    goal = ip.PointGoal(end_state.position, threshold=0.1)
    if start_state.speed < ip.Stop.STOP_VELOCITY and goal.reached(start_state.position):
        stop_duration = (end_state.time - start_state.time) / agent.fps
        config = ip.MacroActionConfig({'type': 'Stop', "stop_duration": stop_duration})
        ma = ip.MacroActionFactory.create(
            config, agent.agent_id, start_observation.frame, start_observation.scenario_map)
        plan = [ma]
        trajectory = ma.get_trajectory()
    else:
        trajectory, plan = ip.AStar().search(
            agent.agent_id, start_observation.frame, goal, start_observation.scenario_map)
        trajectory, plan = trajectory[0], plan[0]
    trajectory = ip.StateTrajectory.from_velocity_trajectory(trajectory, fps=agent.fps)
    return trajectory, plan


def fill_missing_actions(
        trajectory: ip.StateTrajectory,
        plan: List[ip.MacroAction] = None,
        agent: ip.Agent = None,
        start_observation: ip.Observation = None
):
    """ Infer maneuver and macro action data for each point in the trajectory and modify trajectory in-place.

    Args:
        trajectory: The StateTrajectory with missing macro and maneuver information.
        plan: The macro action plan that generated the StateTrajectory.
        agent: The agent for which to fill the missing actions.
        start_observation: The observation at the start of the trajectory for trajectory agents
    """
    assert plan is not None or agent is not None, "Either plan or agent must be provided."

    if agent is not None and plan is None:
        _, plan = get_occluded_trajectory(agent, start_observation, agent.trajectory_cl.states[-1])

    ma_man_list = []
    points = None
    for ma in plan:
        for man in ma.maneuvers:
            ma_man_list.extend([(repr(ma), repr(man))] * len(man.trajectory))
            points = man.trajectory.path if points is None else \
                np.append(points, man.trajectory.path, axis=0)
    for state in trajectory:
        nearest_idx = np.argmin(np.linalg.norm(points - state.position, axis=1))
        state.macro_action, state.maneuver = ma_man_list[nearest_idx]
