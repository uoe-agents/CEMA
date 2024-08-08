from typing import List

import numpy as np
import igp2 as ip


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
        goal = ip.PointGoal(agent.state.position, threshold=0.1)
        if agent.state.speed < ip.Stop.STOP_VELOCITY and goal.reached(agent.state.position):
            config = ip.MacroActionConfig({'type': 'Stop', "duration": agent.trajectory_cl.duration})
            ma = ip.MacroActionFactory.create(
                config, agent.agent_id, start_observation.frame, start_observation.scenario_map)
            plan = [ma]
        else:
            _, plan = ip.AStar().search(agent.agent_id, start_observation.frame, goal, start_observation.scenario_map)
            plan = plan[0]

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
