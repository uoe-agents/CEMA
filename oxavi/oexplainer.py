from typing import Tuple
import logging
import time

import gofi
import xavi
from xavi.util import *
import numpy as np
import igp2 as ip
from oxavi.ofeatures import OFeatures

logger = logging.getLogger(__name__)


class OXAVITree(gofi.OTree):
    """ Overwrite the original MCTS tree to disable give-way with some chance. """
    STOP_CHANCE = 1.0

    def select_action(self, node: ip.Node) -> ip.MCTSAction:
        action = super(OXAVITree, self).select_action(node)
        if action.macro_action_type == ip.Exit:
            give_way_stop = np.random.random() >= 1.0 - OXAVITree.STOP_CHANCE
            action.ma_args["stop"] = give_way_stop
        return action


class OXAVIAgent(gofi.GOFIAgent, xavi.XAVIAgent):
    """ Generate new rollouts and save results after MCTS in the observation tau time before. """

    def __init__(self,
                 cf_n_trajectories: int = 3,
                 cf_n_simulations: int = 15,
                 cf_max_depth: int = 5,
                 tau_limits: Tuple[float, float] = (1., 5.),
                 time_limits: Tuple[float, float] = (5., 5.),
                 alpha: float = 0.1,
                 **kwargs):
        """
        Initialise a new OXAVIAgent. The arguments to this agent are the same as xavi.XAVIAgent.
        """
        super(OXAVIAgent, self).__init__(
            cf_n_trajectories, cf_n_simulations, cf_max_depth, tau_limits, time_limits, alpha, **kwargs)

        mcts_params = {"scenario_map": self._scenario_map,
                       "n_simulations": self._cf_n_simulations,
                       "max_depth": self._cf_max_depth,
                       "reward": self.mcts.reward,
                       "store_results": "all",
                       "tree_type": OXAVITree,
                       "action_type": xavi.XAVIAction,
                       "trajectory_agents": False}
        self._cf_mcts_dict = {
            "tau": gofi.OMCTS(**mcts_params),
            "t_action": gofi.OMCTS(**mcts_params),
        }

        self._features = OFeatures(self._scenario_map)
        self._matching = xavi.ActionMatching(scenario_map=self._scenario_map)
        self._language = xavi.Language()

    def __repr__(self):
        return f"OXAVIAgent(ID={self.agent_id})"

    def __str__(self):
        return repr(self)

    def update_plan(self, observation: ip.Observation):
        super(OXAVIAgent, self).update_plan(observation)

        # Retrieve maneuvers and macro actions for non-ego vehicles
        for rollout in self.mcts.results:
            last_node = rollout.leaf
            if len(last_node.key) == 2:
                logger.debug("MCTS node terminated during Super, Root/OF.")
                continue
            for agent_id, agent in last_node.run_result.agents.items():
                agent.trajectory_cl.calculate_path_and_velocity()

                # Fill in missing macro action and maneuver information
                if isinstance(agent, ip.TrajectoryAgent):
                    if agent_id in self.goal_probabilities:
                        plan = self.goal_probabilities[agent_id].trajectory_to_plan(*rollout.samples[agent_id])
                        fill_missing_actions(agent.trajectory_cl, plan)
                    else:
                        goal = ip.PointGoal(agent.state.position, threshold=0.1)
                        start_frame = {aid: ag.trajectory_cl.states[0] for aid, ag in
                                       last_node.run_result.agents.items()}
                        if agent.state.speed < ip.Stop.STOP_VELOCITY and goal.reached(agent.state.position):
                            config = ip.MacroActionConfig({'type': 'Stop', "duration": agent.trajectory_cl.duration})
                            plan = [ip.MacroActionFactory.create(config, agent_id, start_frame, self._scenario_map)]
                        else:
                            _, plan = ip.AStar().search(agent_id, start_frame, goal, self._scenario_map)
                            plan = plan[0]
                        fill_missing_actions(agent.trajectory_cl, plan)

        current_t = int(self.observations[self.agent_id][0].states[-1].time)
        self._mcts_results_buffer.append((current_t, self.mcts.results))

    def _get_goals_probabilities(self,
                                 observation: ip.Observation,
                                 previous_frame: Dict[int, ip.AgentState]) -> Dict[int, gofi.OGoalsProbabilities]:
        goals = self.get_goals(observation)
        occluded_factors = self.get_occluded_factors(observation)
        gps = {}
        for aid in previous_frame.keys():
            if aid != self.agent_id:
                gps[aid] = gofi.OGoalsProbabilities(
                    goals, occluded_factors, occluded_factors_priors=self._occluded_factors_prior)
        return gps
