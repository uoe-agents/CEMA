import numpy as np

import igp2 as ip
from typing import Dict, List
from dataclasses import dataclass
from igp2.agents.mcts_agent import MCTSAgent


@dataclass
class feature_set:
    states: List[ip.AgentState]
    feature_exist: List[bool]
    reward: float


class rollout_generation(ip.MCTSAgent):
    """ generate new rollouts and save results after MCTS in the observation tau time before """

    def __init__(self,
                 agent_id: int,
                 initial_state: ip.AgentState,
                 t_update: float,
                 scenario_map: ip.Map,
                 goal: ip.Goal = None,
                 view_radius: float = 50.0,
                 fps: int = 20,
                 cost_factors: Dict[str, float] = None,
                 reward_factors: Dict[str, float] = None,
                 n_simulations: int = 5,
                 max_depth: int = 3,
                 store_results: str = 'final'):

        super(rollout_generation, self).__init__(agent_id, initial_state, t_update, scenario_map, goal, view_radius,
                                                 fps,
                                                 cost_factors, reward_factors, n_simulations, max_depth, store_results)

        self._mcts = ip.MCTS(scenario_map, n_simulations=n_simulations, max_depth=max_depth,
                             store_results=store_results)

        self._scenario_map = scenario_map
        self._tau = fps
        self._state_buffer = []

    def state_buffer(self, frame: Dict[int, ip.AgentState]):
        """
        Args:
            frame: the current states of agents tau time before.
        """
        self._state_buffer.append(frame)
        if len(self._state_buffer) >= self._tau:
            previous_state = self._state_buffer[-int(self._tau)]

            observation = ip.Observation(frame, self._scenario_map)
            self.next_action(observation)
            dataset = self.get_results(observation)

    def next_action(self, observation: ip.Observation) -> ip.Action:
        action = super(rollout_generation, self).next_action(observation)

        return action

    @staticmethod
    def get_outcome_y(states: List[ip.AgentState]) -> List[bool]:
        """ Return boolean value for each predefined feature """
        """
        Args:
            states: the state of ego.
        """
        # only define the first 7 features, should be added more
        y = {
            'accelerating': bool(0),
            'decelerating': bool(0),
            'maintaining': bool(0),
            'relative slower': bool(0),
            'relative faster': bool(0),
            'same speed': bool(0),
            'ever stop': bool(0)
        }
        acc = []
        for state in states:
            acc.append(state.acceleration)
        if np.average(acc) > 0:
            y['accelerating'] = 1
        elif np.average(acc) == 0:
            y['maintaining'] = 1
        else:
            y['decelerating'] = 1

        return list(y.values())

    def get_results(self, observation: ip.Observation) -> Dict[int, feature_set]:
        """ Return dataset recording states, boolean feature, and reward """
        """
        Args:
            observation: the observation of the environment tau time before.
        """
        dataset = {}
        super(rollout_generation, self).update_plan(observation)
        mcts_results = self._mcts.results
        if isinstance(mcts_results, ip.MCTSResult):
            mcts_results = ip.AllMCTSResult()
            mcts_results.add_data(self._mcts.results)

        # save results of rollout generation for explanation
        for m, rollout in enumerate(mcts_results):
            states = []

            # add state of ego
            inx = 0
            r = None
            for node_key, state_value in rollout.tree.tree.items():
                states.append(state_value.state[0])
                # save the reward for the node next to root
                if inx == 1:
                    r = state_value.q_values[0]
                inx += 1
            data_set_m = feature_set(states, self.get_outcome_y(states), r)
            dataset[m] = data_set_m

        return dataset
