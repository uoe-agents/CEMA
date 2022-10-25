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


class XAVIAgent(ip.MCTSAgent):
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
                 store_results: str = 'final',
                 kinematic: bool = False):

        super(XAVIAgent, self).__init__(agent_id, initial_state, t_update, scenario_map, goal, view_radius,
                                        fps, cost_factors, reward_factors, n_simulations, max_depth, store_results,
                                        kinematic)

        self._mcts = ip.MCTS(scenario_map, n_simulations=n_simulations, max_depth=max_depth,
                             store_results=store_results)

        self._scenario_map = scenario_map
        self._tau = fps
        self._previous_observations = {}
        self._dataset = {}

    def rollout_generation(self):
        previous_state = {}
        for agent_id, observation in self.observations.items():
            frame = observation[1]
            len_states = len(observation[0].states)
            if len_states > self._tau:
                self._previous_observations[agent_id] = (observation[0].slice(0, len_states - self._tau), frame)
                previous_state[agent_id] = observation[0].states[len_states - self._tau]

        if previous_state:
            previous_observation = ip.Observation(previous_state, self._scenario_map)
            self.get_goals(previous_observation)
            self.update_previous_plan(previous_observation)
            self._dataset = self.get_results()

    def update_previous_plan(self, previous_observation: ip.Observation):
        """ Runs MCTS to generate a new sequence of macro actions to execute using previous observations."""
        frame = previous_observation.frame
        agents_metadata = {aid: state.metadata for aid, state in frame.items()}
        self._goal_probabilities = {aid: ip.GoalsProbabilities(self._goals)
                                    for aid in frame.keys() if aid != self.agent_id}
        visible_region = ip.Circle(frame[self.agent_id].position, self.view_radius)

        for agent_id in frame:
            if agent_id == self.agent_id:
                continue

            self._goal_recognition.update_goals_probabilities(self._goal_probabilities[agent_id],
                                                              self._previous_observations[agent_id][0],
                                                              agent_id, self._previous_observations[agent_id][1], frame,
                                                              visible_region=visible_region)
        self._macro_actions = self._mcts.search(self.agent_id, self.goal, frame,
                                                agents_metadata, self._goal_probabilities)
        self._current_macro_id = 0

    @staticmethod
    def get_outcome_y(states: List[ip.AgentState]) -> List[bool]:
        """ Return boolean value for each predefined feature """
        """
        Args:
            states: the state of all  vehicles.
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

    def get_results(self) -> Dict[int, feature_set]:
        """ Return dataset recording states, boolean feature, and reward """
        dataset = {}
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
