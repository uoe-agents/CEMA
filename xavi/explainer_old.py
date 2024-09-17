""" Old implementation of the XAVIAgent class that was used for the AANMAS 2024 paper. """

import numpy as np
import pandas as pd
import logging
import time
import igp2 as ip
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import Counter

from sklearn.linear_model import LogisticRegression

from xavi.features import Features
from xavi.util import fill_missing_actions, truncate_observations, \
    to_state_trajectory, find_join_index, Observations, get_coefficient_significance, \
    find_optimal_rollout_in_subset, split_by_query, list_startswith, find_matching_rollout, \
    Item, XAVITree, XAVIAction, overwrite_predictions
from xavi.matching import ActionMatching, ActionGroup, ActionSegment
from xavi.query import Query, QueryType
from xavi.language import Language

logger = logging.getLogger(__name__)


class XAVIAgent(ip.MCTSAgent):
    """ Generate new rollouts and save results after MCTS in the observation tau time before. """

    def __init__(self,
                 cf_n_trajectories: int = 3,
                 cf_n_simulations: int = 15,
                 cf_max_depth: int = 5,
                 tau_limits: Tuple[float, float] = (1., 5.),
                 time_limits: Tuple[float, float] = (5., 5.),
                 alpha: float = 0.1,
                 **kwargs):
        """ Create a new XAVIAgent.

        Args:
            tau: The interval to roll back for counterfactual generation. By default set to FPS.
            cf_n_trajectories: Number of maximum trajectories to generate with A*.
            cf_n_simulations: Number of MCTS simulations to run for counterfactual generation.
            cf_d_max: Maximum MCTS search depth for counterfactual simulations.
            tau_limits: Lower and upper bounds on the distance of tau from t_action.
            time_limits: The maximal amount of time to look back in the past and future.
            alpha: The distribution smoothing weight

        Keyword Args: See arguments of parent-class MCTSAgent.
        """

        super(XAVIAgent, self).__init__(**kwargs)

        self._n_trajectories = cf_n_trajectories
        self._tau_limits = np.array(tau_limits)
        self._time_limits = np.array(time_limits)
        self._scenario_map = kwargs["scenario_map"]
        self._alpha = alpha

        self._cf_n_simulations = kwargs.get("cf_n_simulations", cf_n_simulations)
        self._cf_max_depth = kwargs.get("cf_max_depth", cf_max_depth)
        self._cf_goal_probabilities_dict = {"tau": None, "t_action": None}
        self._cf_observations_dict = {"tau": None, "t_action": None}
        self._cf_dataset_dict = {"tau": None, "t_action": None}
        mcts_params = {"scenario_map": self._scenario_map,
                       "n_simulations": self._cf_n_simulations,
                       "max_depth": self._cf_max_depth,
                       "reward": self.mcts.reward,
                       "store_results": "all",
                       "tree_type": XAVITree,
                       "action_type": XAVIAction,
                       "trajectory_agents": False}
        self._cf_mcts_dict = {
            "tau": ip.MCTS(**mcts_params),
            "t_action": ip.MCTS(**mcts_params),
        }

        self._features = Features(self._scenario_map)
        self._matching = ActionMatching(scenario_map=self._scenario_map)
        self._language = Language()

        self._previous_queries = []
        self._user_query = None
        self._current_t = None
        self._observations_segments = None
        self._total_trajectories = None
        self._mcts_results_buffer = []

    def update_plan(self, observation: ip.Observation):
        super(XAVIAgent, self).update_plan(observation)

        # Retrieve maneuvers and macro actions for non-ego vehicles
        for rollout in self.mcts.results:
            last_node = rollout.leaf
            if last_node.key == ("Root",):
                logger.debug("MCTS node terminated during Root.")
                continue
            for agent_id, agent in last_node.run_result.agents.items():
                if isinstance(agent, ip.TrajectoryAgent):
                    plan = self.goal_probabilities[agent_id].trajectory_to_plan(*rollout.samples[agent_id])
                    fill_missing_actions(agent.trajectory_cl, plan)
                agent.trajectory_cl.calculate_path_and_velocity()

        current_t = int(self.observations[self.agent_id][0].states[-1].time)
        self._mcts_results_buffer.append((current_t, self.mcts.results))

    def explain_actions(self, user_query: Query = None):
        """ Explain the behaviour of the ego considering the last tau time-steps and the future predicted actions.

        Args:
            user_query: The parsed query of the user.

        Returns: A natural language explanation of the query, and the causes that generated the sentence.
        """
        t_start = time.time()

        self._user_query = user_query
        self._user_query.fps = self.fps
        self._user_query.tau_limits = self.tau_limits

        self._current_t = int(self.observations[self.agent_id][0].states[-1].time)
        if self._observations_segments is None or user_query.t_query != self._current_t:
            self._observations_segments = {}
            for aid, obs in self.observations.items():
                self._observations_segments[aid] = self._matching.action_segmentation(obs[0])
        self._total_trajectories = self._get_total_trajectories()

        # Determine timing information of the query.
        try:
            self.query.get_tau(
                self._current_t, self._scenario_map,
                self.total_observations, self._mcts_results_buffer)
            logger.info(f"t_action is {self.query.t_action}, tau is {self.query.tau}")
        except ValueError as ve:
            logger.exception(str(ve), exc_info=ve)
            return str(ve), None

        logger.info(f"Running explanation for {self.query}.")

        if self.query.type == QueryType.WHAT:
            causes = self._explain_what()
        elif self.query.type in [QueryType.WHY, QueryType.WHY_NOT]:
            causes = self._explain_why()
        elif self.query.type == QueryType.WHAT_IF:
            causes = self._explain_whatif()
        else:
            raise ValueError(f"Unknown query type: {self.query.type}")

        self._previous_queries.append(self._user_query)
        logger.debug(f"Runtime: {time.time() - t_start}")
        return causes

    def _teleological_causes(self,
                             tau_dataset: List[Item],
                             t_action_dataset: List[Item]) \
            -> Tuple[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]:
        """ Generate final causes for the queried action.

        Args:
            tau_dataset: Dataset of items for past counterfactuals.
            t_action_dataset: Dataset of items for present counterfactuals.

        Returns:
            Dataframe of reward components with the absolute and relative changes for each component
            and probability of occurrences.
        """

        def get_causes(ref_items, alt_items):
            def get_values(items, comp):
                r = [item.reward.reward_components[comp] for item in items
                     if item.reward.reward_components[comp] is not None]
                p = len(r) / len(items) if r else 0.0
                r = np.sum(r) / len(r) if r else 0.0
                return r, p
            diffs = {}
            for component in self._reward.reward_components:
                r_qp, p_qp = get_values(ref_items, component)
                r_qnp, p_qnp = get_values(alt_items, component)
                diff = r_qp - r_qnp
                diffs[component] = (r_qp, r_qnp,
                                    diff if not np.isnan(diff) else 0.0,
                                    p_qp, p_qnp)
            columns = ["r_qp", "r_qnp", "d_abs", "p_r_qp", "p_r_qnp"]
            df = pd.DataFrame.from_dict(diffs, orient="index", columns=columns)
            return df.sort_values(ascending=False, by="d_abs", key=abs)

        query_present, query_not_present = split_by_query(tau_dataset)
        tau_rewards = pd.DataFrame([item.reward.reward_components for item in tau_dataset])
        tau_rewards["query_present"] = [item.query_present for item in tau_dataset]
        tau_causes = get_causes(query_present, query_not_present)

        query_present, query_not_present = split_by_query(t_action_dataset)
        t_action_rewards = pd.DataFrame([item.reward.reward_components for item in t_action_dataset])
        t_action_rewards["query_present"] = [item.query_present for item in t_action_dataset]
        t_action_causes = get_causes(query_present, query_not_present)
        return (tau_causes, tau_rewards), (t_action_causes, t_action_rewards)

    def _mechanistic_causes(self,
                            tau_dataset: List[Item] = None,
                            t_action_dataset: List[Item] = None) \
            -> Tuple[Optional[pd.DataFrame], pd.DataFrame, Tuple[Optional[LogisticRegression], LogisticRegression]]:
        """ Generate efficient causes for the queried action.

        Args:
            tau_dataset: Counterfactual items starting from timestep tau.
            t_action_dataset: Counterfactual items starting from timestep t_action.

        Returns:
            Dataframes for past and future causes with causal effect size, and optionally the linear regression models
        """

        def process_dataset(dataset: List[Item], t_slice: Tuple[Optional[int], Optional[int]]):
            xs_, ys_ = [], []
            agent_id = self.agent_id
            if self.query.type in [QueryType.WHY, QueryType.WHY_NOT]:
                agent_id = self.query.agent_id
            if dataset is not None:
                for item in dataset:
                    xs_.append(self._features.to_features(
                        agent_id, item, self.query, t_slice=t_slice))
                    ys_.append(item.query_present)
            return xs_, ys_

        def get_ranking(xs_, ys_):
            X_, y_, model_, coefs_ = None, None, None, None
            if xs_ and ys_:
                X_, y_ = self._features.binarise(xs_, ys_)
                model_ = LogisticRegression().fit(X_, y_)
                coefs_ = get_coefficient_significance(X_, y_, model_)
            return X_, y_, model_, coefs_

        if tau_dataset is None and t_action_dataset is None:
            return None, None

        # Get past and future datasets by truncating trajectories to relevant length and getting features
        xs_past, ys_past = process_dataset(tau_dataset, (self.query.tau, self.query.t_action))
        xs_future, ys_future = process_dataset(t_action_dataset, (self.query.t_action, None))

        # Run a logistic regression classifier
        X_past, y_past, model_past, coefs_past = get_ranking(xs_past, ys_past)
        X_future, y_future, model_future, coefs_future = get_ranking(xs_future, ys_future)

        # Get coefficients using K-fold cross validation
        return coefs_past, coefs_future, (X_past, y_past, model_past), (X_future, y_future, model_future)

    # ---------Explanation generation functions---------------

    def _explain_what(self) -> List[ActionGroup]:
        """ Generate an explanation to a what query. Involves looking up the trajectory segment at T and
        returning a feature set of it. We assume for the future that non-egos follow their MAP-prediction for
        goal and trajectory.

        Returns:
            An action group of the executed action at the user given time point.
        """
        logger.info("Generating a what explanation.")
        if self.query.agent_id is None:
            logger.warning(f"No Agent ID given for what-query. Falling back to ego ID.")
            self.query.agent_id = self.agent_id

        trajectory = self.total_observations[self.query.agent_id][0]
        segments = self._matching.action_segmentation(trajectory)
        grouped_segments = ActionGroup.group_by_maneuver(segments)
        if self.query.t_action is None:
            return grouped_segments

        start_t = self.query.t_action
        if start_t >= len(trajectory):
            logger.warning(f"Total trajectory for Agent {self.query.agent_id} is not "
                           f"long enough for query! Falling back to final timestep.")
            start_t = len(trajectory) - 1
        return [seg for seg in grouped_segments if seg.start <= start_t <= seg.end]

    def _explain_why(self) -> Tuple[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """ Generate a why explanation.

        Returns: The final and past and future efficient causes for the query.
        """
        logger.info("Generating a why or why-not explanation.")
        if self.query.tau is None:
            self._get_counterfactuals(["t_action"])
            tau = None
        else:
            self._get_counterfactuals(["t_action", "tau"])
            tau = list(self.cf_datasets["tau"].values())

        assert self._cf_dataset_dict["t_action"] is not None, f"Missing counterfactual dataset."
        t_action = list(self.cf_datasets["t_action"].values())

        final_causes = self._teleological_causes(tau, t_action)
        efficient_causes = self._mechanistic_causes(tau, t_action)

        return final_causes, efficient_causes

    def _explain_whatif(self) -> Tuple[ActionGroup, pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """ Generate an explanation to a whatif query.
        Labels trajectories in query and finding optimal one among them, then compares the optimal one with
        factual optimal one regarding their reward components to extract final causes.

        Returns:
            An action group of the executed action at the user given time point.
            The final causes to explain that action and the efficient causes of the explanation.
        """
        # Generate a new dataset, output the most likely action. Split dataset by the cf action.
        logger.info("Generating what-if explanation.")
        self._get_counterfactuals(["t_action"])
        cf_items, f_items = split_by_query(list(self.cf_datasets["t_action"].values()))

        # Find the maximum q-value and the corresponding action sequence of the ego for ea
        cf_optimal_rollout = find_optimal_rollout_in_subset(cf_items, self._reward.factors)
        f_optimal_rollout = find_optimal_rollout_in_subset(f_items, self._reward.factors)

        # Retrieve ego's action plan in the counterfactual case
        cf_ego_agent = cf_optimal_rollout.leaf.run_result.agents[self.agent_id]
        cf_optimal_trajectory = cf_ego_agent.trajectory_cl
        observed_segments = self._matching.action_segmentation(cf_optimal_trajectory)
        observed_grouped_segments = ActionGroup.group_by_maneuver(observed_segments)
        cf_action_group = [g for g in observed_grouped_segments
                           if g.start <= self.query.t_action <= g.end][0]

        # Check if the change in the non-ego action did not change the observed ego actions
        if cf_optimal_rollout.trace == self.mcts.results.optimal_trace:
            logger.info("Ego actions remain the same even in counterfactual case.")

        # Determine the actual optimal maneuver and rewards
        f_optimal_items = [it for it in f_items if
                           list_startswith(f_optimal_rollout.trace, it.rollout.trace)]
        cf_optimal_items = [it for it in cf_items if
                            list_startswith(cf_optimal_rollout.trace, it.rollout.trace)]

        # compare reward initial and reward counter
        final_causes = self._teleological_causes(cf_optimal_items, f_optimal_items)
        efficient_causes = self._mechanistic_causes(None, cf_optimal_items + f_optimal_items)

        return cf_action_group, final_causes, efficient_causes

    # ---------Counterfactual rollout generation---------------

    def _get_counterfactuals(self, times: List[str]):
        """ Get observations from tau time steps before, and call MCTS from that joint state.

        Args:
            times: The time reference points at which timestep to run MCTS from. Either tau or t_action for now.
        """
        logger.info("Generating counterfactual rollouts.")

        for time_reference in times:
            self._generate_counterfactuals_from_time(time_reference)

    def _generate_counterfactuals_from_time(self, time_reference: str):
        """ Generate a counterfactual dataset from the time reference point.

         Args:
             time_reference: Either tau or t_action.
         """
        t = getattr(self.query, time_reference)
        truncated_observations, previous_frame = truncate_observations(self.observations, t)
        self._cf_observations_dict[time_reference] = truncated_observations

        logger.debug(f"Generating counterfactuals at {time_reference} ({t})")
        if previous_frame:
            mcts = self._cf_mcts_dict[time_reference]
            goal_probabilities = self._cf_goal_probabilities_dict[time_reference]

            previous_query = self._previous_queries[-1] if self._previous_queries else None
            if self.cf_datasets[time_reference] is None or not previous_query or \
                    previous_query.t_query != self.query.t_query or \
                    previous_query.type != self.query.type:
                observation = ip.Observation(previous_frame, self._scenario_map)
                goal_probabilities = self._get_goals_probabilities(observation, previous_frame)

                self._generate_rollouts(previous_frame,
                                        truncated_observations,
                                        goal_probabilities,
                                        mcts)
                self._cf_goal_probabilities_dict[time_reference] = goal_probabilities
            ref_t = self.query.t_action if time_reference == "t_action" else self.query.tau
            self._cf_dataset_dict[time_reference] = self._get_dataset(
                mcts.results, goal_probabilities, truncated_observations, ref_t)

    def _get_goals_probabilities(self,
                                 observation: ip.Observation,
                                 previous_frame: Dict[int, ip.AgentState]) \
            -> Dict[int, ip.GoalsProbabilities]:
        """ Create a new data structure to store goal probability computations.

        Args:
            observation: the observation of the environment for which to generate the data structure
        """
        goals = self.get_goals(observation)
        return {aid: ip.GoalsProbabilities(goals) for aid in previous_frame.keys() if aid != self.agent_id}

    def _generate_rollouts(self,
                           frame: Dict[int, ip.AgentState],
                           observations: Observations,
                           goal_probabilities: Dict[int, ip.GoalsProbabilities],
                           mcts: ip.MCTS):
        """ Runs MCTS to generate a new sequence of macro actions to execute using previous observations.

        Args:
            frame: Observation of the env tau time steps back.
            observations: Dictionary of observation history.
            goal_probabilities: Dictionary of predictions for each non-ego agent.
        """
        visible_region = ip.Circle(frame[self.agent_id].position, self.view_radius)

        # Increase number of trajectories to generate
        n_trajectories = self._goal_recognition._n_trajectories
        self._goal_recognition._n_trajectories = self._n_trajectories

        for agent_id in frame:
            if agent_id == self.agent_id:
                continue

            # Generate all possible trajectories for non-egos from tau time steps back
            gps = goal_probabilities[agent_id]
            self._goal_recognition.update_goals_probabilities(
                goals_probabilities=gps,
                observed_trajectory=observations[agent_id][0],
                agent_id=agent_id,
                frame_ini=observations[agent_id][1],
                frame=frame,
                visible_region=visible_region)

            # For past queries, use existing most recent goal probabilities
            latest_predictions = self.mcts_results_buffer[-1][1].predictions
            if self.query.tense == "past" and agent_id in latest_predictions:
                overwrite_predictions(gps, latest_predictions[agent_id])

            # Set the probabilities equal for each goal and trajectory
            #  to make sure we can sample all counterfactual scenarios
            gps.add_smoothing(self._alpha, uniform_goals=True)

        # Reset the number of trajectories for goal generation
        self._goal_recognition._n_trajectories = n_trajectories

        # Run MCTS search for counterfactual simulations while storing run results
        agents_metadata = {aid: state.metadata for aid, state in frame.items()}
        mcts.search(
            agent_id=self.agent_id,
            goal=self.goal,
            frame=frame,
            meta=agents_metadata,
            predictions=goal_probabilities)

    def _get_dataset(self,
                     mcts_results: ip.AllMCTSResult,
                     goal_probabilities: Dict[int, ip.GoalsProbabilities],
                     observations: Observations,
                     reference_t: int) \
            -> Dict[int, Item]:
        """ Return dataset recording states, boolean feature, and reward

         Args:
             mcts_results: MCTS results class to convert to a dataset.
             goal_probabilities: Predictions for non-ego vehicles.
             observations: The observations that preceded the planning step.
             reference_t: The time of the start of the counterfactual simulation.
         """
        dataset = {}
        for m, rollout in enumerate(mcts_results):
            trajectories = {}
            r = []
            last_node = rollout.leaf
            trajectory_queried_agent = None

            # save trajectories of each agent
            for agent_id, agent in last_node.run_result.agents.items():
                trajectory = ip.StateTrajectory(self.fps)
                observed_trajectory = observations[agent_id][0]
                trajectory.extend(observed_trajectory, reload_path=False)
                sim_trajectory = agent.trajectory_cl.slice(1, None)

                # Retrieve maneuvers and macro actions for non-ego vehicles
                if isinstance(agent, ip.TrafficAgent):
                    plan = goal_probabilities[agent_id].trajectory_to_plan(*rollout.samples[agent_id])
                    fill_missing_actions(sim_trajectory, plan)

                if agent_id == self.query.agent_id:
                    trajectory_queried_agent = sim_trajectory

                trajectory.extend(sim_trajectory, reload_path=True)
                trajectories[agent_id] = trajectory

            # save reward for each component
            for last_action, reward_value, in last_node.reward_results.items():
                if last_action == rollout.trace[-1]:
                    r = reward_value[-1]

            # Slice the trajectory according to the tense in case of multiply actions in query exist in a trajectory
            sliced_trajectory = self.query.slice_segment_trajectory(
                trajectory_queried_agent, self._current_t, present_ref_t=reference_t)
            query_factual = self.query.factual if not self.query.all_factual and self.query.exclusive else None
            y = self._matching.action_matching(
                self.query.action, sliced_trajectory, query_factual)
            if self.query.negative:
                y = not y

            data_set_m = Item(trajectories, y, r, rollout)
            dataset[m] = data_set_m

        logger.debug('Dataset generation done.')
        return dataset

    def _get_total_trajectories(self) -> Tuple[Observations, List]:
        """ Return the optimal predicted trajectories for all agents. This would be the optimal MCTS plan for
        the ego and the MAP predictions for non-ego agents.

         Returns:
             Optimal predicted trajectories and their initial state as Observations.
             The reward
         """
        # Use observations until current time
        ret = {}
        map_predictions = {aid: p.map_prediction() for aid, p in self.goal_probabilities.items()}

        for agent_id in self.observations:
            trajectory = ip.StateTrajectory(self.fps)
            trajectory.extend(self.observations[agent_id][0], reload_path=False)

            # Find simulated trajectory that matches best with observations and predictions
            if agent_id == self.agent_id:
                optimal_rollouts = self.mcts.results.optimal_rollouts
                matching_rollout = find_matching_rollout(optimal_rollouts, map_predictions)
                if matching_rollout is None:
                    c = Counter([tuple(r.samples.items()) for r in optimal_rollouts])
                    most_common = c.most_common(1)[0][0]
                    matching_rollout = find_matching_rollout(optimal_rollouts, dict(most_common))
                last_node = matching_rollout.tree[matching_rollout.trace[:-1]]
                agent = last_node.run_result.agents[agent_id]
                sim_trajectory = agent.trajectory_cl
            else:
                goal, sim_trajectory = map_predictions[agent_id]
                plan = self.goal_probabilities[agent_id].trajectory_to_plan(goal, sim_trajectory)
                sim_trajectory = to_state_trajectory(sim_trajectory, plan, self.fps)

            # Truncate trajectory to time step nearest to the final observation of the agent
            join_index = find_join_index(self._scenario_map, trajectory, sim_trajectory)
            sim_trajectory = sim_trajectory.slice(int(join_index), None)

            trajectory.extend(sim_trajectory, reload_path=True, reset_times=True)
            ret[agent_id] = (trajectory, sim_trajectory.states[0])
        return ret

    # -------------Field access properties-------------------

    @property
    def cf_datasets(self) -> Dict[str, Optional[Dict[int, Item]]]:
        """ The most recently generated set of counterfactuals rolled back to tau. """
        return self._cf_dataset_dict

    @property
    def cf_goals_probabilities(self) -> Dict[str, Optional[Dict[int, ip.GoalsProbabilities]]]:
        """ The goal and trajectory probabilities inferred from tau time steps ago. """
        return self._cf_goal_probabilities_dict

    @property
    def cf_n_simulations(self) -> int:
        """ The number of rollouts to perform in counterfactual MCTS. """
        return self._cf_n_simulations

    @property
    def cf_mcts(self) -> Dict[str, ip.MCTS]:
        """ MCTS planners for each time reference point. """
        return self._cf_mcts_dict

    @property
    def total_observations(self) -> Observations:
        """ Returns the factual observations extended with the most optimal predicted trajectory for all agents. """
        return self._total_trajectories

    @property
    def observation_segmentations(self) -> Dict[int, List[ActionSegment]]:
        """ Segmentations of the observed trajectories for each vehicle. """
        return self._observations_segments

    @property
    def tau_limits(self) -> np.ndarray:
        """ The lower and upper bound of the distance of tau from t_action. """
        return self._tau_limits

    @property
    def query(self) -> Query:
        """ The most recently asked user query. """
        return self._user_query

    @property
    def alpha(self) -> float:
        """ The smoothing weight for goal recognition. """
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        assert isinstance(value, float) and value >= 0., f"Invalid new alpha {value}."
        self._alpha = value

    @property
    def mcts_results_buffer(self) -> List[Tuple[int, ip.AllMCTSResult]]:
        """ The results buffer for all previous MCTS planning steps. """
        return self._mcts_results_buffer
