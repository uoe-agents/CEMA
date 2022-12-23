import numpy as np
import pandas as pd
import logging
import igp2 as ip
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression

from xavi.features import Features
from xavi.util import fill_missing_actions, truncate_observations, \
    to_state_trajectory, find_join_index, Observations, get_coefficient_significance, \
    find_optimal_rollout_in_subset, split_by_query, list_startswith
from xavi.matching import ActionMatching, ActionGroup, ActionSegment
from xavi.query import Query, QueryType
from xavi.language import LanguageTemplate
from xavi.plotting import plot_dataframe

logger = logging.getLogger(__name__)


@dataclass
class Item:
    """ Class to store a (counter)factual trajectories, the actions of the ego,
    the rewards, the queried observation generated by MCTS, and the rollout. """
    trajectories: Dict[int, ip.StateTrajectory]
    query_present: bool
    reward: Dict[str, float]
    rollout: ip.MCTSResult


class XAVITree(ip.Tree):
    """ Overwrite the original MCTS tree to disable give-way with some chance. """
    STOP_CHANCE = 1.0

    def select_action(self, node: ip.Node) -> ip.MCTSAction:
        action = super(XAVITree, self).select_action(node)
        if action.macro_action_type == ip.Exit:
            give_way_stop = np.random.random() >= 1.0 - XAVITree.STOP_CHANCE
            action.ma_args["stop"] = give_way_stop
        return action


class XAVIAction(ip.MCTSAction):
    """Overwrite MCTSAction to exclude give-way stop from planning trace. """

    def __repr__(self) -> str:
        stop_val = None
        if "stop" in self.ma_args:
            stop_val = self.ma_args["stop"]
            del self.ma_args["stop"]
        repr = super(XAVIAction, self).__repr__()
        if stop_val is not None:
            self.ma_args["stop"] = stop_val
        return repr


class XAVIAgent(ip.MCTSAgent):
    """ Generate new rollouts and save results after MCTS in the observation tau time before. """

    def __init__(self,
                 cf_n_trajectories: int = 3,
                 cf_n_simulations: int = 15,
                 cf_max_depth: int = 5,
                 tau_limits: Tuple[float, float] = (1., 5.),
                 time_limits: Tuple[float, float] = (5., 5.),
                 **kwargs):
        """ Create a new XAVIAgent.

        Args:
            tau: The interval to roll back for counterfactual generation. By default set to FPS.
            cf_n_trajectories: Number of maximum trajectories to generate with A*.
            cf_n_simulations: Number of MCTS simulations to run for counterfactual generation.
            cf_d_max: Maximum MCTS search depth for counterfactual simulations.
            tau_limits: Lower and upper bounds on the distance of tau from t_action.
            time_limits: The maximal amount of time to look back in the past and future.

        Keyword Args: See arguments of parent-class MCTSAgent.
        """

        super(XAVIAgent, self).__init__(**kwargs)

        self.__n_trajectories = cf_n_trajectories
        self.__tau_limits = np.array(tau_limits)
        self.__time_limits = np.array(time_limits)
        self.__scenario_map = kwargs["scenario_map"]

        self.__cf_n_simulations = kwargs.get("cf_n_simulations", cf_n_simulations)
        self.__cf_max_depth = kwargs.get("cf_max_depth", cf_max_depth)
        self.__cf_goal_probabilities_dict = {"tau": None, "t_action": None}
        self.__cf_observations_dict = {"tau": None, "t_action": None}
        self.__cf_dataset_dict = {"tau": None, "t_action": None}
        mcts_params = {"scenario_map": self.__scenario_map,
                       "n_simulations": self.__cf_n_simulations,
                       "max_depth": self.__cf_max_depth,
                       "reward": self.mcts.reward,
                       "store_results": "all",
                       "tree_type": XAVITree,
                       "action_type": XAVIAction,
                       "trajectory_agents": False}
        self.__cf_mcts_dict = {
            "tau": ip.MCTS(**mcts_params),
            "t_action": ip.MCTS(**mcts_params),
        }

        self.__features = Features()
        self.__matching = ActionMatching()
        self.__language = LanguageTemplate()

        self.__previous_queries = []
        self.__user_query = None
        self.__current_t = None
        self.__observations_segments = None
        self.__total_trajectories = None
        self.__mcts_results_buffer = []

    def update_plan(self, observation: ip.Observation):
        super(XAVIAgent, self).update_plan(observation)

        # Retrieve maneuvers and macro actions for non-ego vehicles
        for rollout in self.mcts.results:
            last_node = rollout.leaf
            for agent_id, agent in last_node.run_result.agents.items():
                if isinstance(agent, ip.TrajectoryAgent):
                    plan = self.goal_probabilities[agent_id].trajectory_to_plan(*rollout.samples[agent_id])
                    fill_missing_actions(agent.trajectory_cl, plan)
                agent.trajectory_cl.calculate_path_and_velocity()

        current_t = int(self.observations[self.agent_id][0].states[-1].time)
        self.__mcts_results_buffer.append((current_t, self.mcts.results))

    def explain_actions(self, user_query: Query) -> str:
        """ Explain the behaviour of the ego considering the last tau time-steps and the future predicted actions.

        Args:
            user_query: The parsed query of the user.

        Returns: A natural language explanation of the query.
        """
        self.__user_query = user_query
        self.__user_query.fps = self.fps
        self.__user_query.tau_limits = self.tau_limits

        self.__current_t = int(self.observations[self.agent_id][0].states[-1].time)
        if self.__observations_segments is None or user_query.t_query != self.__current_t:
            self.__observations_segments = {}
            for aid, obs in self.observations.items():
                self.__observations_segments[aid] = self.__matching.action_segmentation(obs[0])
        self.__total_trajectories = self.__get_total_trajectories()

        # Determine timing information of the query.
        try:
            self.query.get_tau(self.__current_t, self.total_observations, self.__mcts_results_buffer)
        except ValueError as ve:
            logger.exception(str(ve), exc_info=ve)
            return str(ve)

        logger.info(f"Running explanation for {self.query}.")

        if self.query.type == QueryType.WHAT:
            causes = self.__explain_what()
        elif self.query.type in [QueryType.WHY, QueryType.WHY_NOT]:
            causes = self.__explain_why()
            plot_dataframe(causes[0], causes[1])
        elif self.query.type == QueryType.WHAT_IF:
            causes = self.__explain_whatif()
            plot_dataframe(causes[1], causes[2])
            logger.info(f"I will {causes[0]}")
        else:
            raise ValueError(f"Unknown query type: {self.query.type}")

        # TODO (high): Convert to NL explanations through language templates.
        sentence = self.__language.convert_to_sentence(causes)

        self.__previous_queries.append(self.__user_query)
        logger.info(f"t_action is {self.query.t_action}, tau is {self.query.tau}")
        return sentence

    def __final_causes(self, ref_items: List[Item], alt_items: List[Item]) -> pd.DataFrame:
        """ Generate final causes for the queried action.

        Args:
            ref_items: The items which provide the baseline for comparison.
            alt_items: The alternative items whose deviation we want measure from the references.

        Returns:
            Dataframe of reward components with the absolute and relative changes for each component.
        """
        diffs = {}
        for component in self._reward.COMPONENTS:
            factor = self._reward.factors.get(component, 1.0)
            r_qp = [factor * item.reward[component] for item in ref_items
                    if item.reward[component] is not None]
            r_qp = np.sum(r_qp) / len(r_qp) if r_qp else 0.0
            r_qnp = [factor * item.reward[component] for item in alt_items
                     if item.reward[component] is not None]
            r_qnp = np.sum(r_qnp) / len(r_qnp) if r_qnp else 0.0
            diff = r_qp - r_qnp
            rel_diff = diff / np.abs(r_qnp) if r_qnp else 0.0
            diffs[component] = (r_qp, r_qnp,
                                diff if not np.isnan(diff) else 0.0,
                                rel_diff if not np.isnan(rel_diff) else 0.0)
        columns = ["reference", "alternative", "absolute", "relative"]
        df = pd.DataFrame.from_dict(diffs, orient="index", columns=columns)
        return df.sort_values(ascending=False, by="absolute", key=abs)

    def __efficient_causes(self,
                           tau_dataset: List[Item] = None,
                           t_action_dataset: List[Item] = None) \
            -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """ Generate efficient causes for the queried action.

        Args:
            tau_dataset: Counterfactual items starting from timestep tau.
            t_action_dataset: Counterfactual items starting from timestep t_action.

        Returns:
            Dataframes for past and future causes with causal effect size.
        """
        xs_past, ys_past = [], []
        xs_future, ys_future = [], []
        if tau_dataset is None and t_action_dataset is None:
            return None, None

        # Get past and future datasets by truncate trajectories to relevant length and getting features
        agent_id = self.agent_id
        if self.query.type in [QueryType.WHY, QueryType.WHY_NOT]:
            agent_id = self.query.agent_id
        if tau_dataset is not None:
            for item_past in tau_dataset:
                trajectories_past = {}
                for aid, traj in item_past.trajectories.items():
                    trajectories_past[aid] = traj.slice(self.query.tau, self.query.t_action)
                xs_past.append(self.__features.to_features(agent_id, trajectories_past))
                ys_past.append(item_past.query_present)
        for item_future in t_action_dataset:
            trajectories_future = {}
            for aid, traj in item_future.trajectories.items():
                trajectories_future[aid] = traj.slice(self.query.t_action, None)
            xs_future.append(self.__features.to_features(agent_id, trajectories_future))
            ys_future.append(item_future.query_present)

        # Run a logistic regression classifier
        coefs_past = None
        if tau_dataset is not None:
            X_past, y_past = self.__features.binarise(xs_past, ys_past)
            model_past = LogisticRegression().fit(X_past, y_past)
            coefs_past = get_coefficient_significance(X_past, y_past, model_past)

        X_future, y_future = self.__features.binarise(xs_future, ys_future)
        model_future = LogisticRegression().fit(X_future, y_future)
        coefs_future = get_coefficient_significance(X_future, y_future, model_future)

        # Get coefficients using K-fold cross validation
        return coefs_past, coefs_future

    # ---------Explanation generation functions---------------

    def __explain_what(self) -> List[ActionGroup]:
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
        segments = self.__matching.action_segmentation(trajectory)
        grouped_segments = ActionGroup.group_by_maneuver(segments)
        if self.query.t_action is None:
            return grouped_segments

        start_t = self.query.t_action
        if start_t >= len(trajectory):
            logger.warning(f"Total trajectory for Agent {self.query.agent_id} is not "
                           f"long enough for query! Falling back to final timestep.")
            start_t = len(trajectory) - 1
        return [seg for seg in grouped_segments if seg.start <= start_t <= seg.end]

    def __explain_why(self) -> (pd.DataFrame, (pd.DataFrame, pd.DataFrame)):
        """ Generate a why explanation.

        Returns: The final and past and future efficient causes for the query.
        """
        logger.info("Generating a why or why-not explanation.")
        if self.query.tau is None:
            self.__get_counterfactuals(["t_action"])
            tau = None
        else:
            self.__get_counterfactuals(["tau", "t_action"])
            tau = list(self.cf_datasets["tau"].values())

        assert self.__cf_dataset_dict["t_action"] is not None, f"Missing counterfactual dataset."
        t_action = list(self.cf_datasets["t_action"].values())

        query_present, query_not_present = split_by_query(t_action)
        final_causes = self.__final_causes(query_present, query_not_present)
        efficient_causes = self.__efficient_causes(tau, t_action)

        return final_causes, efficient_causes

    def __explain_whatif(self) -> (ActionGroup, pd.DataFrame, (pd.DataFrame, pd.DataFrame)):
        """ Generate an explanation to a whatif query.
        Labels trajectories in query and finding optimal one among them, then compares the optimal one with
        factual optimal one regarding their reward components to extract final causes.

        Returns:
            An action group of the executed action at the user given time point.
            The final causes to explain that action and the efficient causes of the explanation.
        """
        # Generate a new dataset, output the most likely action. Split dataset by the cf action.
        logger.info("Generating what-if explanation.")
        self.__get_counterfactuals(["t_action"])
        cf_items, f_items = split_by_query(list(self.cf_datasets["t_action"].values()))

        # Find the maximum q-value and the corresponding action sequence of the ego for ea
        cf_optimal_rollout = find_optimal_rollout_in_subset(cf_items, self._reward.factors)
        f_optimal_rollout = find_optimal_rollout_in_subset(f_items, self._reward.factors)

        # Retrieve ego's action plan in the counterfactual case
        cf_ego_agent = cf_optimal_rollout.leaf.run_result.agents[self.agent_id]
        cf_optimal_trajectory = cf_ego_agent.trajectory_cl
        observed_segments = self.__matching.action_segmentation(cf_optimal_trajectory)
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
        final_causes = self.__final_causes(f_optimal_items, cf_optimal_items)
        efficient_causes = self.__efficient_causes(None, cf_optimal_items + f_optimal_items)

        return cf_action_group, final_causes, efficient_causes

    # ---------Counterfactual rollout generation---------------

    def __get_counterfactuals(self, times: List[str]):
        """ Get observations from tau time steps before, and call MCTS from that joint state.

        Args:
            times: The time reference points at which timestep to run MCTS from. Either tau or t_action for now.
        """
        logger.info("Generating counterfactual rollouts.")

        for time_reference in times:
            self.__generate_counterfactuals_from_time(time_reference)

    def __generate_counterfactuals_from_time(self, time_reference: str):
        """ Generate a counterfactual dataset from the time reference point.

         Args:
             time_reference: Either tau or t_action.
         """
        t = getattr(self.query, time_reference)
        truncated_observations, previous_frame = truncate_observations(self.observations, t)
        self.__cf_observations_dict[time_reference] = truncated_observations

        logger.debug(f"Generating counterfactuals at {time_reference} ({t})")
        if previous_frame:
            mcts = self.__cf_mcts_dict[time_reference]
            goal_probabilities = self.__cf_goal_probabilities_dict[time_reference]

            previous_query = self.__previous_queries[-1] if self.__previous_queries else None
            if self.cf_datasets[time_reference] is None or not previous_query or \
                    previous_query.t_query != self.query.t_query or \
                    previous_query.type != self.query.type:
                observation = ip.Observation(previous_frame, self.__scenario_map)
                goals = self.get_goals(observation)
                goal_probabilities = {aid: ip.GoalsProbabilities(goals)
                                      for aid in previous_frame.keys() if aid != self.agent_id}
                self.__generate_rollouts(previous_frame,
                                         truncated_observations,
                                         goal_probabilities,
                                         mcts)
                self.__cf_goal_probabilities_dict[time_reference] = goal_probabilities
            ref_t = self.query.t_action if time_reference == "t_action" else self.query.tau
            self.__cf_dataset_dict[time_reference] = self.__get_dataset(
                mcts.results, goal_probabilities, truncated_observations, ref_t)

    def __generate_rollouts(self,
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
        self._goal_recognition._n_trajectories = self.__n_trajectories

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

            # Set the probabilities equal for each goal and trajectory
            #  to make sure we can sample all counterfactual scenarios
            n_reachable = sum(map(lambda x: len(x) > 0, gps.trajectories_probabilities.values()))
            for goal, traj_prob in gps.trajectories_probabilities.items():
                traj_len = len(traj_prob)
                if traj_len > 0:
                    gps.goals_probabilities[goal] = 1 / n_reachable
                    gps.trajectories_probabilities[goal] = [1 / traj_len for _ in range(traj_len)]

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

    def __get_dataset(self,
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
                    r = reward_value[-1].reward_components

            # Slice the trajectory according to the tense in case of multiply actions in query exist in a trajectory
            sliced_trajectory = self.query.slice_segment_trajectory(
                trajectory_queried_agent, self.__current_t, present_ref_t=reference_t)
            query_factual = self.query.factual if not self.query.all_factual else None
            y = self.__matching.action_matching(
                self.query.action, sliced_trajectory, query_factual)
            if self.query.negative:
                y = not y

            data_set_m = Item(trajectories, y, r, rollout)
            dataset[m] = data_set_m

        logger.debug('Dataset generation done.')
        return dataset

    def __get_total_trajectories(self) -> [Observations, List]:
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
                matching_rollout = None
                for rollout in optimal_rollouts:
                    for aid, prediction in map_predictions.items():
                        if rollout.samples[aid] != prediction: break
                    else:
                        matching_rollout = rollout
                        break
                last_node = matching_rollout.tree[matching_rollout.trace[:-1]]
                agent = last_node.run_result.agents[agent_id]
                sim_trajectory = agent.trajectory_cl
            else:
                goal, sim_trajectory = map_predictions[agent_id]
                plan = self.goal_probabilities[agent_id].trajectory_to_plan(goal, sim_trajectory)
                sim_trajectory = to_state_trajectory(sim_trajectory, plan, self.fps)

            # Truncate trajectory to time step nearest to the final observation of the agent
            join_index = find_join_index(self.__scenario_map, trajectory, sim_trajectory)
            sim_trajectory = sim_trajectory.slice(int(join_index), None)

            trajectory.extend(sim_trajectory, reload_path=True, reset_times=True)
            ret[agent_id] = (trajectory, sim_trajectory.states[0])
        return ret

    # -------------Field access properties-------------------

    @property
    def cf_datasets(self) -> Dict[str, Optional[Dict[int, Item]]]:
        """ The most recently generated set of counterfactuals rolled back to tau. """
        return self.__cf_dataset_dict

    @property
    def cf_goals_probabilities(self) -> Dict[str, Optional[Dict[int, ip.GoalsProbabilities]]]:
        """ The goal and trajectory probabilities inferred from tau time steps ago. """
        return self.__cf_goal_probabilities_dict

    @property
    def cf_n_simulations(self) -> int:
        """ The number of rollouts to perform in counterfactual MCTS. """
        return self.__cf_n_simulations

    @property
    def cf_mcts(self) -> Dict[str, ip.MCTS]:
        """ MCTS planners for each time reference point. """
        return self.__cf_mcts_dict

    @property
    def total_observations(self) -> Observations:
        """ Returns the factual observations extended with the most optimal predicted trajectory for all agents. """
        return self.__total_trajectories

    @property
    def observation_segmentations(self) -> Dict[int, List[ActionSegment]]:
        """ Segmentations of the observed trajectories for each vehicle. """
        return self.__observations_segments

    @property
    def tau_limits(self) -> np.ndarray:
        """ The lower and upper bound of the distance of tau from t_action. """
        return self.__tau_limits

    @property
    def query(self) -> Query:
        """ The most recently asked user query. """
        return self.__user_query
