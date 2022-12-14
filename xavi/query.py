from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Dict, List, Optional

import logging
import igp2 as ip
import numpy as np

from xavi.matching import ActionMatching, ActionSegment

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """ Supported query types. """
    WHY = "why"
    WHY_NOT = "whynot"
    WHAT_IF = "whatif"
    WHAT = "what"


@dataclass
class Query:
    """ Dataclass to store parsed query information.

    Args:
        type: The type of the query. Either why, whynot, whatif, or what.
        t_query: The time the user asked the query.
        action: The action the user is interested in.
        agent_id: The specific ID of the agent, the user queried.
        negative: If true then the query will be matched against all trajectories that are NOT equal to action.
        t_action: The start timestep of the action in question.
        tau: The number of timesteps to rollback from the present for counterfactual generation.
        tense: Past, future, or present. Indicates the time of the query.
    """

    type: QueryType
    t_query: int = None
    action: str = None
    agent_id: int = None
    negative: bool = None
    t_action: int = None
    tau: int = None
    tense: str = None

    __longest_action: Tuple[str, ...] = None

    fps: int = 20
    tau_limits: np.ndarray = np.array([1, 5])

    def __post_init__(self):
        self.__matching = ActionMatching()

        # Perform value checks
        self.type = QueryType(self.type)
        if self.action is not None:
            assert self.action in self.__matching.action_library, f"Unknown action {self.action}."
        assert self.tense in ["past", "present", "future", None], f"Unknown tense {self.tense}."

    def get_tau(self,
                current_t: int,
                observations: Dict[int, Tuple[ip.StateTrajectory, ip.AgentState]],
                rollouts_buffer: List[ip.AllMCTSResult]):
        """ Calculate tau and the start time step of the queried action.
        Storing results in fields tau, and t_action.

        Args:
            current_t: The current timestep of the simulation
            observations: Trajectories observed (and possibly extended with future predictions) of the environment.
            rollouts_buffer: The actual MCTS rollouts of the agent.
        """
        agent_id = self.agent_id
        if self.type == QueryType.WHAT_IF:
            agent_id = self.agent_id

        trajectory = observations[agent_id][0]
        action_segmentations = self.slice_segment_trajectory(trajectory, current_t)

        if self.type == QueryType.WHAT_IF and not self.negative or \
                self.type == QueryType.WHY_NOT:
            action_segmentations = self.__determine_matched_rollout(
                rollouts_buffer, action_segmentations, agent_id, current_t)

        tau = len(trajectory) - 1
        if self.type in [QueryType.WHAT_IF, QueryType.WHY_NOT, QueryType.WHY]:
            t_action, tau = self.__get_t_tau(action_segmentations, self.type != QueryType.WHAT_IF)
        elif self.type == QueryType.WHAT:
            t_action = self.t_action
        else:
            raise ValueError(f"Unknown query type {self.type}.")

        if tau is not None:
            assert tau > 0, f"Tau has to be positive."

        if self.tau is None:
            self.tau = tau  # If user gave fixed tau then we shouldn't override that.
        self.t_action = t_action

    def __get_t_tau(self,
                    action_segmentations: List[ActionSegment],
                    rollback: bool) -> (int, int):
        """ determine t_action for final causes, tau for efficient cause.
        Args:
            action_segmentations: the segmented action of the observed trajectory.
            rollback: if rollback is needed

        Returns:
            t_action: the timestep when the factual action starts
            tau: the timesteps to rollback
        """
        tau = None
        action_matched = False
        n_segments = len(action_segmentations)
        for i, action in enumerate(action_segmentations[::-1]):
            if self.action in action.actions:
                action_matched = True
            elif action_matched:
                t_action = action.times[-1] + 1
                segment_inx = i
                break
        else:
            if action_matched:
                t_action = action_segmentations[0].times[0]  # t at the beginning
                segment_inx = n_segments - 1
            else:
                raise ValueError(f"Could not match action {self.action} to trajectory.")

        if rollback and segment_inx >= 0:
            # In case one extra segment is too short, then lower limit is used.
            lower_limit, upper_limit = self.tau_limits * self.fps
            previous_inx = max(0, n_segments - segment_inx - 1)
            previous_segment = action_segmentations[previous_inx]
            tau = previous_segment.times[0]

            if t_action - tau < lower_limit:
                tau = int(t_action - lower_limit)
            elif t_action - tau > upper_limit:
                tau = int(t_action - upper_limit)

            tau = max(1, tau)

        t_action = max(1, t_action)
        return t_action, tau

    def __determine_matched_rollout(self,
                                    rollouts_buffer: List[ip.AllMCTSResult],
                                    observation_segmentations: List[ActionSegment],
                                    agent_id: int,
                                    current_t: int) -> List[ActionSegment]:
        """ determine the action segmentation of the rollout that matches the query for whynot and what-if questions.
        Args:
            rollouts_buffer: all previous rollouts from MCTS.
            observation_segmentations: the action segmentation using observation (factual actions)
            agent_id: the agent id in query
            current_t: current time, unit:s

        Returns:
            action_segmentations: the action segmentation of a rollout matched with the query
        """
        for rollouts in rollouts_buffer[::-1]:
            action_statistic = {}
            segmentations = []
            for rollout in rollouts.mcts_results:
                # get the start and end time of a rollout
                trajectory = rollout.leaf.run_result.agents[agent_id].trajectory_cl
                action_segmentations = self.slice_segment_trajectory(trajectory, current_t)
                start_t = action_segmentations[0].times[0]
                end_t = action_segmentations[-1].times[-1]
                for seg in observation_segmentations:
                    for time in seg.times:
                        if not (start_t <= time <= end_t):
                            continue
                        key = tuple(seg.actions)  # to handle act
                        if key not in action_statistic:
                            action_statistic[key] = 1
                        else:
                            action_statistic[key] += 1
                segmentations.append(action_segmentations)
            self.__longest_action = max(action_statistic, key=action_statistic.get)

            for segmentation in segmentations:
                # skip the rollout that includes the longest action
                if ActionMatching.action_exists(segmentation, self.__longest_action):
                    continue
                if ActionMatching.action_exists(segmentation, self.action):
                    return segmentation
        return []

    def slice_segment_trajectory(self,
                                 trajectory: ip.StateTrajectory,
                                 current_t: int) -> List[ActionSegment]:
        # Obtain relevant trajectory slice and segment it
        """ Obtain relevant trajectory slice and segment it.
        Args:
            trajectory: the agent trajectory.
            current_t: the current time

        Returns:
            action_segmentations: the segmented actions
        """
        current_inx = int(current_t - trajectory[0].time + 1)
        if self.tense in ["past", "present"]:
            trajectory = trajectory.slice(0, current_inx)
        elif self.tense == "future":
            trajectory = trajectory.slice(current_inx, None)
        elif self.tense is None:
            logger.warning(f"Query time was not given. Falling back to observed trajectory.")
            trajectory = trajectory.slice(0, current_inx)

        action_segmentations = self.__matching.action_segmentation(trajectory)
        return action_segmentations

    @property
    def longest_action(self) -> Optional[Tuple[str, ...]]:
        """ Used for whynot and positive whatif queries.
        Contains the longest factual action(s) of the specified agent. """
        return self.__longest_action
