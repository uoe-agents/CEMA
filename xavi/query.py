from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Dict, List

import logging
import igp2 as ip
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
    """

    type: QueryType
    t_query: int = None
    action: str = None
    agent_id: int = None
    negative: bool = None
    t_action: int = None
    tau: int = None

    def __post_init__(self):
        self.type = QueryType(self.type)
        self.__matching = ActionMatching()

    def get_tau(self, observations: Dict[int, Tuple[ip.StateTrajectory, ip.AgentState]]):
        """ Calculate tau and the start time step of the queried action.
        Storing results in fields tau, and t_action.

        Args:
            observations: Trajectories observed (and possibly extended with future predictions) of the environment.
        """
        agent_id = self.agent_id
        if self.type == QueryType.WHAT_IF:
            agent_id = self.agent_id

        trajectory = observations[agent_id][0]
        len_states = len(trajectory.states)
        action_segmentations = self.__matching.action_segmentation(trajectory)
        if self.type == QueryType.WHY:
            # tau for efficient cause
            t_action, tau = self.determine_tau_factual(action_segmentations, len_states, True)

        elif self.type == QueryType.WHY_NOT:
            t_action, tau = self.determine_tau_counterfactual(action_segmentations, len_states, True)

        elif self.type == QueryType.WHAT_IF:
            tau = len_states - 1
            if self.negative:
                t_action, _ = self.determine_tau_factual(action_segmentations, len_states, False)
            else:
                t_action, _ = self.determine_tau_counterfactual(action_segmentations, len_states, False)

        elif self.type == QueryType.WHAT:
            tau = len_states - 1
            # TODO (mid): Assumes query parsing can extract reference time point.
            t_action = self.t_action
        else:
            raise ValueError(f"Unknown query type {self.type}.")

        assert tau >= 0, f"Tau cannot be negative."
        if tau == 0:
            logger.warning(f"Rollback to the start of an entire observation.")

        if self.tau is None:
            self.tau = tau  # If user gave fixed tau then we shouldn't override that.
        self.t_action = t_action

    def determine_tau_factual(self,
                              action_segmentations: List[ActionSegment],
                              len_states: int,
                              rollback: bool) -> (int, int):
        """ determine t_action for final causes, tau for efficient cause for why and whatif negative question.
        Args:
            action_segmentations: the segmented action of the observed trajectory.
            len_states: the length of observation
            rollback: if rollback is needed

        Returns:
            t_action: the timestep when the factual action starts
            tau: the timesteps to rollback
        """
        tau = -1
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
                t_action = 0
                segment_inx = n_segments - 1
            else:
                raise ValueError(f"Could not match action {self.action} to trajectory.")

        if rollback and segment_inx >= 0:
            # TODO (high): One extra segment is usually very short. Could go back 2 segments or enforce a minimum limit.
            previous_inx = max(0, n_segments - segment_inx - 1)
            previous_segment = action_segmentations[previous_inx]
            tau = max(1, previous_segment.times[0])

        return t_action, tau

    def determine_tau_counterfactual(self,
                                     action_segmentations: List[ActionSegment],
                                     len_states: int,
                                     rollback: bool) -> (int, int):
        """ determine t_action for final causes, tau for efficient cause for whynot and whatif positive question.
        Args:
            action_segmentations: the segmented action of the observed trajectory.
            len_states: the length of observation
            rollback: if rollback is needed

        Returns:
            t_action: the timestep when the factual action starts
            tau: the timesteps to rollback
        """
        counter_actions = self.__matching.find_counter_actions(self.action)
        matched_action = None
        action_matched = False
        seg_inx = -1
        t_action = -1
        tau = -1
        for i, action in enumerate(action_segmentations[::-1]):
            if not matched_action:
                for action_ in action.actions:
                    if action_ not in counter_actions:
                        continue
                    action_matched = True
                    matched_action = action_
                    break
            if action_matched and matched_action not in action.actions:
                t_action = action.times[-1] + 1
                seg_inx = i
                break
        logger.info(f'factual action found is {matched_action}')
        if rollback and seg_inx >= 0:
            previous_segment = action_segmentations[len(action_segmentations) - seg_inx - 1]
            tau = len_states - previous_segment.times[0]

        return t_action, tau

