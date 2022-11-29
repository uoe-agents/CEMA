from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Dict

import logging
import igp2 as ip
from xavi.matching import ActionMatching, ActionData

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
            Observations of the environment up to current time step.
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
            tau = 0
            if self.negative:
                t_action, _ = self.determine_tau_factual(action_segmentations, len_states, False)
            else:
                t_action, _ = self.determine_tau_counterfactual(action_segmentations, len_states, False)

        elif self.type == QueryType.WHAT:
            tau = 0
            t_action = self.t_action

        else:
            raise ValueError(f"Unknown query type {self.type}.")

        assert tau >= 0, f"Tau cannot be negative."
        if tau == len_states:
            logger.warning(f"Rollback to the start of an entire observation, "
                           f"cannot generate past causes for efficient explanations.")

        self.tau = tau
        self.t_action = t_action

    def determine_tau_factual(self, action_segmentations: [ActionData], len_states: int, rollback: bool) -> [int, int]:
        """ determine t_action for final causes, tau for efficient cause for why and whatif negative question.
        Args:
            action_segmentations: the segmented action of the observed trajectory.
            len_states: the length of observation
            rollback: if rollback is needed

        Returns:
            t_action: the timestep when the factual action starts
            tau: the timesteps to rollback
        """
        seg_inx = -1
        t_action = -1
        tau = -1
        action_matched = False
        for i, action in enumerate(action_segmentations[::-1]):
            if self.action in action.actions:
                action_matched = True
            if action_matched and self.action not in action.actions:
                t_action = action.times[-1] + 1
                seg_inx = i
                break

        if rollback and seg_inx >= 0:
            previous_segment = action_segmentations[len(action_segmentations) - seg_inx - 1]
            tau = len_states - previous_segment.times[0]

        return t_action, tau

    def determine_tau_counterfactual(self, action_segmentations: [ActionData], len_states: int, rollback: bool) -> [int, int]:
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

