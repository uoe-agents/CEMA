from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Dict

import logging
import igp2 as ip
from xavi.matching import ActionMatching

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """ Supported query types. """
    WHY = "why"
    WHY_NOT = "whynot"
    WHAT_IF = "whatif"
    WHAT = "what"


@dataclass
class Query:
    """ Dataclass to store parsed query information. """
    time: int
    type: QueryType
    action: str = None
    agent_id: int = None
    negative: bool = None
    __tau: list = None
    __t_action: int = None

    def __post_init__(self):
        self.type = QueryType(self.type)
        self.__matching = ActionMatching()

    def get_tau(self, observations: Dict[int, Tuple[ip.StateTrajectory, ip.AgentState]]):
        """ Calculate tau and the start time step of the queried action.
        Storing results in fields tau, and t_action.

        Args:
            Observations of the environment up to current time step.
        """
        # saving tau for final cause and efficient cause
        tau = [-1, -1]
        agent_id = self.agent_id
        if self.type == QueryType.WHAT_IF:
            agent_id = self.agent_id

        trajectory = observations[agent_id][0]
        len_states = len(trajectory.states)
        action_segmentations = self.__matching.action_segmentation(trajectory)
        if self.type == QueryType.WHY:
            tau[0], seg_inx = self.determine_tau_factual(action_segmentations, len_states)
            # tau for efficient cause, rollback starts from tau for final causes
            previous_segment = action_segmentations[len(action_segmentations) - seg_inx - 1]
            tau[1] = len_states - previous_segment.times[0]

        elif self.type == QueryType.WHY_NOT:
            tau[0], seg_inx = self.determine_tau_counterfactual(action_segmentations, len_states)
            previous_segment = action_segmentations[len(action_segmentations) - seg_inx - 1]
            tau[1] = len_states - previous_segment.times[0]

        elif self.type == QueryType.WHAT_IF:
            if self.negative:
                tau[0], seg_inx = self.determine_tau_factual(action_segmentations, len_states)
            else:
                tau[0], seg_inx = self.determine_tau_counterfactual(action_segmentations, len_states)

        elif self.type == QueryType.WHAT:
            tau[0] = 0
        else:
            raise ValueError(f"Unknown query type {self.type}.")

        if tau[0] or tau[1] < 0:
            logger.warning(f"Couldn't find tau for man")
        elif tau[1] == len_states:
            logger.warning(f"rollback to the start of an entire observation, "
                           f"cannot generate past cases for efficient explanations")

        t_action = len_states - tau[0]
        self.__tau = tau
        self.__t_action = t_action

    def determine_tau_factual(self, action_segmentations, len_states) -> [float, int]:
        """ determine tau for final cause for why and whatif negative question. factual action is mentioned"""
        tau_final = -1
        seg_inx = -1
        action_matched = False
        for i, action in enumerate(action_segmentations[::-1]):
            if self.action in action.actions:
                action_matched = True
            if action_matched and self.action not in action.actions:
                tau_final = len_states - action.times[-1] - 1
                seg_inx = i
                break
        return tau_final, seg_inx

    def determine_tau_counterfactual(self, action_segmentations, len_states) -> [float, int]:
        """ determine tau for final cause for whynot and whatif positive question. counterfactual action is mentioned"""
        counter_actions = self.__matching.find_counter_actions(self.action)
        matched_action = None
        action_matched = False
        tau_final = -1
        seg_inx = -1
        for i, action in enumerate(action_segmentations[::-1]):
            if not matched_action:
                for action_ in action.actions:
                    if action_ not in counter_actions:
                        continue
                    action_matched = True
                    matched_action = action_
                    break
            if action_matched and matched_action not in action.actions:
                tau_final = len_states - action.times[-1] - 1
                seg_inx = i
                break
        logger.info(f'factual action found is {matched_action}')
        return tau_final, seg_inx

    @property
    def tau(self) -> Optional[list]:
        """ Rollback time span for counterfactual generation. """
        return self.__tau

    @property
    def t_action(self) -> int:
        """ The start timestep of the queried action. """
        return self.__t_action
