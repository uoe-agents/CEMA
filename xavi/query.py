from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Dict

import logging
import igp2 as ip

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
    __tau: int = None
    __t_action: int = None

    def __post_init__(self):
        self.type = QueryType(self.type)

    def get_tau(self, observations: Dict[int, Tuple[ip.StateTrajectory, ip.AgentState]]):
        """ Calculate tau and the start time step of the queried action.
        Stores results in fields tau, and t_action.

        Args:
            Observations of the environment up to current time step.
        """
        tau = -1
        agent_id = self.agent_id
        if self.type == QueryType.WHAT_IF:
            agent_id = self.agent_id

        trajectory = observations[agent_id][0]
        len_states = len(trajectory.states)
        if self.type in [QueryType.WHY, QueryType.WHAT_IF]:
            # TODO: This code only works for macro actions, but not for behaviour like slow-down.
            action_matched = False
            for i, state in enumerate(trajectory.states[::-1]):
                if state.macro_action is not None:
                    if not action_matched and self.action in state.macro_action:
                        action_matched = True
                    if action_matched and self.action not in state.macro_action:
                        tau = i
                        break
                # macro action is none at the first timestamp
                elif i == len_states - 1:
                    tau = i
            if tau < 0:
                logger.warning(f"Couldn't find tau for man")
        elif self.type == QueryType.WHY_NOT:
            # Iterate across consecutive pairs in reverse order and count number of time steps to go back.
            for i, (s_curr, s_prev) in enumerate(zip(trajectory.states[:0:-1], trajectory.states[-2::-1]), 2):
                if s_curr.macro_action != s_prev.macro_action:
                    tau = i
                    break
            else:
                logger.warning(f"Tau is equal to the length of the entire trajectory.")
                tau = len_states
        elif self.type == QueryType.WHAT:
            tau = 0
        else:
            raise ValueError(f"Unknown query type {self.type}.")

        assert self.__tau >= 0, f"Tau cannot be negative."
        t_action = tau  # TODO: Placeholder, should be calculated correctly.
        self.__tau = tau
        self.__t_action = t_action

    @property
    def tau(self) -> Optional[int]:
        """ Rollback time span for counterfactual generation. """
        return self.__tau

    @property
    def t_action(self) -> int:
        """ The start timestep of the queried action. """
        return self.__t_action
