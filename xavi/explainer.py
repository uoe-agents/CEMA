import igp2 as ip
from typing import Dict


class rollout_generation:
    """ generate new rollouts in the observation tau time before """

    def __init__(self,
                 layout: ip.Map,
                 tau: float = 0.05):
        """
        Args:
            tau: the time parameter to shift the environment back, defined as 1/fps.
            layout: scenario map.
        """
        self._layout = layout
        self._tau = tau

    def past_trajectories(self, frame: Dict[int, ip.AgentState]) -> Dict[int, ip.GoalsProbabilities]:
        """ Return possible trajectories of non-ego vehicles tau time before """

        """
        Args:
            frame: states of agents at current time step.
        """
        goal_probabilities = {}
        for aid, agent_state in frame.items():
            # 0 should be changed to the id of ego
            if aid == 0 or self._tau > agent_state.time:
                continue


        return goal_probabilities

