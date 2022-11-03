import numpy as np
import logging
from typing import Dict, Any, List
import igp2 as ip
from xavi.util import most_common

logger = logging.getLogger(__name__)


class Features:
    """ Convert joint trajectories into a set features. """

    def __init__(self):
        self.__features = {}
        logger.info("Features are still placeholder.")

    def convert(self,
                agent_id: int,
                trajectories: Dict[int, ip.StateTrajectory],
                eps: float = 1e-1) \
            -> Dict[str, Any]:
        """ Convert a joint set of trajectories to binary features.

        Args:
            agent_id: ID of agent for which to convert joint trajectories to features.
            trajectories: The joint trajectories of agents
            eps: Threshold to check equality to zero
        """
        features = {}
        agent_trajectory = trajectories[agent_id]

        for aid, trajectory in trajectories.items():
            if aid == agent_id:
                continue

            # Agent acceleration as the weighted sum of acceleration with temporal difference
            if len(trajectory) == 0:
                acceleration = np.nan
            else:
                acceleration = np.dot(trajectory.timesteps, trajectory.acceleration)
            features.update({f"{agent_id}_decelerate": acceleration < -eps,
                                    f"{agent_id}_maintain": -eps <= acceleration <= eps,
                                    f"{agent_id}_accelerate": eps < acceleration})

            # Get average relative velocity as compared to each agent
            if len(trajectory) == 0:
                rel_velocity = np.nan
            else:
                min_len = min(len(trajectory.velocity), len(agent_trajectory.velocity))
                d_velocity = trajectory.velocity[:min_len] - agent_trajectory.velocity[:min_len]
                rel_velocity = d_velocity.mean()
            features.update({f"{aid}_slower": rel_velocity < -eps,
                                    f"{aid}_same_velocity": -eps <= rel_velocity <= eps,
                                    f"{aid}_faster": eps < rel_velocity})

            # Get stopping feature for each vehicle.
            if len(trajectory) == 0:
                stopped = False
            else:
                stopped = trajectory.velocity < trajectory.VELOCITY_STOP
            features[f"{aid}_stops"] = np.any(stopped)

            # Select the most frequent macro action
            mas, mans = [], []
            if len(trajectory) != 0:
                for state in trajectory:
                    mas.append(state.macro_action)
                    mans.append(state.maneuver)
            features[f"{aid}_macro"] = most_common(mas) or None
            # cat_features[f"{aid}_maneuver"] = most_common(mans)

        return features

    def binarise(self, data: List[List[Any]]) -> List[List[int]]:
        """ Binarise a data set of features. """
        pass