import numpy as np
import logging
from typing import Dict
import igp2 as ip

logger = logging.getLogger(__name__)


class Features:
    """ Convert joint trajectories into a set features. """

    def __init__(self):
        self.__features = {}
        logger.info("Features are still placeholder.")

    def convert(self, agent_id: int, trajectories: Dict[int, ip.Trajectory], eps: float = 1e-1) -> Dict[str, bool]:
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
            acceleration = np.dot(trajectory.timesteps, trajectory.acceleration)
            features.update({f"{agent_id}_decelerate": acceleration < -eps,
                             f"{agent_id}_maintain": -eps <= acceleration <= eps,
                             f"{agent_id}_accelerate": eps < acceleration})

            # Get average relative velocity as compared to each agent
            min_len = min(len(trajectory.velocity), len(agent_trajectory.velocity))
            d_velocity = trajectory.velocity[:min_len] - agent_trajectory.velocity[:min_len]
            rel_velocity = d_velocity.mean()
            features.update({f"{aid}_slower": rel_velocity < -eps,
                             f"{aid}_same_velocity": -eps <= rel_velocity <= eps,
                             f"{aid}_faster": eps < rel_velocity})

            # Get stopping feature for each vehicle.
            stopped = trajectory.velocity < trajectory.VELOCITY_STOP
            features[f"{aid}_stops"] = np.any(stopped)

        return features
