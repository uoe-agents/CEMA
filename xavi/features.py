import numpy as np
import logging
from typing import Dict
import igp2 as ip

logger = logging.getLogger(__name__)


class Features:
    """ Convert joint trajectories into a set features. """
    def __init__(self):
        logger.info("Features are still placeholder.")

    def convert(self, trajectories: Dict[int, ip.Trajectory]) -> Dict[str, bool]:
        """ Convert a joint set of trajectories to binary features. """
        return {a: 0 for a in "abcdefg"}
