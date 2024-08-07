from typing import Dict, Any

import igp2 as ip

from xavi.features import Features


class OFeatures(Features):
    def to_features(self,
                    agent_id: int,
                    trajectories: Dict[int, ip.StateTrajectory],
                    eps: float = 1e-1) \
            -> Dict[str, Any]:
        features = super().to_features(agent_id, trajectories, eps)
