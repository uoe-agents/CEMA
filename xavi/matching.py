import numpy as np
import logging
import igp2 as ip

logger = logging.getLogger(__name__)


class matching:
    """ determine if the maneuver asked by a user presents in a trajectory """
    def __init__(self):
        self._tra = None
        self._maneuver = None
        self._eps = 0.1

    def maneuver_matching(self, maneuver: str, ego_tra: ip.StateTrajectory) -> bool:
        """ match use query with trajectories from MCTS. return true if matched

        Args:
            maneuver: the user queried maneuver.
            ego_tra: The ego trajectory
        """
        self._tra = ego_tra
        self._maneuver = maneuver
        if maneuver == 'SlowDown':
            return self.maneuver_slow_down()
        elif maneuver == 'Stop':
            return self.maneuver_stop()
        elif maneuver == 'GiveWay':
            return self.maneuver_give_way()
        elif maneuver == 'GoStraight':
            return self.maneuver_go_straight()
        elif maneuver == 'ChangeLaneLeft' or maneuver == 'ChangeLaneRight':
            return self.maneuver_lc()
        elif maneuver == 'TurnLeft':
            return self.maneuver_turn_left()
        elif maneuver == 'TurnRight':
            return self.maneuver_turn_right()
        else:
            raise Exception('use query not exits in library')

    def maneuver_slow_down(self):
        """ find maneuver includes slow down. """
        acceleration = np.dot(self._tra.timesteps, self._tra.acceleration)
        if acceleration < -self._eps:
            return True
        else:
            return False

    def maneuver_stop(self):
        """ find maneuver includes stop. """
        stopped = self._tra.velocity < self._tra.VELOCITY_STOP
        return np.any(stopped)

    def maneuver_lc(self):
        """ find maneuver includes lane changing. """
        for state in self._tra.states:
            if state.macro_action is None:
                continue
            if self._maneuver in state.macro_action:
                return True
        return False

    def maneuver_turn_left(self):
        """ find maneuver includes turning.
        suppose left turn angular velocity is positive,and right turn angular velocity is negative
        """
        for inx, state in enumerate(self._tra.states):
            if state.maneuver is None:
                continue
            if 'TurnCL' in state.maneuver and self._tra.angular_velocity[inx] > self._eps:
                return True
        return False

    def maneuver_turn_right(self):
        """ find maneuver includes turning.
        suppose left turn angular velocity is positive,and right turn angular velocity is negative
        """
        for inx, state in enumerate(self._tra.states):
            if state.maneuver is None:
                continue
            if 'TurnCL' in state.maneuver and self._tra.angular_velocity[inx] < -self._eps:
                return True
        return False

    def maneuver_give_way(self):
        """ find maneuver includes give way. """
        for state in self._tra.states:
            if state.maneuver is None:
                continue
            if 'GiveWayCL' in state.maneuver:
                return True
        return False

    def maneuver_go_straight(self):
        """ find maneuver includes go straight. """
        for state in self._tra.states:
            if state.maneuver is None:
                continue
            if 'FollowLaneCL' in state.maneuver:
                return True
        return False