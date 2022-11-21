import numpy as np
import logging
import igp2 as ip

logger = logging.getLogger(__name__)


class ActionMatching:
    """ Determines if the action asked by a user is present in a trajectory. """

    def __init__(self):
        self.__trajectory = None
        self.__action = None
        self._eps = 0.1

    def action_matching(self, action: str, ego_trajectory: ip.StateTrajectory) -> bool:
        # TODO: Rewrite functions here to return all occurrences of action and the time spans during which they happen
        """ Match user queried action with trajectories from MCTS.

        Args:
            action: the user queried action.
            ego_trajectory: the ego trajectory
            
        Returns:
            True if action was matched with trajectory
        """
        self.__trajectory = ego_trajectory
        self.__action = action
        if action == 'SlowDown':
            return self.action_slow_down()
        elif action == 'Stop':
            return self.action_stop()
        elif action == 'GiveWay':
            return self.action_give_way()
        elif action == 'GoStraight':
            return self.action_go_straight()
        elif action == 'ChangeLaneLeft':
            return self.action_lc_left()
        elif action == 'ChangeLaneRight':
            return self.action_lc_right()
        elif action == 'TurnLeft':
            return self.action_turn_left()
        elif action == 'TurnRight':
            return self.action_turn_right()
        else:
            raise Exception('User action does not exist in action library.')

    def action_slow_down(self):
        """ find action includes slow down. """
        acceleration = np.dot(self.__trajectory.timesteps, self.__trajectory.acceleration)
        if acceleration < -self._eps:
            return True
        else:
            return False

    def action_stop(self):
        """ find action includes stop. """
        stopped = self.__trajectory.velocity < self.__trajectory.VELOCITY_STOP
        return np.any(stopped)

    def action_lc_left(self):
        """ find action includes lane changing. """
        for state in self.__trajectory.states:
            if state.macro_action is None:
                continue
            if self.__action in state.macro_action:
                return True
        return False

    def action_lc_right(self):
        """ find action includes lane changing. """
        for state in self.__trajectory.states:
            if state.macro_action is None:
                continue
            if self.__action in state.macro_action:
                return True
        return False

    def action_turn_left(self):
        """ find action includes turning.
        suppose left turn angular velocity is positive,and right turn angular velocity is negative
        """
        for inx, state in enumerate(self.__trajectory.states):
            if state.maneuver is None:
                continue
            if 'TurnCL' in state.maneuver and self.__trajectory.angular_velocity[inx] > self._eps:
                return True
        return False

    def action_turn_right(self):
        """ find action includes turning.
        suppose left turn angular velocity is positive,and right turn angular velocity is negative
        """
        for inx, state in enumerate(self.__trajectory.states):
            if state.maneuver is None:
                continue
            if 'TurnCL' in state.maneuver and self.__trajectory.angular_velocity[inx] < -self._eps:
                return True
        return False

    def action_give_way(self):
        """ find action includes give way. """
        for state in self.__trajectory.states:
            if state.maneuver is None:
                continue
            if 'GiveWayCL' in state.maneuver:
                return True
        return False

    def action_go_straight(self):
        """ find action includes go straight. """
        for state in self.__trajectory.states:
            if state.maneuver is None:
                continue
            if 'FollowLaneCL' in state.maneuver:
                return True
        return False
