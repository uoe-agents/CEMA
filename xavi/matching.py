import logging
import igp2 as ip
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class ActionData:
    times: List[int]
    actions: List[str]


class ActionMatching:
    """ Determines if the action asked by a user is present in a trajectory. """

    def __init__(self):
        self.__actions = None
        self._eps = 0.1

    def action_segmentation(self, trajectory: ip.StateTrajectory):
        # TODO: Rewrite functions here to return all occurrences of action and the time spans during which they happen
        """ segment the trajectory into different actions and sorted with time.

        Args:
            trajectory: the trajectory
        """

        action_sequences = []
        for inx in range(len(trajectory.times)):
            action = []
            if trajectory.acceleration[inx] < - self._eps:
                action.append('SlowDown')
            elif trajectory.acceleration[inx] > self._eps:
                action.append('Accelerate')

            if trajectory.velocity[inx] < trajectory.VELOCITY_STOP:
                action.append('Stop')

            if trajectory.states[inx].macro_action is not None:
                if 'ChangeLaneLeft' in trajectory.states[inx].macro_action:
                    action.append('ChangeLaneLeft')
                elif 'ChangeLaneRight' in trajectory.states[inx].macro_action:
                    action.append('ChangeLaneRight')

            if trajectory.states[inx].maneuver is not None:
                if 'TurnCL' in trajectory.states[inx].maneuver and trajectory.angular_velocity[inx] > self._eps:
                    action.append('TurnLeft')
                elif 'TurnCL' in trajectory.states[inx].maneuver and trajectory.angular_velocity[inx] < -self._eps:
                    action.append('TurnRight')
                elif 'GiveWayCL' in trajectory.states[inx].maneuver:
                    action.append('GiveWay')
                elif 'FollowLaneCL' in trajectory.states[inx].maneuver:
                    action.append('GoStraight')

            action_sequences.append(action)

        # aggregate same actions during a period
        action_segmentations = []
        times = []
        previous_actions = action_sequences[0]
        for inx, actions in enumerate(action_sequences):
            if previous_actions != actions:
                action_segmentations.append(ActionData(times.copy(), previous_actions))
                times.clear()
                previous_actions = actions
            times.append(inx)
            if inx == len(action_sequences) - 1:
                action_segmentations.append(ActionData(times.copy(), actions))

        return action_segmentations

    def action_matching(self, action: str, trajectory: ip.StateTrajectory) -> bool:
        """ Match user queried action with trajectories from MCTS.

        Args:
            action: the user queried action.
            trajectory: the rollout trajectories

        Returns:
            True if action was matched with trajectory
        """
        self.action_lib()
        if action not in self.__actions:
            raise Exception('User action does not exist in action library.')
        action_segmentations = self.action_segmentation(trajectory)
        for action_segmentation in action_segmentations:
            if action in action_segmentation.actions:
                return True
        return False

    def action_lib(self):
        """ the actions that we can answer in current framework. """
        self.__actions = ['SlowDown',
                          'Accelerate',
                          'Stop',
                          'ChangeLaneLeft',
                          'ChangeLaneRight',
                          'TurnLeft',
                          'TurnRight',
                          'GiveWay',
                          'GoStraight']

    @staticmethod
    def find_counter_actions(action: str) -> List[str]:
        """ find the counter action for whynot question.
        Args:
            action: the user queried action in whynot question.

        Returns:
            possible counter actions
        """
        # define actions and counter actions
        action_set = [['GoStraight', 'ChangeLaneLeft', 'ChangeLaneRight'],
                      ['SlowDown', 'Accelerate'],
                      ['GoStraight', 'Stop'],
                      ['GoStraight', 'TurnLeft', 'TurnRight']
                      ]
        counter_actions = []
        for set_ in action_set:
            if action not in set_:
                continue
            else:
                for action_ in set_:
                    if action != action_:
                        counter_actions.append(action_)
        if not counter_actions:
            raise Exception('No counter action is found for the whynot question')

        return counter_actions
