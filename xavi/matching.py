import logging
import igp2 as ip
from dataclasses import dataclass
from typing import List, Dict

from xavi.util import fix_initial_state

logger = logging.getLogger(__name__)


@dataclass
class ActionSegment:
    times: List[int]
    actions: List[str]


@dataclass
class ActionGroup:
    maneuver: str
    start: int
    end: int
    segments: List[ActionSegment]

    def __repr__(self):
        return f"{self.maneuver}[{self.start}-{self.end}]({len(self.segments)} segments)"

    @staticmethod
    def group_by_maneuver(segmentation: List[ActionSegment]) -> List["ActionGroup"]:
        """ Group consecutive action segmentations by maneuvers. """
        ret, group = [], []
        prev_man = segmentation[0].actions[-1]
        for segment in segmentation:
            man = segment.actions[-1]  # TODO (low): Relies on final action type being the maneuver.
            if prev_man != man:
                ret.append(ActionGroup(prev_man, group[0].times[0], group[-1].times[-1], group))
                group = []
            group.append(segment)
            prev_man = man
        else:
            ret.append(ActionGroup(prev_man, group[0].times[0], group[-1].times[-1], group))
        return ret


class ActionMatching:
    """ Determines if the action asked by a user is present in a trajectory. """

    def __init__(self, eps: float = 0.1):
        self.__actions = ['SlowDown',
                          'Accelerate',
                          'Stop',
                          'ChangeLaneLeft',
                          'ChangeLaneRight',
                          'TurnLeft',
                          'TurnRight',
                          'GoStraightJunction',
                          'GiveWay',
                          'GoStraight']
        self.__trajectory = None
        self.__segmentation = None
        self.__eps = eps

    def action_segmentation(self, trajectory: ip.StateTrajectory) -> List[ActionSegment]:
        """ Segment the trajectory into different actions and sorted with time.

        Args:
            trajectory: Trajectory to segment.
        """
        if self.__segmentation is not None and self.__trajectory == trajectory:
            return self.__segmentation

        fix_initial_state(trajectory)
        self.__trajectory = trajectory

        action_sequences = []
        for inx in range(len(trajectory.times)):
            action = []
            state = trajectory.states[inx]
            if trajectory.acceleration[inx] < - self.__eps:
                action.append('SlowDown')
            elif trajectory.acceleration[inx] > self.__eps:
                action.append('Accelerate')

            if trajectory.velocity[inx] < trajectory.VELOCITY_STOP:
                action.append('Stop')

            if state.macro_action is not None:
                if 'ChangeLaneLeft' in state.macro_action:
                    action.append('ChangeLaneLeft')
                elif 'ChangeLaneRight' in state.macro_action:
                    action.append('ChangeLaneRight')

            if state.maneuver is not None:
                if 'Turn' in state.maneuver:
                    angular_vel = trajectory.angular_velocity[inx]
                    if angular_vel > self.__eps:
                        action.append('TurnLeft')
                    elif angular_vel < -self.__eps:
                        action.append('TurnRight')
                    else:
                        action.append('GoStraightJunction')
                elif 'GiveWay' in state.maneuver:
                    action.append('GiveWay')
                elif 'FollowLane' in state.maneuver:
                    action.append('GoStraight')

            action_sequences.append(action)

        # aggregate same actions during a period
        action_segmentations = []
        actions, times = [], []
        start_time = int(trajectory.states[0].time)
        previous_actions = action_sequences[0]
        for inx, actions in enumerate(action_sequences, start_time):
            if previous_actions != actions:
                action_segmentations.append(ActionSegment(times, previous_actions))
                times = []
                previous_actions = actions
            times.append(inx)
        else:
            action_segmentations.append(ActionSegment(times, actions))

        self.__segmentation = action_segmentations
        return action_segmentations

    def action_matching(self, action: str, trajectory: ip.StateTrajectory) -> bool:
        """ Match user queried action with trajectories from MCTS.

        Args:
            action: the user queried action.
            trajectory: the rollout trajectories

        Returns:
            True if action was matched with trajectory
        """
        if action not in self.__actions:
            raise Exception('User action does not exist in action library.')
        action_segmentations = self.action_segmentation(trajectory)
        for action_segmentation in action_segmentations:
            if action in action_segmentation.actions:
                return True
        return False

    @staticmethod
    def find_counter_actions(action: str) -> List[str]:
        """ find the counter action for whynot or whatif positive question.
        Args:
            action: the user queried action in whynot question.

        Returns:
            possible counter actions
        """
        # define actions and counter actions
        action_set = [['GoStraight', 'ChangeLaneLeft', 'ChangeLaneRight'],
                      ['SlowDown', 'Accelerate'],
                      ['GoStraight', 'Stop'],
                      ['GoStraight', 'TurnLeft', 'TurnRight']]
        counter_actions = []
        for set_ in action_set:
            if action not in set_:
                continue
            else:
                for action_ in set_:
                    if action != action_:
                        counter_actions.append(action_)
        if not counter_actions:
            raise Exception('No counter action is found!')

        return counter_actions

    @property
    def action_library(self) -> List[str]:
        """ The available actions for a query. """
        return self.__actions

    @property
    def segmentation(self) -> List[ActionSegment]:
        """ The most recently calculated action segmentation. """
        return self.__segmentation

    @property
    def trajectory(self) -> ip.StateTrajectory:
        """ The most recently passed trajectory for segmentation. """
        return self.__trajectory
