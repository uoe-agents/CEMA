import igp2 as ip
import logging
import matplotlib.pyplot as plt

from typing import Dict, Union

logger = logging.getLogger(__name__)


class Simulation:
    """ A wrapper around the lightweight simulator of IGP2 to perform rapid testing. """

    def __init__(self,
                 scenario_map: ip.Map,
                 fps: int = 20,
                 open_loop: bool = False,
                 **kwargs):
        """ Initialise new simulation.

        Args:
            scenario_map: The current road layout.
            fps: Execution frame-rate.
            open_loop: If true then no physical controller will be applied.
        """
        self.__scenario_map = scenario_map
        self.__fps = fps
        self.__open_loop = open_loop

        self.__t = 0
        self.__state = {}
        self.__agents = {}

    def add_agent(self, new_agent: ip.Agent):
        """ Add a new agent to the simulation.

        Args:
            new_agent: Agent to add.
        """
        if new_agent.agent_id in self.__agents \
                and self.__agents[new_agent.agent_id] is not None:
            raise ValueError(f"Agent with ID {new_agent.agent_id} already exists.")

        self.__agents[new_agent.agent_id] = new_agent
        self.__state[new_agent.agent_id] = new_agent.vehicle.get_state(0)

    def remove_agent(self, agent_id: int):
        """ Remove an agent from the simulation.

        Args:
            agent_id: Agent ID to remove.
        """
        self.__agents[agent_id].alive = False
        self.__agents[agent_id] = None

    def reset(self):
        """ Remove all agents and reset internal state of simulation. """
        self.__t = 0
        self.__agents = {}
        self.__state = {}

    def step(self):
        """ Advance simulation by one time step. """
        logger.debug(f"Simulation step {self.__t}")
        self.__t += 1
        self.__take_actions()

    def plot(self, axes: plt.Axes = None, debug: bool = False):
        """ Plot the current agents and the road layout for visualisation purposes.

        Args:
            axes: Axis to draw on
            debug: If True then plot diagnostic information.
        """
        if axes is None:
            fig, axes = plt.subplots()

        color_map_ego = plt.cm.get_cmap('Reds')
        color_map_non_ego = plt.cm.get_cmap('Blues')
        color_ego = 'r'
        color_non_ego = 'b'
        color_bar_non_ego = None

        ip.plot_map(self.__scenario_map, markings=True, ax=axes)
        for agent_id, agent in self.__agents.items():
            if not agent.alive:
                continue

            if isinstance(agent, ip.MCTSAgent):
                color = color_ego
                color_map = color_map_ego
            else:
                color = color_non_ego
                color_map = color_map_non_ego

            if isinstance(agent, ip.MacroAgent):
                path = agent.current_macro.current_maneuver.trajectory.path
                velocity = agent.current_macro.current_maneuver.trajectory.velocity
            elif isinstance(agent, ip.TrajectoryAgent):
                path = agent.trajectory.path
                velocity = agent.trajectory.velocity

            vehicle = agent.vehicle
            pol = plt.Polygon(vehicle.boundary, color=color)
            axes.add_patch(pol)
            agent_plot = axes.scatter(path[:, 0], path[:, 1], c=velocity, cmap=color_map, vmin=-4, vmax=20, s=8)
            if isinstance(agent, ip.MCTSAgent):
                plt.colorbar(agent_plot)
                plt.text(0, 0.1, 'Current Velocity: ' + str(agent.state.speed), horizontalalignment='left',
                         verticalalignment='bottom', transform=axes.transAxes)
                plt.text(0, 0.05, 'Current Macro Action: ' + agent.current_macro.__repr__(), horizontalalignment='left',
                         verticalalignment='bottom', transform=axes.transAxes)
                plt.text(0, 0, 'Current Maneuver: ' + agent.current_macro.current_maneuver.__repr__(),
                         horizontalalignment='left', verticalalignment='bottom', transform=axes.transAxes)
            elif isinstance(agent, ip.TrajectoryAgent) and color_bar_non_ego is None:
                color_bar_non_ego = plt.colorbar(agent_plot)
            plt.text(*agent.state.position, agent_id)
        return axes

    def __take_actions(self):
        new_frame = {}
        observation = ip.Observation(self.__state, self.__scenario_map)

        for agent_id, agent in self.__agents.items():
            new_state = agent.next_state(observation)

            agent.trajectory_cl.add_state(new_state, reload_path=False)
            new_frame[agent_id] = new_state

            agent.alive = len(self.__scenario_map.roads_at(new_state.position)) > 0
            if not agent.alive or agent.done(observation):
                self.remove_agent(agent_id)

        self.__state = new_frame
