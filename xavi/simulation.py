from collections import defaultdict

import igp2 as ip
import logging
import matplotlib.pyplot as plt

from typing import Dict, Union, List

import numpy as np

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
        self.__actions = defaultdict(list)

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
        logger.debug(f"Added Agent {new_agent.agent_id}")

    def remove_agent(self, agent_id: int):
        """ Remove an agent from the simulation.

        Args:
            agent_id: Agent ID to remove.
        """
        self.__agents[agent_id].alive = False
        self.__agents[agent_id] = None
        logger.debug(f"Removed Agent {agent_id}")

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

    def __take_actions(self):
        new_frame = {}
        observation = ip.Observation(self.__state, self.__scenario_map)

        for agent_id, agent in self.__agents.items():
            if agent is None or not agent.alive:
                continue

            new_state, action = agent.next_state(observation, return_action=True)

            agent.trajectory_cl.add_state(new_state, reload_path=False)
            self.__actions[agent_id].append(action)
            new_frame[agent_id] = new_state

            agent.alive = len(self.__scenario_map.roads_at(new_state.position)) > 0
            if not agent.alive or agent.done(observation):
                self.remove_agent(agent_id)

        self.__state = new_frame

    def plot(self, axes: plt.Axes = None, debug: bool = False) -> (plt.Figure, plt.Axes):
        """ Plot the current agents and the road layout for visualisation purposes.

        Args:
            axes: Axis to draw on
            debug: If True then plot diagnostic information.
        """
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        else:
            fig = plt.gcf()

        color_map_ego = plt.cm.get_cmap('Reds')
        color_map_non_ego = plt.cm.get_cmap('Blues')
        color_ego = 'r'
        color_non_ego = 'b'
        color_bar_non_ego = None

        ax = axes[0]
        ip.plot_map(self.__scenario_map, markings=True, ax=ax)
        for agent_id, agent in self.__agents.items():
            if agent is None or not agent.alive:
                continue

            if isinstance(agent, ip.MCTSAgent):
                color = color_ego
                color_map = color_map_ego
            else:
                color = color_non_ego
                color_map = color_map_non_ego

            path = None
            velocity = None
            if isinstance(agent, ip.MacroAgent) and agent.current_macro is not None:
                path = agent.current_macro.current_maneuver.trajectory.path
                velocity = agent.current_macro.current_maneuver.trajectory.velocity
            elif isinstance(agent, ip.TrajectoryAgent) and agent.trajectory is not None:
                path = agent.trajectory.path
                velocity = agent.trajectory.velocity

            agent_plot = None
            if path is not None and velocity is not None:
                agent_plot = ax.scatter(path[:, 0], path[:, 1], c=velocity, cmap=color_map, vmin=-4, vmax=20, s=8)

            vehicle = agent.vehicle
            pol = plt.Polygon(vehicle.boundary, color=color)
            ax.add_patch(pol)
            ax.text(*agent.state.position, agent_id)
            if isinstance(agent, ip.MCTSAgent) and agent_plot is not None:
                plt.colorbar(agent_plot, ax=ax)
                plt.text(0, 0.1, 'Current Velocity: ' + str(agent.state.speed), horizontalalignment='left',
                         verticalalignment='bottom', transform=ax.transAxes)
                plt.text(0, 0.05, 'Current Macro Action: ' + agent.current_macro.__repr__(), horizontalalignment='left',
                         verticalalignment='bottom', transform=ax.transAxes)
                plt.text(0, 0, 'Current Maneuver: ' + agent.current_macro.current_maneuver.__repr__(),
                         horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

                # Plot goals
                for gid, goal in enumerate(agent.possible_goals):
                    loc = (goal.center.x, goal.center.y)
                    ax.plot(*loc, "ro")
                    ax.plot(*loc, "kx")
                    ax.text(*loc, gid)

                # Plot goal probabilities
                self.__plot_predictions(agent, self.__agents, axes[1], debug)
            elif isinstance(agent, ip.TrajectoryAgent) and color_bar_non_ego is None:
                color_bar_non_ego = plt.colorbar(agent_plot, location="left")
            plt.text(*agent.state.position, agent_id)

        if debug:
            self.__plot_diagnostics(self.__agents, self.__actions)

        return fig, axes

    @staticmethod
    def __plot_diagnostics(agents: Dict[int, ip.Agent], actions: Dict[int, List[ip.Action]]) -> (plt.Figure, plt.Axes):
        # attributes = ["velocity", "heading", "angular_velocity"]
        attributes = ["velocity", "acceleration", "jerk"]
        n_agents = len(agents)
        n_attributes = len(attributes)
        subplot_w = 5

        # Plot observations
        fig, axes = plt.subplots(n_agents, n_attributes,
                                 figsize=(n_attributes * subplot_w, n_agents * subplot_w))
        for i, (aid, agent) in enumerate(agents.items()):
            if agent is None:
                continue
            agent.trajectory_cl.calculate_path_and_velocity()
            ts = agent.trajectory_cl.times
            for j, attribute in enumerate(attributes):
                ax = axes[i, j]
                ys = getattr(agent.trajectory_cl, attribute)
                ys = np.round(ys, 4)
                ax.plot(ts, ys)
                ax.scatter(ts, ys, s=5)

                # Plot target velocities
                if attribute == "velocity":
                    ys = [action.target_speed for action in actions[aid]]
                    ys = [ys[0]] + ys
                    ax.plot(ts, ys, c="red")
                    ax.scatter(ts, ys, s=5, c="red")
                axes[0, j].set_title(attribute)
                Simulation.__plot_maneuvers(agent, ax)
            axes[i, 0].set_ylabel(f"Agent {aid}")
        fig.tight_layout()
        return fig, axes

    @staticmethod
    def __plot_maneuvers(agent: ip.Agent, ax: plt.Axes) -> plt.Axes:
        man_list = np.array([state.maneuver for state in agent.trajectory_cl.states])
        man_list[0] = man_list[1]
        ts = agent.trajectory_cl.times
        colors = ["red", "blue", "green"]
        t_start = 0
        i = 0
        t_max = len(man_list)
        for t_end, (a, b) in enumerate(zip(man_list[:-1], man_list[1:]), 1):
            if a != b:
                ax.axvspan(ts[t_start], ts[t_end], facecolor=colors[i % len(colors)], alpha=0.2)
                ax.annotate(a, xy=((t_start + 0.5 * (t_end - t_start)) / t_max, 0.0), rotation=-45,
                            xycoords='axes fraction', fontsize=10, xytext=(-20, 5), textcoords='offset points')
                t_start = t_end
                i += 1
        if ts[t_start] != ts[-1]:
            ax.axvspan(ts[t_start], ts[-1], facecolor=colors[i % len(colors)], alpha=0.2)
            ax.annotate(a, xy=((t_start + 0.5 * (t_max - t_start)) / t_max, 0.0), rotation=-45,
                        xycoords='axes fraction', fontsize=10, xytext=(-30, 5), textcoords='offset points')
        return ax

    @staticmethod
    def __plot_predictions(ego_agent: ip.MCTSAgent,
                           agents: Dict[int, ip.Agent],
                           ax: plt.Axes,
                           debug: bool = False) -> plt.Axes:
        x, y = 0., 1.
        dx, dy = 0.5, 0.05
        ax.text(x, y, "Goal Prediction Probabilities", fontsize="large")
        y -= 2 * dy
        for i, (aid, goals_probs) in enumerate(ego_agent.goal_probabilities.items()):
            ax.text(x, y, f"Agent {aid}:", fontsize="medium")
            y -= dy
            for gid, (goal, gp) in enumerate(goals_probs.goals_probabilities.items()):
                ax.text(x, y,
                        rf"   $P(g^{aid}_{gid}|s^{aid}_{{1:{ego_agent.trajectory_cl.states[-1].time}}})={gp:.3f}$:")
                y -= dy
                for tid, tp in enumerate(goals_probs.trajectories_probabilities[goal]):
                    ax.text(x, y, rf"       $P(\hat{{s}}^{{{aid}, {tid}}}_{{1:n}}|g^{aid}_{gid})={tp:.3f}$")
                    y -= dy
            y -= dy
            if i > 0 and i % 3 == 0:
                x += dx
        ax.axis("off")

        # Plot prediction trajectories
        if debug:
            attribute = "velocity"
            n_agents = max(2, len(agents) - 1)  # To make sure indexing works later on, at least 2 agents
            n_goals = len(ego_agent.possible_goals)
            subplot_w = 5

            fig, axes = plt.subplots(n_agents, n_goals,
                                     figsize=(n_goals * subplot_w, n_agents * subplot_w, ))
            i = 0
            for aid, agent in agents.items():
                if agent.agent_id == ego_agent.agent_id:
                    continue
                axes[i, 0].set_ylabel(f"Agent {aid}")
                probs = ego_agent.goal_probabilities[aid]
                for gid, goal in enumerate(probs.goals_probabilities):
                    axes[0, gid].set_title(f"{goal[0]}", fontsize=10)
                    ax = axes[i, gid]
                    opt_trajectory = probs.optimum_trajectory[goal]
                    if probs.all_trajectories[goal]:
                        trajectory = probs.all_trajectories[goal][0]
                        ax.plot(opt_trajectory.times, getattr(opt_trajectory, attribute), "r", label="Optimal")
                        ax.plot(trajectory.times, getattr(trajectory, attribute), "b", label="Observed")
                axes[i, 0].legend()
                i += 1
            fig.suptitle(attribute)
            fig.tight_layout()
        return ax

    @property
    def state(self) -> Dict[int, ip.AgentState]:
        """ A dictionary to output the states of agents. """
        return self.__state
