import os.path
import pickle
import re
from typing import Dict, List, Tuple, Optional

import igp2 as ip
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold, cross_validate

from xavi.simulation import Simulation


# -----------Simulation plotting functions---------------------


def plot_simulation(simulation: Simulation, axes: plt.Axes = None, debug: bool = False) -> (plt.Figure, plt.Axes):
    """ Plot the current agents and the road layout for visualisation purposes.

    Args:
        simulation: The simulation to plot.
        axes: Axis to draw on
        debug: If True then plot diagnostic information.
    """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    else:
        fig = plt.gcf()
    fig.suptitle(f"T={simulation.t}")

    color_map_ego = plt.cm.get_cmap('Reds')
    color_map_non_ego = plt.cm.get_cmap('Blues')
    color_ego = 'r'
    color_non_ego = 'b'
    color_bar_non_ego = None

    ax = axes[0]
    ip.plot_map(simulation.scenario_map, markings=True, ax=ax)
    for agent_id, agent in simulation.agents.items():
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
            plot_predictions(agent, simulation.agents, axes[1], debug)
        elif isinstance(agent, ip.TrajectoryAgent) and color_bar_non_ego is None:
            color_bar_non_ego = plt.colorbar(agent_plot, location="left")
        plt.text(*agent.state.position, agent_id)

    if debug:
        plot_diagnostics(simulation.agents, simulation.actions)

    return fig, axes


def plot_diagnostics(agents: Dict[int, ip.Agent], actions: Dict[int, List[ip.Action]]) -> (plt.Figure, plt.Axes):
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
            plot_maneuvers(agent, ax)
        axes[i, 0].set_ylabel(f"Agent {aid}")
    fig.tight_layout()
    return fig, axes


def plot_maneuvers(agent: ip.Agent, ax: plt.Axes) -> plt.Axes:
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


def plot_predictions(ego_agent: ip.MCTSAgent,
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
                                 figsize=(n_goals * subplot_w, n_agents * subplot_w,))
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


# -----------Explanation plotting functions---------------------
def plot_dataframe(
        df: pd.DataFrame,
        coefs: Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]],
        save_path: str = None):
    # plot final cause

    # plot absolute reward difference
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    ax = axs[0]
    rewards = {"time": "Time to goal\n(s)",
               "jerk": f"Jerk\n(m/s^3)",
               "angular_velocity": "Angular velocity\n(rad/s)",
               "curvature": "Curvature\n(1/m)",
               "coll": "Collision",
               "dead": "Goal not reached"}
    binaries = df.loc[["coll", "dead"]]
    df = df.drop(["coll", "dead"])
    y_tick_labels = [rewards[indx] for indx in df.index]
    r_diffs = df.absolute
    rewards, widths = list(zip(*[(k, v) for (k, v) in r_diffs.items() if not np.isnan(v)]))
    ax.barh(rewards, widths, left=0, height=1.0, color=plt.cm.get_cmap("tab10").colors)
    c_star = max(r_diffs.index, key=lambda k: np.abs(r_diffs[k]))
    r_star = r_diffs[c_star]
    # plt.title(rf"$c^*:{c_star}$  $r^*={np.round(r_star, 3)}$")
    ax.set_title(f"Collision: {binaries.loc['coll', 'absolute'] > 0}; "
                 f"Goal not reached: {binaries.loc['dead', 'absolute'] > 0}")
    ax.set_yticklabels(y_tick_labels)

    # plot past and future efficient causes
    macro_re = re.compile(r"^(\w+)\(([^,]+)(,[^,]+)*\)$")
    for inx, coef in enumerate(coefs, 1):
        ax = axs[inx]
        if coef is None:
            continue
        coef = coef.reindex(sorted(coef.columns, key=lambda x: x[0]), axis=1)
        strip = sns.stripplot(data=coef, orient="h", palette="dark:k", alpha=0.5, ax=ax)
        violin = sns.violinplot(data=coef, orient="h", color="cyan", saturation=0.5, whis=10, ax=ax)
        ax.axvline(x=0, color=".5")
        ax.set_xlabel("Coefficient importance")
        if inx == 1:
            # ax.set_title("Coefficient importance and its variability (past causes)")
            ax.set_title("Past causes")
        else:
            ax.set_title("Present-future causes")
        y_tick_labels = []
        line_pos = []
        prev_veh = None
        for i, lbl in enumerate(coef.columns):
            lbl_split = lbl.split("_")
            if "macro" in lbl_split:
                lbl_split.remove("macro")
                match = macro_re.match(lbl_split[1])
                action = match.groups()[0]
                params = match.groups()[1:]
                if action == "Exit":
                    action += " " + params[0]
            else:
                action = ' '.join(lbl_split[1:]).capitalize()
            vehicle = lbl_split[0]
            if prev_veh is None:
                prev_veh = vehicle
            elif prev_veh != vehicle:
                line_pos.append(i)
                prev_veh = vehicle
            y_tick_labels.append(f"{action} (V{vehicle})")
        ax.set_yticklabels(y_tick_labels)
        # else:
        #     ax.set_yticklabels([])
        for pos in line_pos:
            ax.axhline(pos - 0.5)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(os.path.join(save_path, f"attributions.pdf"))
    # show the plot
    plt.show()


def plot_explanation(
        r_diffs: Dict[str, float],
        data: pd.DataFrame,
        labels: np.ndarray,
        model: LogisticRegression,
        save_path: str = None,
        future: bool = False) -> plt.Axes:
    """ Plot final and efficient explanations from the calculated information.

    Args:
        r_diffs: Reward component differences.
        data: Input data to the logistic regression model
        labels: Labels indicating the presence of a query
        model: The logistic regression model
        save_path: If not None, then save figures to this path
        future: Whether the data relates to future trajectories.

    Returns:
        plt.Axes that was plotted on.
    """
    # Plot reward differences
    rewards, widths = list(zip(*[(k, v) for (k, v) in r_diffs.items() if not np.isnan(v)]))

    # Plot reward differences
    plt.barh(rewards, widths, left=0, height=1.0, color=plt.cm.get_cmap("tab10").colors)
    c_star = max(r_diffs, key=lambda k: np.abs(r_diffs[k]))
    r_star = r_diffs[c_star]
    plt.title(rf"$c^*:{c_star}$  $r^*={np.round(r_star, 3)}$")
    plt.gcf().tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f"{'future' if future else 'past'}_final.png"))

    # Plot model coefficients
    feature_names = [col[:15] for col in data.columns]
    coefs = pd.DataFrame(
        np.squeeze(model.coef_) * data.std(axis=0),
        columns=["Coefficient importance"],
        index=feature_names,
    )
    coefs.plot(kind="barh", figsize=(9, 7), alpha=0.45)
    plt.xlabel("Coefficient values corrected by the feature's std. dev.")
    plt.title("Logistic model, small regularization")
    plt.axvline(x=0, color=".5")
    plt.subplots_adjust(left=0.3)

    cv = RepeatedKFold(n_splits=5, n_repeats=7, random_state=0)
    cv_model = cross_validate(model, data, labels,
                              cv=cv, return_estimator=True, n_jobs=2)
    coefs = pd.DataFrame(
        [
            np.squeeze(est.coef_) * data.iloc[train_idx].std(axis=0)
            for est, (train_idx, _) in zip(cv_model["estimator"], cv.split(data, labels))
        ],
        columns=feature_names,
    )
    # plt.figure(figsize=(9, 7))
    sns.stripplot(data=coefs, orient="h", palette="dark:k", alpha=0.5)
    sns.boxplot(data=coefs, orient="h", color="cyan", saturation=0.5, whis=10)
    plt.axvline(x=0, color=".5")
    plt.xlabel("Coefficient importance")
    plt.title("Coefficient importance and its variability")
    plt.suptitle("Logistic model, small regularization")
    plt.subplots_adjust(left=0.3)
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f"{'future' if future else 'past'}_efficient.png"))


if __name__ == '__main__':
    scenario = 1
    future = False
    save_path = os.path.join("output", str(scenario))
    if not os.path.exists("output"):
        os.mkdir("output")
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    plot_explanation(*pickle.load(open(f"s{scenario}_{'future' if future else 'past'}.p", "rb")),
                     save_path=save_path,
                     future=future)
    plt.show()
