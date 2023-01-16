# This file generates the plots in the paper from the saved causes
# that can be generated using run.py and the final_causes.json query file.
import pickle
import json
import pandas as pd
import xavi
import re
import os
import logging
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scenarios.util import setup_xavi_logging, parse_eval_args
from sklearn.linear_model import LogisticRegression

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE + 2)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE + 2)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE + 2)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

logger = logging.getLogger(__name__)
macro_re = re.compile(r"^(\w+)\(([^,]*)(,[^,]+)*\)$")


def get_y_tick_label(lbl) -> str:
    if not isinstance(lbl, str):
        return str(lbl)
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
    if "Lane" in action:
        idx = action.index("Lane")
        action = action[:idx] + " " + action[idx + 4:].lower()
    if "velocity" in action:
        action = action.replace("velocity", "vel.")
    vehicle = lbl_split[0]
    if sid == "s1" or sid == 1:
        return f"{action}"
    else:
        return f"{action} ({vehicle})"


def plot_dataframe(
        rew_difs: Optional[pd.DataFrame],
        coefs: Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]],
        save_path: str = None):
    # plot absolute reward difference
    fig, axs = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={'width_ratios': [3, 3, 3]})
    if rew_difs is not None:
        ax = axs[0]
        rewards = {"time": "Time to goal\n(s)",
                   "jerk": f"Jerk\n($m/s^3$)",
                   "angular_velocity": "Angular velocity\n(rad/s)",
                   "curvature": "Curvature\n(1/m)",
                   "coll": "Collision",
                   "dead": "Goal not reached"}
        binaries = rew_difs.loc[["coll", "dead"]]
        rew_difs = rew_difs.drop(["coll", "dead"])
        y_tick_labels = [rewards[indx] for indx in rew_difs.index]
        r_diffs = rew_difs.absolute
        rewards, widths = list(zip(*[(k, v) for (k, v) in r_diffs.items() if not np.isnan(v)]))
        ax.barh(rewards, widths, left=0, height=1.0, color=plt.cm.get_cmap("tab10").colors)
        c_star = max(r_diffs.index, key=lambda k: np.abs(r_diffs[k]))
        r_star = r_diffs[c_star]
        # plt.title(rf"$c^*:{c_star}$  $r^*={np.round(r_star, 3)}$")
        ax.set_xlabel("(a) Cost difference")
        ax.set_title(f"Collision: {binaries.loc['coll', 'absolute'] > 0}; \n"
                     f"Goal not reached: {binaries.loc['dead', 'absolute'] > 0}")
        ax.set_yticklabels(y_tick_labels)

    # plot past and future efficient causes
    for inx, coef in enumerate(coefs, 1):
        ax = axs[inx]
        if coef is None:
            continue
        if sid == 1:  # Remove V2 for S1 plotting as it is not relevant for paper
            coef = coef.loc[:, ~coef.columns.str.startswith("2")]
        inxs = (-coef.mean(0)).argsort()
        coef = coef.iloc[:, inxs]
        inxs = np.isclose(coef.mean(0), 0)
        coef_rest = coef.loc[:, inxs].sum(1)
        coef = coef.loc[:, ~inxs]
        coef = pd.concat([coef, coef_rest], axis=1)
        # coef = coef.reindex(sorted(coef.columns, key=lambda x: x[0]), axis=1)
        sns.stripplot(data=coef, orient="h", palette="dark:k", alpha=0.5, ax=ax)
        sns.violinplot(data=coef, orient="h", color="cyan", saturation=0.5, whis=10, width=.8, scale="count", ax=ax)
        ax.axvline(x=0, color=".5")
        ax.set_xlabel(f"({'b' if inx == 1 else 'c'}) Coefficient importance")
        if inx == 1:
            # ax.set_title("Coefficient importance and its variability (past causes)")
            ax.set_title("Past causes")
        else:
            ax.set_title("Present-future causes")
        y_tick_labels = [get_y_tick_label(lbl) for lbl in coef.columns]
        y_tick_labels.append(f"Rest of {sum(inxs)}")
        ax.set_yticklabels(y_tick_labels)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(os.path.join(save_path, f"attributions.pdf"), bbox_inches='tight')
    # show the plot
    plt.show()


def eval_robustness(xp, yp, xf, yf, iters=50):
    def coef_evolution(x, y, fname):
        plt.figure(figsize=(6, 3.5))
        data = pd.concat([x, pd.DataFrame(y, columns=["y"])], axis=1)
        m = LogisticRegression().fit(x, y)
        coef = xavi.util.get_coefficient_significance(x, y, m)
        coef_ = coef.copy()
        coef["n"] = 5
        coef_["n"] = 100
        coef = pd.concat([coef, coef_], axis=0)
        cs = []
        for n in range(5, len(xp) + 5, 5):
            for i in range(iters):
                sample = data.sample(n)
                trunc_x = sample.iloc[:, :-1]
                trunc_y = sample.iloc[:, -1]
                try:
                    tm = LogisticRegression().fit(trunc_x, trunc_y)
                except:
                    continue
                c = pd.DataFrame([np.squeeze(tm.coef_) * trunc_x.std(axis=0)], columns=data.columns[:-1])
                c["n"] = int(n)
                cs.append(c)
        cs = pd.concat(cs, axis=0)
        cs = cs.loc[:, (cs != 0.).any(axis=0)]
        if len(cs.columns) > 7:
            cs = cs.loc[:, (cs.abs() > 0.28).any(axis=0)]
        if sid == "s1":  # Remove V2 for S1 plotting as it is not relevant for paper
            cs = cs.loc[:, ~cs.columns.str.startswith("2")]
        coef = coef.melt(value_vars=cs.columns, id_vars=["n"])
        cs.columns = [get_y_tick_label(c) if c != "n" else c for c in cs.columns]
        inxs = (-cs.mean(0)).argsort()
        cs = cs[cs.columns[inxs]]
        cs = cs.melt(value_vars=cs.columns, id_vars=["n"])
        cs = cs.rename({"variable": "Feature"}, axis=1)
        sns.lineplot(cs, x="n", y="value", hue="Feature", style="Feature")
        # sns.lineplot(coef, x="n", y="value", hue="variable", style="variable", legend=False, errorbar=None)
        plt.xlabel("Number of samples ($K$)")
        plt.ylabel("Feature weight")
        plt.legend(loc="upper right")
        plt.xlim([4.9, 100.1])
        plt.tight_layout()
        plt.savefig(f"output/scenario_{sid[1]}/{fname}", bbox_inches='tight')
        plt.show()
    coef_evolution(xf, yf, fname="coef_evolution_future.pdf")
    coef_evolution(xp, yp, fname="coef_evolution_past.pdf")


def load_data(scenario_id: int, query_in: Union[int, xavi.Query]):
    if isinstance(query_in, int):
        query = xavi.Query(**json.load(open("scenarios/queries/final_queries.json"))[f"s{scenario_id}"][query_in])
    elif isinstance(query_in, xavi.Query):
        query = query_in
    else:
        raise ValueError(f"Invalid input query given: {query_in}")

    causes_file = f"output/scenario_{scenario_id}/q_t{query.t_query}_m{query.type}.pkl"
    causes = pickle.load(open(causes_file, "rb"))

    # Split causes according to query type
    act_seg = None
    if query.type == xavi.QueryType.WHY or query.type == xavi.QueryType.WHY_NOT:
        f_exp, (cp, cf, (xp, yp, mp), (xf, yf, mf)) = causes
    elif query.type == xavi.QueryType.WHAT_IF:
        act_seg, f_exp, (cp, cf, (xp, yp, mp), (xf, yf, mf)) = causes
    elif query.type == xavi.QueryType.WHAT:
        act_seg = causes
        f_exp = None
        xp, yp, mp = None, None, None
        xf, yf, mf = None, None, None
        cp, cf = None, None
    else:
        raise ValueError(f"Unknown query type: {query.type}.")
    return query, f_exp, (cp, cf, (xp, yp, mp), (xf, yf, mf)), act_seg


if __name__ == '__main__':
    setup_xavi_logging()
    args = parse_eval_args()

    sid = args.scenario  # Which scenario to evaluate
    qix = args.query  # Query index to evaluate

    if args.robustness:
        df = {}
        queries = json.load(open("scenarios/queries/final_queries.json"))
        for sid in queries:
            for qix in queries[sid]:
                _, _, (_, _, (X_past, y_past, _), (X_future, y_future, _)), _ = load_data(sid[1], xavi.Query(**qix))
                eval_robustness(X_past, y_past, X_future, y_future)

    else:
        user_query, final_explanation, (coef_past, coef_future, (X_past, y_past, m_past),
                                        (X_future, y_future, m_future)), action_segment = load_data(sid, qix)

        if final_explanation is not None:
            final_explanation = final_explanation.drop(["term"])

        # Generate language explanation
        lang = xavi.Language(n_final=1, collision=sid in [2, 3])
        if coef_past is not None and coef_future is not None and \
                user_query.type == xavi.QueryType.WHY_NOT or user_query.negative:
            coef_past, coef_future = -coef_past, -coef_future
        s = lang.convert_to_sentence(user_query, final_explanation, (coef_past, coef_future), action_segment)
        logger.info(s)

        # Generate plots
        if final_explanation is not None:
            plot_dataframe(final_explanation, (coef_past, coef_future), save_path=f"output/scenario_{sid}")
            plt.show()
