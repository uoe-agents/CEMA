# This file generates the plots in the paper from the saved causes
# that can be generated using run.py and the final_causes.json query file.
import pickle
import json
import sys

import xavi

import matplotlib.pyplot as plt
from scenarios.util import setup_xavi_logging
from xavi.plotting import plot_dataframe

if __name__ == '__main__':
    setup_xavi_logging()

    scenario = 1  # Which scenario to evaluate
    query_no = 0  # Query index to evaluate
    query = xavi.Query(**json.load(open("scenarios/queries/final_queries.json"))[f"s{scenario}"][query_no])
    causes_file = f"output/scenario_{scenario}/q_t{query.t_query}_m{query.type}.pkl"
    causes = pickle.load(open(causes_file, "rb"))

    # Split causes according to query type
    action_segment = None
    if query.type == xavi.QueryType.WHY or query.type == xavi.QueryType.WHY_NOT:
        final_expl, (coef_past, coef_future,
                     (X_past, y_past, m_past),
                     (X_future, y_future, m_future)) = causes
    elif query.type == xavi.QueryType.WHAT_IF:
        action_segment, final_expl, (coef_past, coef_future,
                                     (X_past, y_past, m_past),
                                     (X_future, y_future, m_future)) = causes
    elif query.type == xavi.QueryType.WHAT:
        pass

    if final_expl is not None:
        final_expl = final_expl.drop(["term"])

    lang = xavi.Language()
    s = lang.convert_to_sentence(query, final_expl, (coef_past, coef_future), action_segment)
    print(s)

    # Generate plots
    plot_dataframe(final_expl, (coef_past, coef_future), save_path=f"output/scenario_{scenario}")
    plt.show()
