import igp2 as ip
import numpy as np
import matplotlib.pyplot as plt

scenario_map = ip.Map.parse_from_opendrive("scenarios/maps/scenario3.xodr")
ip.plot_map(scenario_map, midline=True)
plt.show()

h = -np.pi / 2
state = ip.AgentState(
    time=0,
    position=np.array([-2, -40]),
    velocity=10 * np.array([np.cos(h), np.sin(h)]),
    acceleration=np.array([0., 0.]),
    heading=h
)
frame = {0: state}
exit = ip.Exit.get_possible_args(state, scenario_map)
print(ip.ChangeLaneRight.applicable(state, scenario_map))
print(scenario_map.best_lane_at(state.position, state.heading).link.predecessor)
ip.plot_map(scenario_map, midline=True)
plt.plot(*state.position, marker="o")
for dic in exit:
    plt.plot(*dic["turn_target"], marker="x")
plt.show()
