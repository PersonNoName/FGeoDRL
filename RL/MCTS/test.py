import os

from agent import ForwardEnvironment
from solver.aux_tools.utils import load_json
import warnings

path_preset = "../../data/preset/"
path_formalized = "../../data/formalized-problems/"
warnings.filterwarnings("ignore")

env = ForwardEnvironment(load_json(path_preset + "predicate_GDL.json"),  # init solver
                         load_json(path_preset + "theorem_GDL.json"))

pid = 1584
filename = "{}.json".format(pid)
if filename not in os.listdir(path_formalized):
    print("No file \'{}\' in \'{}\'.\n".format(filename, path_formalized))
env.init_root(load_json(path_formalized + filename))

state = env.get_state()
print(state)
d = {state: env.node}
for action in env.get_legal_moves():
    if env.step(action):
        print("成功进入", env.get_state())
        if env.get_state() not in d:
            env.node = d[state]
            break
print(env.get_state())
for action in env.get_legal_moves():
    if env.step(action):
        print("成功进入", env.get_state())
        if env.get_state() not in d:
            env.node = d[state]
            break
