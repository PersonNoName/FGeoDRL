import copy

import numpy as np

from core.problem.problem import Problem
from core.aux_tools.parser import InverseParser as IvParser
from core.solver.engine import GeometryPredicateLogic as GeoLogic
from core.aux_tools.utils import *
from core.aux_tools.output import *
from core.aux_tools.parser import FormalLanguageParser as FLParser
from core.solver.engine import EquationKiller as EqKiller
from core.solver.fw_search import Theorem
import warnings
from core.aux_tools.parser import EquationParser as EqParser


def splitTheorem():
    pass


visited_node = {}


class PBranch:
    def __init__(self):
        self.visit_count = 0


class TBranch:
    def __init__(self, prior=None):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.pbranches = {}


class Node:
    def __init__(self, father, problem, state2node, theorem_GDL,
                 theorem_path='../Encoder/theorem_sort.json'):
        self.problem = problem  # instance of class <Problem>
        self.state = None  # tuple of <str>
        self.legal_moves = None  # [(t_name, t_para, t_branch)]
        self.conclusions = None  # {(t_name, t_para, t_branch): conclusions}
        self.solved = None  # <bool> problem solved or not

        self.probs = None  # {(t_name, t_para, t_branch): <float>}
        self.visits = 0  # <int>

        self.fathers = [father]  # father node
        self.children = {}  # {(t_name, t_para, t_branch): node}

        self.state2node = state2node  # {state: node}
        self.theorem_GDL = theorem_GDL  # theorem GDL

        # Test
        # branches保存的是可以选择的定理
        self.branches = {}
        self.theorem_order = None
        with open(theorem_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            self.theorem_order = [i for i in data]

        for move in self.get_legal_moves():
            theorem = self.theorem_order.index(move[0])
            if theorem not in self.branches:
                self.branches[theorem] = TBranch()
            if move[1] not in self.branches[theorem].pbranches:
                self.branches[theorem].pbranches[move[1]] = PBranch()

    def step(self, t_msg):
        if t_msg not in self.get_legal_moves():
            return False, None
        if self.probs[t_msg] == 0:
            return False, None
        if t_msg in self.children:
            return True, self.children[t_msg]

        t_name, t_para, t_branch = t_msg  # check algebra constraint
        letters = {}  # used for vars-letters replacement
        for i in range(len(self.theorem_GDL[t_name]["vars"])):
            letters[self.theorem_GDL[t_name]["vars"][i]] = t_para[i]
        gpl = copy.deepcopy(self.theorem_GDL[t_name]["body"][t_branch])
        for equal, item in gpl["algebra_constraints"]:
            oppose = False
            if "~" in equal:
                oppose = True
            eq = EqParser.get_equation_from_tree(self.problem, item, True, letters)
            solved_eq = False

            result, premise = EqKiller.solve_target(eq, self.problem)
            if result is not None and rough_equal(result, 0):
                solved_eq = True

            for i in range(len(self.conclusions[t_msg])):
                self.conclusions[t_msg][i][2] = list(self.conclusions[t_msg][i][2]) + premise

            if (not oppose and not solved_eq) or (oppose and solved_eq):
                self.probs[t_msg] = 0
                return False, None

        theorem = IvParser.inverse_parse_logic(t_name, t_para, self.theorem_GDL[t_name]["para_len"])
        child_problem = Problem()
        child_problem.load_problem_by_copy(self.problem)
        for predicate, item, premise in self.conclusions[t_msg]:
            child_problem.add(predicate, item, premise, theorem, skip_check=True)
        EqKiller.solve_equations(child_problem)
        child_problem.step(theorem, 0)

        child_node = Node(self, child_problem, self.state2node, self.theorem_GDL)
        child_node_state = child_node.get_state()
        if child_node_state in self.state2node:
            child_node = self.state2node[child_node_state]
            child_node.fathers.append(self)

        self.children[t_msg] = child_node

        return True, self.children[t_msg]

    def get_state(self):
        if self.state is not None:
            return self.state

        self.state = []
        anti_parsed_cdl = InverseParser.inverse_parse_logic_to_cdl(self.problem)
        for step in anti_parsed_cdl:
            for cdl in anti_parsed_cdl[step]:
                self.state.append(cdl)

        self.state = tuple(sorted(self.state))

        return self.state

    def get_legal_moves(self):
        if self.legal_moves is not None:
            return self.legal_moves

        self.legal_moves = []
        self.conclusions = {}
        for t_name in self.theorem_GDL:
            if t_name.endswith("definition") or Theorem.t_msg[t_name][1] == 0 or Theorem.t_msg[t_name][0] == 3:
                continue

            for t_branch in self.theorem_GDL[t_name]["body"]:
                gpl = copy.deepcopy(self.theorem_GDL[t_name]["body"][t_branch])
                r = GeoLogic.run_logic(gpl, self.problem)
                results = GeoLogic.make_conclusion(r, gpl, self.problem)  # get gpl reasoned result
                for letters, premise, conclusion in results:
                    t_para = tuple([letters[i] for i in self.theorem_GDL[t_name]["vars"]])
                    premise = tuple(premise)
                    conclusions = []
                    for predicate, item in conclusion:  # add conclusion
                        if self.problem.can_add(predicate, item, premise, t_name):
                            if predicate != "Equation":
                                item = tuple(item)
                            conclusions.append([predicate, item, premise])

                    if len(conclusions) > 0:
                        self.legal_moves.append((t_name, t_para, t_branch))
                        self.conclusions[(t_name, t_para, t_branch)] = conclusions

        init_probs = 1 / len(self.legal_moves)
        self.probs = {}
        for move in self.legal_moves:
            self.probs[move] = init_probs

        return self.legal_moves

    def get_solved(self):
        if self.solved is not None:
            return self.solved

        self.problem.check_goal()
        self.solved = self.problem.goal.solved
        return self.solved

    def reset_node(self):
        """
        将节点用于在Expand阶段时扩展出的节点用于Simulate阶段后进行重置，删除children
        """
        self.children = {}

    def moves(self):
        return self.branches.keys()

    def expected_value(self, move):
        branch = self.branches[move]
        if branch.visit_count == 0:
            return 0.0
        return branch.total_value / branch.visit_count

    def visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count

        return 0

    def has_child(self, theorem):
        return theorem in self.branches


class ForwardEnvironment:
    def __init__(self, predicate_GDL, theorem_GDL):
        """Initialize Environment."""
        self.predicate_GDL = FLParser.parse_predicate(predicate_GDL)
        self.theorem_GDL = FLParser.parse_theorem(theorem_GDL, self.predicate_GDL)
        self.state2node = {}
        self.root = None
        self.node = None
        self.goal = None

    def init_root(self, problem_CDL):
        problem = Problem()
        problem.load_problem_by_fl(self.predicate_GDL, FLParser.parse_problem(problem_CDL))
        EqKiller.solve_equations(problem)
        problem.step("init_problem", 0)

        self.root = Node(None, problem, self.state2node, self.theorem_GDL)
        self.reset()
        self.goal = problem_CDL["goal_cdl"]

    def reset(self):
        self.node = self.root
        self.node.visits += 1

    def step(self, t_msg):
        stepped, child = self.node.step(t_msg)
        # 测试
        global visited_node

        if stepped:
            self.node = child
            self.node.visits += 1
            if child.get_state() in visited_node:
                print("已经出现过该节点了,请检查是否出现问题", child.get_state(), t_msg)
            else:
                visited_node[child.get_state()] = child

        return stepped

    def get_state(self):
        return self.node.get_state()

    def get_legal_moves(self):
        return self.node.get_legal_moves()

    def get_solved(self):
        return self.node.get_solved()

    def get_probs(self):
        return self.node.probs

    def set_probs(self, probs):
        self.node.probs = probs

    def get_visits(self):
        return self.node.visits


class TreeAgent:
    def __init__(self,
                 env,
                 rl_network,
                 sl_network,
                 collector=None,
                 simulation_num=30,
                 rollout_num=80,
                 c=2.0,
                 lamda=0.5,
                 state_dim=(10, 10),
                 action_dim=197):
        self.env = env
        self.rl_model = rl_network
        self.sl_model = sl_network
        self.cuda = True

        self.collector = collector
        self.rollout_num = rollout_num
        self.simulation_num = simulation_num
        self.c = c
        self.lamda = lamda
        self.state_dim = state_dim
        self.action_dim = action_dim

    def get_prior(self):
        return

    def select_branch(self, node):
        total_n = node.visits

        def score_branch(move):
            q = node.expected_value(move)
            p = node.prior(move)
            n = node.visit_count(move)
            # 注意此时的p还没有赋值
            return q + self.c * p * np.sqrt(total_n) / (n + 1)

        next_move = max(node.moves(), key=score_branch)
        return next_move

    def select_move(self, game_state):
        # 注意Step的时候会自动创建Node,所以将每次访问过的节点放在step中进行保存
        global visited_node
        if game_state in visited_node:
            root = visited_node[game_state]
        else:
            print("visited没有在step中保存该节点，请检查原因")
            return

        # 获取最近的move
        move = None

        for i in range(self.simulation_num):
            node = root

            next_move = self.select_branch(node)
            while node.has_child(next_move):
                # 需要进行特殊处理，因为branch是只有Theorem而没有param
                node = node.step()

    # 创建game_state的节点


if __name__ == '__main__':
    path_preset = "../../data/preset/"
    path_formalized = "../../data/formalized-problems/"
    warnings.filterwarnings("ignore")

    env = ForwardEnvironment(load_json(path_preset + "predicate_GDL.json"),  # init solver
                             load_json(path_preset + "theorem_GDL.json"))
    filename = "{}.json".format(1584)
    if filename not in os.listdir(path_formalized):
        print("No file \'{}\' in \'{}\'.\n".format(filename, path_formalized))
    env.init_root(load_json(path_formalized + filename))
    print(env.get_state())
    for theorem in env.node.branches:
        print(theorem)
        for param in env.node.branches[theorem].pbranches:
            print(param,end=' ')
        print()
    # print(env.get_legal_moves())
