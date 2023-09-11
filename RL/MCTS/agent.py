import warnings
from solver.problem.problem import Problem
from solver.core.engine import GeometryPredicateLogic as GeoLogic
from solver.aux_tools.utils import load_json
from solver.aux_tools.parser import InverseParserM2F
from solver.aux_tools.parser import GDLParser, CDLParser
from solver.core.engine import EquationKiller

path_gdl = "../../datasets/gdl/"
path_problems = "../../datasets/problems/"


class SNode:
    """Super node that store all information."""

    def __init__(self, father, problem):
        self.fathers = [father]  # <PNode> or None, father node

        self.problem = problem  # instance of <Problem>
        self.children = None  # <dict>, {tuple('t_name', 't_branch'): <TNode>}
        self.conclusions = None  # {(t_name, t_para, t_branch): conclusions}
        self.state = None  # current problem state
        self.solved = None  # indicates whether the problem has been solved

        self.prior = 0  # probability of selecting the current node
        self.visits = 0  # number of visits to the current node
        self.value = 0  # evaluation value of the current node

    def get_legal_moves(self, theorem_GDL):
        """
        Build SNode-TNode-PNode tree and return legal moves of current SNode.
        :param theorem_GDL: theorem GDL.
        :return legal_moves: <list> of tuple('t_name', 't_branch').
        """
        if self.children is not None:
            return self.children.keys()

        self.children = {}
        self.conclusions = {}
        for t_name in theorem_GDL:
            if t_name.endswith("definition"):
                continue
            for t_branch in theorem_GDL[t_name]["body"]:
                gpl = theorem_GDL[t_name]["body"][t_branch]
                r = GeoLogic.run_logic(gpl, self.problem)
                results = GeoLogic.make_conclusion(r, gpl, self.problem)  # get gpl reasoned result
                for letters, premise, conclusion in results:
                    t_para = tuple([letters[i] for i in theorem_GDL[t_name]["vars"]])
                    premise = tuple(premise)
                    selected_conclusion = []
                    for predicate, item in conclusion:  # add conclusion
                        if self.problem.check(predicate, item, premise, t_name):
                            if predicate != "Equation":
                                item = tuple(item)
                            selected_conclusion.append((predicate, item, premise))

                    if len(selected_conclusion) > 0:
                        self.conclusions[(t_name, t_branch, t_para)] = selected_conclusion  # add conclusions
                        if (t_name, t_branch) not in self.children:  # add child
                            self.children[(t_name, t_branch)] = [t_para]
                        else:
                            self.children[(t_name, t_branch)].append(t_para)

        t_node_prior = 1 / len(self.children)
        for t_name_and_branch in self.children:
            t_node = TNode(self, t_name_and_branch)
            t_node.prior = t_node_prior

            t_node.children = {}
            p_node_prior = 1 / len(self.children[t_name_and_branch])
            for t_para in self.children[t_name_and_branch]:
                p_node = PNode(t_node, t_para)
                p_node.prior = p_node_prior
                t_node.children[t_para] = p_node  # add p_node

            self.children[t_name_and_branch] = t_node  # add t_node

        return self.children.keys()

    def step(self, t_name_and_branch):
        """
        Attempt to move to the next node based on the move.
        :param t_name_and_branch: tuple('t_name', 't_branch').
        :return moved: <bool>, indicate whether Successfully moved or not.
        :return child_node: <TNode>, node moved to.
        """
        if t_name_and_branch in self.children:
            next_node = self.children[t_name_and_branch]
            return next_node

        e_msg = "<{}> is not a legal move.".format(t_name_and_branch)
        raise Exception(e_msg)

    def get_state(self):
        """
        Return current state.
        :return state: <tuple>.
        """
        if self.state is not None:
            return self.state

        self.state = []
        anti_parsed_cdl = InverseParserM2F.inverse_parse_logic_to_cdl(self.problem)
        for step in anti_parsed_cdl:
            for cdl in anti_parsed_cdl[step]:
                self.state.append(cdl)

        self.state = tuple(sorted(self.state))

        return self.state

    def get_solved(self):
        """
        Check if the problem has been solved.
        :return solved: <bool>.
        """
        if self.solved is not None:
            return self.solved

        self.problem.check_goal()
        self.solved = self.problem.goal.solved
        return self.solved


class TNode:
    """Theorem Node."""

    def __init__(self, father, t_name_and_branch):
        self.father = father  # <SNode>
        self.t_name_and_branch = t_name_and_branch  # tuple('t_name', 't_branch')
        self.children = None  # <dict>, {'t_para': <PNode>}

        self.prior = 0  # probability of selecting the current node
        self.visits = 0  # number of visits to the current node
        self.value = 0  # evaluation value of the current node

    def step(self, t_para, theorem_GDL, state2node):
        """
        Apply theorem and Attempt to move to the next s_node based on the move.
        :param t_para: <str>.
        :param theorem_GDL: theorem GDL.
        :param state2node: map state to node.
        :return moved: <bool>, indicate whether Successfully moved or not.
        :return child_node: <SNode>, node moved to.
        """
        if t_para not in self.children:
            e_msg = "<{}> is not a legal move.".format(t_para)
            raise Exception(e_msg)

        p_node = self.children[t_para]
        if p_node.child is not None:
            return True, p_node.child

        t_name, t_branch = self.t_name_and_branch
        problem = self.father.problem
        conclusion = self.father.conclusions[(t_name, t_branch, t_para)]

        letters = {}
        for i in range(len(theorem_GDL[t_name]["vars"])):  # build conclusion
            letters[theorem_GDL[t_name]["vars"][i]] = t_para[i]
        gpl = theorem_GDL[t_name]["body"][t_branch]
        for equal, item in gpl["algebra_constraints"]:  # check algebra constraint
            oppose = False
            if "~" in equal:
                oppose = True
            eq = CDLParser.get_equation_from_tree(problem, item, True, letters)
            solved_eq = False

            result, premise = EquationKiller.solve_target(eq, problem)
            if result is not None and result == 0:
                solved_eq = True

            for i in range(len(conclusion)):
                conclusion[i][2] = list(conclusion[i][2]) + premise

            if (not oppose and not solved_eq) or (oppose and solved_eq):
                self.father.conclusions.pop((t_name, t_branch, t_para))
                self.children.pop(t_para)
                return False, None

        theorem = InverseParserM2F.inverse_parse_one_theorem(t_name, t_branch, t_para, theorem_GDL)
        child_problem = Problem()
        child_problem.load_problem_by_copy(problem)
        update = False
        for predicate, item, premise in conclusion:
            update = child_problem.add(predicate, item, premise, theorem, skip_check=True) or update
        EquationKiller.solve_equations(child_problem)
        child_problem.step(theorem, 0)

        if not update:
            self.father.conclusions.pop((t_name, t_branch, t_para))
            self.children.pop(t_para)
            return False, None

        child_node = SNode(p_node, child_problem)
        child_node_state = child_node.get_state()
        if child_node_state in state2node:
            child_node = state2node[child_node_state]
            child_node.fathers.append(p_node)
        else:
            state2node[child_node_state] = child_node

        p_node.child = child_node

        return True, child_node


class PNode:
    """Theorem Parameter Node."""

    def __init__(self, father, t_para):
        self.father = father  # <TNode>, father node
        self.t_para = t_para    # <str>
        self.child = None  # <SNode> or None

        self.prior = 0  # probability of selecting the current node
        self.visits = 0  # number of visits to the current node
        self.value = 0  # evaluation value of the current node


class ForwardEnvironment:
    def __init__(self, predicate_GDL, theorem_GDL):
        """Initialize Environment."""
        self.predicate_GDL = GDLParser.parse_predicate_gdl(predicate_GDL)
        self.theorem_GDL = GDLParser.parse_theorem_gdl(theorem_GDL, self.predicate_GDL)
        self.state2node = None
        self.goal = None
        self.root = None
        self.node = None

    def init_root(self, problem_CDL):
        """Start a new simulation."""
        self.state2node = {}
        self.goal = problem_CDL["goal_cdl"]

        problem = Problem()
        problem.load_problem_by_fl(self.predicate_GDL, CDLParser.parse_problem(problem_CDL))
        EquationKiller.solve_equations(problem)
        problem.step("init_problem", 0)
        self.root = SNode(None, problem)
        self.reset()

    def reset(self):
        """Back to the root node."""
        if self.node != self.root:
            self.node = self.root
            self.node.visits += 1

    def step(self, move=None):
        """
        Move to the next state.
        SNode --> TNode, TNode --> (PNode -->) SNode.
        """
        if isinstance(self.node, SNode):    # SNode
            child = self.node.step(move)
            stepped = True
        else:    # TNode
            stepped, child = self.node.step(move, self.theorem_GDL, self.state2node)

        if stepped:
            self.node = child
            self.node.visits += 1

        return stepped

    def get_state(self):
        """
        Get state of current node.
        SNode: <list>, conditions + goal
        TNode: <list>, conditions + goal + t_name_and_branch
        >> get_state()
        [('CongruentBetweenTriangle(RST,XYZ)', 'Equation(ll_tr-x-21)')]
        >> get_state()
        [('CongruentBetweenTriangle(RST,XYZ)', 'Equation(ll_tr-x-21)', congruent_triangle_property_angle_equal(1)]
        """
        if isinstance(self.node, SNode):
            return list(self.node.get_state()) + [self.goal]
        else:
            return list(self.node.father.get_state()) + [self.goal] + \
                ["{}({})".format(self.node.t_name_and_branch[0], self.node.t_name_and_branch[1])]

    def get_legal_moves(self):
        """
        Get state of current node.
        SNode: <list> of tuple('t_name', 't_branch')
        TNode: <list> of 't_para'
        """
        if isinstance(self.node, SNode):
            return list(self.node.get_legal_moves(self.theorem_GDL))
        else:
            return list(self.node.children.keys())

    def backward(self, value):
        pass


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    env = ForwardEnvironment(load_json(path_gdl + "predicate_GDL.json"),  # init env
                             load_json(path_gdl + "theorem_GDL.json"))
    pid = input("pid:")
    env.init_root(load_json(path_problems + "{}.json".format(pid)))

    print(env.node.get_solved())
    print(env.get_state())
    print(env.get_legal_moves())
    env.step(('congruent_triangle_property_angle_equal', '1'))
    print(env.get_state())
    print(env.get_legal_moves())
    env.step(('R', 'S', 'T', 'X', 'Y', 'Z'))
    print(env.node.get_solved())
