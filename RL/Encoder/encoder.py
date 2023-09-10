import json
import os
import re
import sys

import numpy as np


def splitEquation(sequence):
    # 去除不必要信息
    sequence = sequence.replace('sin', '')
    sequence = sequence.replace('cos', '')

    param_pattern = r"([a-zA-Z_]+)"
    param_list = sorted(re.findall(param_pattern, sequence))
    param_dict = {}
    for param in param_list:
        result = param.split('_')
        try:
            if result[0] not in param_dict:
                param_dict[result[0]] = []
        except Exception as e:
            print(param, e, "splitEquation中Equation有问题")
        if len(result) == 2:
            param_dict[result[0]].append(result[1])
        elif len(result) == 1:
            param_dict[result[0]].append('')
    return param_dict


def splitCondition(state):
    condition_pattern = r"([a-zA-Z]+)\("
    param_pattern = r"\(([\S\s]+)\)"

    condition = {}

    for s in state:
        condition_match = re.search(condition_pattern, s).group(1)
        param_match = re.search(param_pattern, s).group(1)
        param = param_match.split(',')
        if condition_match in condition:
            condition[condition_match].append(param)
        else:
            condition[condition_match] = [param]
    # 进行预排序
    for c in condition:
        condition[c] = sorted(condition[c])
    return condition


def splitGoal(state, goal, path='./equation_transfer.json'):
    with open(path, 'r', encoding='utf-8') as f:
        goal_transfer = json.load(f)
    condition_pattern = r'([a-zA-Z_]+)\(([a-zA-Z,]+|[-+*/a-z0-9]+)\)'
    # condition_pattern = r'([a-zA-Z_]+)\(([-+*/a-zA-Z0-9]+)\)'
    # param_pattern = r"\(([\S\s]+)\)"
    exclude = ['Value', 'Sin', 'Cos', 'Tan']
    condition_match = re.findall(condition_pattern, goal)
    # 将代数转换为原本模样 f_a --> MeasureofAngle(xxx)
    if condition_match[0][0] in exclude:
        formalized = {}
        split_equation = splitCondition(state)['Equation']
        for alpha in re.findall(r"[a-z]", condition_match[0][1]):
            formalized[alpha] = None
        for equation in split_equation:
            algebra = splitEquation(equation[0])
            if 'f' in algebra:
                for alpha in algebra['f']:
                    if alpha not in formalized:
                        continue
                    elif not formalized[alpha]:
                        c = [i for i in algebra if i != 'f'][0]
                        try:
                            formalized[alpha] = (goal_transfer[c], alpha)
                        except Exception as e:
                            print('转换Goal的时候发生错误：', e)
        for f in formalized:
            condition_match.append(formalized[f])
        # 删除原有的Value
        del condition_match[0]
    return condition_match
    # print(param_match)


def oneHotEncode(param):
    param_encode = np.zeros((8, 26))

    for i in range(len(param)):
        if 'a' < param[i] < 'z':
            print('one hot编码失败，因为定理参数小写不符合')
            return
        pos = ord(param[i]) - ord('A')
        param_encode[i][pos] = 1

    return param_encode


class Encoder:
    def __init__(self,
                 condition_path='./condition_sort.json',
                 equation_path='./equation_sort.json',
                 goal_path='./goal_sort.json',
                 theorem_path='./theorem_sort.json'):
        self.encoder_order = None
        self.equation_order = None
        self.goal_order = None
        self.theorem_order = None
        with open(condition_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            self.encoder_order = [i for i in data]
        with open(equation_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            self.equation_order = [i for i in data]
        with open(goal_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            self.goal_order = [i for i in data]
        with open(theorem_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            self.theorem_order = [i for i in data]

        self.state = None
        self.x_dim = 30
        self.y_dim = 20

    # 注意：传入的goal是已经分割后的结果
    def encoder(self, state, goal=None, theorem=None):
        # 初始化encoder之后的状态
        self.state = np.zeros((len(self.encoder_order) + 1, self.y_dim, self.x_dim))

        conditions = splitCondition(state)
        # 用于处理未出现在已记录的条件
        non_exist_record_y = 0
        for condition in conditions:
            params = conditions[condition]

            if condition != 'Equation' and condition in self.encoder_order:
                pos = self.encoder_order.index(condition)
                exist_record_y = 0
                for param in params:
                    exist_record_x = 0
                    for p in param:
                        for i in range(len(p)):
                            try:
                                self.state[pos][exist_record_y][exist_record_x] = (ord(p[i]) - ord('A') + 1) * 0.02
                                exist_record_x += 1
                            except Exception as e:
                                print('Exist Condition 编码已超过状态维度限制:', e, conditions)
                                return None
                        exist_record_x += 1
                    exist_record_y += 1
            elif condition not in self.encoder_order:
                for param in params:
                    non_exist_record_x = 0
                    for p in param:
                        for i in range(len(p)):
                            try:
                                self.state[-1][non_exist_record_y][non_exist_record_x] = (ord(p[i]) - ord(
                                    'A') + 1) * 0.02
                                non_exist_record_x += 1
                            except Exception as e:
                                print('Non Exist Condition 编码已超过状态维度限制:', e, conditions)
                                return None
                        non_exist_record_x += 1
                    non_exist_record_y += 1
            elif condition == 'Equation':
                pos = self.encoder_order.index(condition)
                # param: ['-110*f_x+ll_bc-53']
                record_y = 0
                for param in params:
                    equation_dict = splitEquation(param[0])
                    record_x = 0
                    # e: f, ll
                    for e in equation_dict:
                        if e not in self.equation_order:
                            print("该条件不存在，请及时添加", e)
                            return None
                        order = self.equation_order.index(e)
                        # p: x
                        # p: bc
                        for p in equation_dict[e]:
                            try:
                                self.state[pos][record_y][record_x] = (order + 1) * 0.3
                                record_x += 1
                            except Exception as e:
                                print('Equation Condition 编码已超过状态维度限制:', e, conditions)
                                return None
                            for i in range(len(p)):
                                try:
                                    self.state[pos][record_y][record_x] = (ord(p[i]) - ord('A') + 1) * 0.02
                                    record_x += 1
                                except Exception as e:
                                    print('Equation Condition 编码已超过状态维度限制:', e, conditions)
                                    return None
                            record_x += 1
                    record_y += 1
            # print(condition, ":", conditions[condition])
            # print(self.state[self.encoder_order.index(condition)][:3])
        # 将目标结果载入状态中
        if goal:
            y_dim = 0
            for attr, params in goal:
                x_dim = 20

                params = params.split(',')
                pos = self.goal_order.index(attr)
                self.state[0][y_dim][pos] = 1
                for param in params:
                    for p in param:
                        self.state[0][y_dim][x_dim] = (ord(p) - ord('A') + 1) * 0.02
                        x_dim += 1
                    x_dim += 1
                y_dim += 1

        elif theorem:
            idx = self.theorem_order.index(theorem)
            x_dim = idx % self.x_dim
            y_dim = int(idx / self.x_dim)

            self.state[0][y_dim][x_dim] = 1
        return self.state


if __name__ == '__main__':
    import warnings
    import sys
    from core.API import ForwardEnvironment
    from core.aux_tools.utils import load_json

    sys.path.append('../Utils')
    from RL.Utils import theorem_seq

    path_preset = "../../data/preset/"
    path_formalized = "../../data/formalized-problems/"
    warnings.filterwarnings("ignore")

    env = ForwardEnvironment(load_json(path_preset + "predicate_GDL.json"),  # init solver
                             load_json(path_preset + "theorem_GDL.json"))
    for pid in range(3554, 3556):
        filename = "{}.json".format(pid)
        if filename not in os.listdir(path_formalized):
            print("No file \'{}\' in \'{}\'.\n".format(filename, path_formalized))
            continue
        try:
            env.init_root(load_json(path_formalized + filename))
        except Exception as e:
            print(e)
            continue
        state = env.get_state()
        # condition = splitCondition(state)
        # for e in condition['Equation']:
        #     print(e, '  ', end=' ')
        #     splitEquation(e[0])
        # splitEquation()
        # print(condition)
        # print(env.goal)
        encode = Encoder()
        # theorems, params = theorem_seq.get_theorem_seq(pid, path='../Seqs/')
        # state = encode.encoder(state, theorem=theorems[0])
        # print(theorems, params)
        # print(state)
        goal = splitGoal(state, env.goal)
        # with open('../goal.json', 'r', encoding='utf-8') as file:
        #     data = json.load(file)
        #

        new_state = encode.encoder(state, goal)
