import json
import os
import re
import warnings

from MCTS.Agent import ForwardEnvironment
from solver.aux_tools.utils import load_json
import numpy as np
from Utils.theorem_seq import get_theorem_seq
from Encoder import splitGoal
from Encoder import oneHotEncode
from Utils.dataset import Dataset

total_num = 0


def splitLegalMoves(legal_moves):
    Moves = {}

    for move in legal_moves:
        theorem = move[0]
        param = ''

        for p in move[1]:
            param += p

        if theorem not in Moves:
            Moves[theorem] = {param: move}
        else:
            Moves[theorem][param] = move

    return Moves


def testEncode(pid, env, encoder, encode_goal, encode_theorem, theorems_list, params_list):
    filename = "{}.json".format(pid)
    with open('./Encoder/theorem_sort.json') as file:
        theore_order = json.load(file)
    if filename not in os.listdir(path_formalized):
        print("No file \'{}\' in \'{}\'.\n".format(filename, path_formalized))
        return
    try:
        env.init_root(load_json(path_formalized + filename))
    except Exception as e:
        print('{}题目不可初始化'.format(pid), e)
        return
    goal = splitGoal(env.get_state(), env.goal, path='./Encoder/equation_transfer.json')
    solved = env.get_solved()

    try:
        theorems, params = get_theorem_seq(pid)
    except Exception as e:
        print('{}题目已被排除'.format(pid), e)
        return

    cur_seq = 0
    # 测试
    global total_num
    # print('start state: ', env.get_state())
    for i in range(len(theorems)):
        state = env.get_state()

        encode_goal.append(encoder.encoder(state=state, goal=goal))
        encode_theorem.append(encoder.encoder(state=state, theorem=theorems[cur_seq]))
        theorems_list.append(theore_order.index(theorems[cur_seq]))
        params_list.append(oneHotEncode(params[cur_seq]))
        total_num += 1

        moves = env.get_legal_moves()
        split_moves = splitLegalMoves(moves)
        # print('初始动作', moves)
        # print('分割后的动作', split_moves)
        try:
            cur_move = split_moves[theorems[cur_seq]][params[cur_seq]]
        except Exception as e:
            print('{}题目中'.format(pid) + '目标move不存在, 不可解', e)
            return
        # print('待执行动作', cur_move)
        try:
            stepped = env.step(cur_move)
        except Exception as e:
            print('{}题目中'.format(pid) + '定理无法运行', e)
            return

        solved = env.get_solved()
        cur_seq += 1

    if solved:
        print("成功")
        return True
    else:
        print("失败")
        return False


if __name__ == '__main__':
    path_preset = "../data/preset/"
    path_formalized = "../data/formalized-problems/"
    warnings.filterwarnings("ignore")

    env = ForwardEnvironment(load_json(path_preset + "predicate_GDL.json"),  # init solver
                             load_json(path_preset + "theorem_GDL.json"))


    def getMostLenEquation():
        from Encoder import Encoder
        SL_theorem_Data = Dataset(mode="SL_theorem", save_path='./Database/SL/pretrain/theorem/E01.h5')
        SL_param_Data = Dataset(mode="SL_param", save_path='./Database/SL/pretrain/param/E01.h5')
        with open('./pids.json', 'r', encoding='utf-8') as file:
            data = json.load(file)

            encode_goal = []
            encode_theorem = []
            theorems_list = []
            params_list = []

            encoder = Encoder(condition_path='./Encoder/condition_sort.json',
                              equation_path='./Encoder/equation_sort.json',
                              goal_path='./Encoder/goal_sort.json',
                              theorem_path='./Encoder/theorem_sort.json')
            i = 0
            # data['solved']
            for pid in data['solved']:
                if pid <= 8859:
                    continue
                env = ForwardEnvironment(load_json(path_preset + "predicate_GDL.json"),  # init solver
                                         load_json(path_preset + "theorem_GDL.json"))
                filename = "{}.json".format(pid)
                if filename not in os.listdir(path_formalized):
                    print("No file \'{}\' in \'{}\'.\n".format(filename, path_formalized))
                    continue
                print('正在处理第{}个问题...'.format(pid), end=' ')
                testEncode(pid, env, encoder, encode_goal, encode_theorem, theorems_list, params_list)
                i += 1

                if i > 100 and i % 200 == 0:
                    SL_theorem_Data.addDataset(state=np.array(encode_goal),
                                               theorem=np.array(theorems_list).reshape((-1, 1)))
                    SL_param_Data.addDataset(state=np.array(encode_theorem),
                                             param=np.array(params_list))
                    encode_goal.clear()
                    encode_theorem.clear()
                    theorems_list.clear()
                    params_list.clear()
                    print('total_num', total_num)

        if len(encode_goal) != 0:
            SL_theorem_Data.addDataset(state=np.array(encode_goal),
                                       theorem=np.array(theorems_list).reshape((-1, 1)))
            SL_param_Data.addDataset(state=np.array(encode_theorem),
                                     param=np.array(params_list))
        print('total_num', total_num)
    getMostLenEquation()
