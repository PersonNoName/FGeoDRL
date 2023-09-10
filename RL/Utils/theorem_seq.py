# 用以save和load每个题目的定理序列
import json
import re


def save_theorem_seq(file_name, path, save_path):
    try:
        with open(path + str(file_name) + '.json', 'r', encoding='utf-8') as file:
            data = json.load(file)

            if 'notes' in data or \
                    'note' in data or \
                    'Notes' in data or \
                    'Note' in data:
                print('保存第{}题定理序列失败'.format(file_name))
                return
            theorem_pattern = r"([a-z_A-Z]+)\("
            param_pattern = r"\(([\S\s]+)\)"

            theorem_seq = {'theorem_seqs': [],
                           'param_seqs': []}

            for t in data['theorem_seqs']:
                theorem_match = re.search(theorem_pattern, t).group(1)
                param_match = re.search(param_pattern, t).group(1)

                theorem_seq['theorem_seqs'].append(theorem_match)
                theorem_seq['param_seqs'].append(param_match)

            with open(save_path + str(file_name) + '.json', 'w') as new_file:
                json.dump(theorem_seq, new_file)

    except Exception as e:
        print('保存第{}题定理序列失败'.format(file_name) + str(e))
        return


def get_theorem_seq(file_name, path='./Seqs/'):
    try:
        with open(path + str(file_name) + '.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
            params = []

            for param in data['param_seqs']:
                params.append(param.replace(",", ""))

            return data['theorem_seqs'], params

    except Exception as e:
        print('打开第{}题定理序列失败'.format(file_name) + str(e))
        return


if __name__ == '__main__':
    path = "../../data/formalized-problems/"
    save_path = '../Seqs/'
    # save_theorem_seq(8644, path, save_path)

    data = get_theorem_seq(1584, save_path)
    print(data)
