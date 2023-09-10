import json
import re

path = '../../useful/GDL/theorem.json'

with open(path, 'r', encoding='utf-8') as file:
    data = json.load(file)

    param_pattern = r"([a-zA-Z_]+)"
    theorem_list = [re.search(param_pattern, t).group(1) for t in data['Theorems']]
    print(len(theorem_list))
    with open('./theorem_sort.json', 'w') as f:
        json.dump(theorem_list, f)
