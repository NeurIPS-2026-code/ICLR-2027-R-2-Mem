import json
import numpy as np

import os
from pathlib import Path

def get_total_tokens_and_iterations(base_path):
    """simple: return tokens and iterations"""
    base_path = Path(base_path)
    total_tokens = 0
    total_iterations = 0
    folder_count = 0
    folder_names = []
    
    for folder in base_path.iterdir():
        if folder.is_dir():
            # process qa_result.json
            qa_result_path = folder / 'qa_result.json'
            if qa_result_path.exists():
                folder_count += 1
                folder_names.append(folder.name)
                try:
                    with open(qa_result_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, dict) and 'tokens' in data:
                        total_tokens += data['tokens']
                except:
                    pass  # ignore
            
            # process research_trace.json
            research_trace_path = folder / 'research_trace.json'
            if research_trace_path.exists():
                try:
                    with open(research_trace_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, dict) and 'iterations' in data:
                        iterations_len = len(data['iterations'])
                        total_iterations += iterations_len
                except:
                    pass  # ignore
    
    return total_tokens, total_iterations, folder_count, folder_names


def get_total_tokens_and_iterations_2(base_path, folder_names):
    """return tokens and iterations"""
    base_path = Path(base_path)
    total_tokens = 0
    total_iterations = 0
    folder_count = 0
    
    for folder in base_path.iterdir():
        if folder.is_dir():
            
            if folder.name in folder_names:
                folder_count += 1
                # process qa_result.json
                qa_result_path = folder / 'qa_result.json'
                if qa_result_path.exists():
                    try:
                        with open(qa_result_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if isinstance(data, dict) and 'total_tokens' in data:
                            total_tokens += data['total_tokens']
                    except:
                        pass  # ignore
                
                # process research_trace.json
                research_trace_path = folder / 'research_trace.json'
                if research_trace_path.exists():
                    try:
                        with open(research_trace_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if isinstance(data, dict) and 'iterations' in data:
                            iterations_len = len(data['iterations'])
                            total_iterations += iterations_len
                    except:
                        pass  # ignore
            else:
                continue
    
    return total_tokens, total_iterations, folder_count






# eval_400, eval_1600, eval_3200

# base_path_exp = "..../eval_3200"
# base_path_noexp = "..../eval_3200"


base_path_exp = "....eval_400"
base_path_noexp = "....eval_400"

# use exp
print("========= exp =========")
total_tokens, total_iterations, folder_count, folder_names = get_total_tokens_and_iterations(base_path_exp)
print(f"process folders nums: {folder_count}")
print(f"tokens: {total_tokens}")
print(f"iterations sum: {total_iterations}")
with open(base_path_exp + "/batch_statistics_0_127.json", 'r', encoding='utf-8') as file:
    data1 = json.load(file)
print(f"AVG F1: {data1['avg_f1']}")


print("========= folder_names len =========")
print(len(folder_names))


# no exp
print("========= no exp =========")
total_tokens, total_iterations, folder_count = get_total_tokens_and_iterations_2(base_path_noexp, folder_names)
print(f"process folders nums: {folder_count}")
print(f"tokens: {total_tokens}")
print(f"iterations sum: {total_iterations}")
with open(base_path_noexp + "/batch_results_0_127.json", 'r', encoding='utf-8') as file:
    data2 = json.load(file)

cnt = 0
sum_F1 = 0
print(len(data2))
# Only compare those that have been run by exp
for data in data2:
    try:
        if data["sample_id"] in folder_names:
            sum_F1 = sum_F1 + data["f1"]
            cnt = cnt + 1
        else:
            continue
    except:
        print(f"errors: {data['sample_id']}")
        continue

print(f"sum cnt: {cnt}")
print(f"AVG F1: {sum_F1/cnt}")



# # test narrativeqa
# base_path_exp = "..../narrativeqa"
# base_path_noexp = "..../narrativeqa"

# # use exp
# print("========= exp =========")
# total_tokens, total_iterations, folder_count, folder_names = get_total_tokens_and_iterations(base_path_exp)
# print(f"process folders nums: {folder_count}")
# print(f"sum tokens: {total_tokens}")
# print(f"iterations sum: {total_iterations}")
# with open(base_path_exp + "/batch_statistics_30_299.json", 'r', encoding='utf-8') as file:
#     data1 = json.load(file)
# print(f"AVG F1: {data1['avg_f1']}")


# print("========= folder_names len =========")
# print(len(folder_names))


# # no exp
# print("========= no exp =========")
# total_tokens, total_iterations, folder_count = get_total_tokens_and_iterations_2(base_path_noexp, folder_names)
# print(f"process folders nums: {folder_count}")
# print(f"sum tokens: {total_tokens}")
# print(f"iterations sum: {total_iterations}")
# with open(base_path_noexp + "/batch_statistics_0_299.json", 'r', encoding='utf-8') as file:
#     data2 = json.load(file)
# print(f"AVG F1: {sum(data2['f1_scores'][-folder_count:])/folder_count}")