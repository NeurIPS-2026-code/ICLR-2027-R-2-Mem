import sys
from pathlib import Path

import json
from openai import OpenAI
from gam.generator import AbsGenerator
from gam.generator.vllm_generator import VLLMGenerator
from judge_model import *
from exp.prompts.self_reflection_prompts import (
    Planning_system_prompt, 
    Planning_prompt, 
    Planning_prompt_extra,
    Reflection_system_prompt, 
    Reflection_prompt
)
import re
from gam import (
    InMemoryMemoryStore,
    InMemoryPageStore,
    IndexRetriever,
    BM25Retriever,
    DenseRetriever,
    IndexRetrieverConfig,
    BM25RetrieverConfig,
    DenseRetrieverConfig,
    ResearchAgent,
    OpenAIGenerator,
    OpenAIGeneratorConfig,
    VLLMGenerator,
    VLLMGeneratorConfig,
)
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import math
import os
import random

# from exp_replay import ReplayEngine
DEEPSEEK_API_key = "your_api_key"
GPT_API_key = "your_api_key"

# Parameter settings
model_size = "7B"
model_name = f"qwen2.5-7B-Instruct"
model_type = "vllm"     # vllm or openai
bge_model_path = "your_bge_model"   # BAAI/bge-m3

# One seed is used per manual run. Replace it with either commented value for
# the other two runs reported in the paper; this script intentionally runs once.
RANDOM_SEED = 43
# RANDOM_SEED = 42
# RANDOM_SEED = 44
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# vllm url
base_url = "http://localhost:8001/v1"

# Select dataset
datasets = "locomo"
path_locomo = "locomo_7B"

# Experience bank path
expbank_path = "expbank"
os.makedirs(expbank_path, exist_ok=True)

# hotpotqa dataset
# hotpotqa_data = "eval_400"
# results_file = f".../{hotpotqa_data}/batch_statistics_0_127.json"
# base_dir = f".../hotpotqa_qwen2.5_7B/{hotpotqa_data}/batch_results_0_127.json"

max_samples = 25    # (20%)

# narrativeqa_base_dir
# narrativeqa_base_dir = ".../narrativeqa/batch_results_0_299.json"

# exp
Planning_exp_path = f"{expbank_path}/Planning_exp_{model_size}_X.json"
Reflection_exp_path = f"{expbank_path}/Reflection_exp_{model_size}_X.json"
name_Planning = f"{expbank_path}/Planning_{model_size}"
name_Reflection = f"{expbank_path}/Reflection_{model_size}"

# If using OpenAI (or third-party API gateway):
# working_model = "Pro/Qwen/Qwen2.5-7B-Instruct"
# working_api_key = "your api key"
# working_base_url = "https://api.siliconflow.cn/v1"

High_threshold = 10
Low_threshold = 5

if model_type == "vllm":
    student_model = VLLMGenerator(
        {
            "model_name": model_name,
            "base_url": base_url,
            "api_key": "empty",
            "temperature": 0.3,
            "max_tokens": 1024,
            "use_schema": True
        }
    )
else:
    working_generator_config = OpenAIGeneratorConfig(
                    model_name=working_model,
                    api_key=working_api_key,
                    base_url=working_base_url,
                    temperature=0.3,
                    max_tokens=256
                )
    student_model = OpenAIGenerator(working_generator_config.__dict__)

# Try setting global variables
Planning_scores = {score: 0 for score in range(13)}
Reflection_scores = {score: 0 for score in range(13)}


def exp_bank(name, data):
    MODEL_PATH = bge_model_path
    model = SentenceTransformer(
        MODEL_PATH,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    # Recommended: normalize for inner product ≈ cosine similarity
    model.max_seq_length = 512

    # Use composite embedding
    situations = [f"({item['situation']})" + item['condition'] for item in data]

    embeddings = model.encode(
        situations,
        normalize_embeddings=True,   # Very important
        batch_size=32,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings).astype("float32")
    print(embeddings.shape)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print("Vector database size:", index.ntotal)

    metadata = []
    for item in data:
        metadata.append({
            "condition": item["condition"],
            "situation": item["situation"],
            "summary": item["summary"],
            "experience": item["experience"]
        })

    faiss.write_index(index, f"{name}_situation.index")
    import json
    with open(f"{name}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def Trace_str(json_path):
    # Load historical trace
    with open(json_path, 'r', encoding='utf-8') as file:
        trace_data = json.load(file)

    trace = ""
    for step in trace_data["iterations"]:
        trace += f"(step {step['step']})\n"
        trace += (
            "[Planning]\n"
            f"plan: {step['plan']}\n"
            "[Reflection]\n"
            f"\'temp_memory\': {step['temp_memory']}\n"
            f"\'decision\': {step['decision']}\n\n"
        )
    trace += "[Answering]\n" + f"integrated_memory:{trace_data['integrated_memory']}"
    return trace_data["question"], trace

# Integrate some information and return historical information and related situations
def info_prompt(json_path, step, module):
    with open(json_path, 'r', encoding='utf-8') as file:
        trace_data = json.load(file)
    info = ""       # Historical related information
    situation = ""  # Historical situation
    for item in trace_data["iterations"]:
        if str(item["step"]) == str(step):
            if module == "Planning":
                # Organize Planning
                if str(step) == "0":
                    current_question = trace_data["question"]
                else:
                    current_question = trace_data["iterations"][int(step)-1]["decision"]["new_request"]
                situation = current_question
                info = "\"question\": " + current_question + "\n" + "\"plan\": " + str(item["plan"])
            elif module == "Reflection":
                # Organize Reflection
                info = "\"temp_memory\": " + str(item["temp_memory"]["content"]) + "\n" + "\"decision\": " + str(item["decision"])
                current_question = trace_data["question"]
                situation = "question:" + current_question + " " + "temp_memory:" + item["temp_memory"]["content"]
            break
    return current_question, info, situation

# Sampling function
def sample_hotpotqa_by_f1(results_file, base_dir, max_samples, seed=RANDOM_SEED):
    """
    Sample examples from the HotpotQA dataset according to f1_scores.
    
    Parameters:
    - results_file: str, JSON file path containing "f1_scores"
    - base_dir: str, original HotpotQA data JSON file path
    - max_samples: int, maximum total number of samples

    Returns:
    - samples: list, sampled examples
    """

    # Step 1: Read f1_scores and calculate distribution
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    f1_scores = results["f1_scores"]
    total = len(f1_scores)
    print(f"Total number of f1_scores: {total}")

    # Category indices
    f1_1_indices = [i for i, f1 in enumerate(f1_scores) if f1 == 1.0]
    f1_0_indices = [i for i, f1 in enumerate(f1_scores) if f1 == 0.0]
    f1_mid_indices = [i for i, f1 in enumerate(f1_scores) if 0.0 < f1 < 1.0]

    # Category ratios
    ratio_1 = len(f1_1_indices) / total
    ratio_0 = len(f1_0_indices) / total
    ratio_mid = len(f1_mid_indices) / total
    print(f"Theoretical sampling ratio: {ratio_1}: {ratio_0}: {ratio_mid}")

    # Step 2: Calculate sample counts according to ratio
    num_1 = math.floor(ratio_1 * max_samples)
    num_0 = math.floor(ratio_0 * max_samples)
    num_mid = math.floor(ratio_mid * max_samples)

    # Adjust total to max_samples
    current_total = num_1 + num_0 + num_mid
    remaining = max_samples - current_total
    counts = {"1": num_1, "0": num_0, "mid": num_mid}
    if remaining > 0:
        max_key = max(counts, key=counts.get)
        counts[max_key] += remaining

    num_1, num_0, num_mid = counts["1"], counts["0"], counts["mid"]
    print(f"num_1: {num_1}; num_0: {num_0}; num_mid: {num_mid}")

    # Step 3: Load original data
    with open(base_dir, 'r', encoding='utf-8') as f:
        items = json.load(f)

    # Step 4: Seeded stratified sampling according to f1
    rng = random.Random(seed)
    samples_1 = rng.sample([items[i] for i in f1_1_indices], min(num_1, len(f1_1_indices)))
    samples_0 = rng.sample([items[i] for i in f1_0_indices], min(num_0, len(f1_0_indices)))
    samples_mid = rng.sample([items[i] for i in f1_mid_indices], min(num_mid, len(f1_mid_indices)))

    samples = samples_1 + samples_0 + samples_mid
    rng.shuffle(samples)

    return samples



# high_quality | low_quality
# Reflection on the Planning module
def Exp_Planning(QUESTION, content, info, flag_type):
    content_type = ""
    if flag_type == 1:
        content_type = "high_quality"
    else:
        content_type = "low_quality"

    # Dynamically set role description (based on Planning quality rather than success/failure)
    if content_type == "low_quality":
        role_desc = "You are an AI [TRACE] Auditor: Extract abstract situation and corrective planning experience from low-quality planning traces."
        planning_task = "summarize a corrective and discriminative planning experience to prevent similar low-quality planning behaviors."
        thinking = "Why is this [TRACE] considered low quality under the [DIAGNOSED REASON]. Where did it perform worst?"
    elif content_type == "high_quality":
        role_desc = "You are an AI [TRACE] Strategist: Distill best-practice planning patterns and reusable experience from high-quality planning traces."
        planning_task = "summarize an effective and discriminative planning experience that leads to high-quality planning behaviors."
        thinking = "Why is this [TRACE] considered high quality under the [DIAGNOSED REASON]. Where did it perform best?"
    
    print(f"content_type: {content_type}")
    # Import prompt templates
    system_prompt = Planning_system_prompt.format(role_desc=role_desc)
    # prompt = Planning_prompt.format(QUESTION=QUESTION, content_type=content_type, info=info, content_reason=content['reason and advice'], thinking=thinking, planning_task=planning_task)
    prompt = Planning_prompt_extra.format(QUESTION=QUESTION, content_type=content_type, info=info, content_reason=content['reason and advice'], thinking=thinking, planning_task=planning_task)  # hotpotqa and narrativeqa long datasets
    SCHEMA = {
            "type": "object",
            "properties": {
                "thinking": {
                    "type": "string"
                },
                "summary": {
                    "type": "string"
                },
                "situation": {
                    "type": "string"
                },
                "experience": {
                    "type": "string"
                }
            },
            "required": ["thinking", "summary", "situation", "experience"],
            "additionalProperties": False
        }

    response = student_model.generate_single(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        schema=SCHEMA
    )
    total_tokens = response["response"]["usage"]["total_tokens"]

    result, flag = response_json(response)
    if flag:
        return result, total_tokens
    else:
        return [], 0

# high_quality | low_quality
# Reflection on the Reflection module
def Exp_Reflection(QUESTION, content, info, flag_type):
    content_type = ""
    if flag_type == 1:
        content_type = "high_quality"
    else:
        content_type = "low_quality"
    # Dynamically set role description (based on TRACE quality rather than success/failure)
    if content_type == "low_quality":
        role_desc = "You are an AI Memory [TRACE] Auditor: Extract abstract abstract situation and corrective from experience low-quality reasoning traces."
        thinking = "Why is this [TRACE] considered low quality under the evaluation? Where did it perform worst?"
        experience = "summarize an abstract and reusable experience to prevent similar low-quality reflection behaviors in future scenarios."

    elif content_type == "high_quality":
        role_desc = "You are an AI Memory [TRACE] Strategist: Distill best-practice patterns and reusable experience from high-quality reasoning traces."
        thinking = "Why is this [TRACE] considered high quality under the evaluation? Where did it perform best?"
        experience = "summarize an abstract and reusable experience to lead high-quality reflection behaviors in future scenarios."
    
    # Use prompt templates
    system_prompt = Reflection_system_prompt.format(role_desc=role_desc)
    prompt = Reflection_prompt.format(QUESTION=QUESTION, content_type=content_type, info=info, content_reason=content['reason and advice'], thinking=thinking, experience=experience)

    # Schema unchanged
    SCHEMA = {
            "type": "object",
            "properties": {
                "thinking": {
                    "type": "string"
                },
                "summary": {
                    "type": "string"
                },
                "situation": {
                    "type": "string"
                },
                "experience": {
                    "type": "string"
                }
            },
            "required": ["thinking", "summary", "situation", "experience"],
            "additionalProperties": False
        }
    response = student_model.generate_single(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        schema=SCHEMA
    )
    total_tokens = response["response"]["usage"]["total_tokens"]
    result, flag = response_json(response)
    return result, total_tokens

def self_reflection(json_path, item):
    # Reflect on historical traces and summarize corresponding experiences
    QUESTION, TRACE = Trace_str(json_path)
    REFERENCE_ANSWER = item["gold_answer"]
    MODEL_ANSWER = item["summary_answer"]

    # Final version uses GPT
    content, total_tokens_GPT = gpt_judger(QUESTION, REFERENCE_ANSWER, MODEL_ANSWER, TRACE, GPT_API_key)
    print(f"Evaluator token usage: {total_tokens_GPT}")
    print(content)
    
    # Self-Evo uses the same backbone as the Evaluator. The reference answer is
    # available only for the offline construction split; after this evaluation
    # and reflection process, the backbone reuses the resulting experience to
    # improve later searches without parameter updates or evaluation answers.
    # content, total_tokens_GPT = judger_vllm(QUESTION, REFERENCE_ANSWER, MODEL_ANSWER, TRACE, student_model)
    # content, total_tokens_GPT = deepseek_judger(QUESTION, REFERENCE_ANSWER, MODEL_ANSWER, TRACE, DEEPSEEK_API_key)
    sum_tokens = total_tokens_GPT
    contents = content["results"]
    print(content["results"])
    print(sum_tokens)
    print(len(contents))
    
    # scores_list = []
    ans_sum = []
    for content in contents:
        total_score = 0
        for key, value in content['rubrics'].items():
            total_score += value
        
        print(f"module: {content['module']}")
        print(f"Current rubrics_score: {total_score}")
        step = content["step"]
        QUESTION, info, condition = info_prompt(json_path, content["step"], content["module"])
        flag_type = -1
        # scores_list.append({content['module']: total_score})
        # Previously maybe considered...(>=9 AND <=6, now 10 and 5)
        if total_score >= 0 and total_score <= 12:
            # Ensure score is within normal range
            if content['module'] == "Planning":
                Planning_scores[total_score] = Planning_scores[total_score] + 1
            elif content['module'] == "Reflection":
                Reflection_scores[total_score] = Reflection_scores[total_score] + 1
        
        if total_score >= High_threshold:    # High-quality module(2,2,2,3) at least one 3, total = 9
            print("Excellent trace found")
            flag_type = 1
        elif total_score <= Low_threshold:  # Low-quality module(1,1,2,2) at least two 1s, total = 6
            print("Low-quality trace found")
            flag_type = 0
        else:
            print("Average trace")
            continue
        
        if content["module"] == "Planning":
            print("============== Planning ==============")
            ans, tokens = Exp_Planning(QUESTION, content, info, flag_type)
            print(ans)
            print("\n")
            if type(ans) == type("A"):
                return [], sum_tokens, sum_tokens - total_tokens_GPT
            ans["condition"] = condition
            ans["module"] = content["module"]

        if content["module"] == "Reflection":
            print("============== Reflection ==============")
            ans, tokens = Exp_Reflection(QUESTION, content, info, flag_type)
            print(ans)
            print("\n")
            if type(ans) == type("A"):
                return [], sum_tokens, sum_tokens - total_tokens_GPT
            ans["condition"] = condition
            ans["module"] = content["module"]
        print(f"{content['module']}_tokens: {tokens}")
        sum_tokens = sum_tokens + tokens
        print(f"ans: {ans}")

        ans_sum.append(ans)
    print(f"One self-reflection completed, sum_tokens: {sum_tokens}")
    self_reflection_tokens = sum_tokens - total_tokens_GPT
    return ans_sum, sum_tokens, self_reflection_tokens

def main():
    # Use the first LOCOMO sample to initialize the experience bank
    
    # BAAI/bge-m3
    DENSE_MODEL_PATH = "your_bge_model"
    # replayer = ReplayEngine(WORKDIR, DENSE_MODEL_PATH)
    TOKENS = 0
    TOKENS_self = 0
    Planning_exp = []
    Reflection_exp = []

    ###########################################################
    # locomo
    ###########################################################
    if datasets == "locomo":
        
        # Statistics of planning and reflection score distributions (0-12 distribution)
        # According to data distribution (question category types and counts)
        all_convs = [26, 30, 41, 42, 43, 44, 47, 48, 49, 50]
        convs = [random.Random(RANDOM_SEED).choice(all_convs)]
        for conv in convs:
            with open(f"{path_locomo}/conv-{conv}/qa_results.json", 'r', encoding='utf-8') as file:
                items = json.load(file)
            for idx, item in enumerate(items, start=1):
                try:
                    WORKDIR = f"{path_locomo}/conv-{conv}"
                    print(f"=========== Processing Q{idx} ===========")
                    json_path = f"{WORKDIR}/research_trace_q{idx}.json"
                    ans_sum, sum_tokens, self_reflection_tokens = self_reflection(json_path, item)
                    TOKENS = TOKENS + sum_tokens
                    TOKENS_self = TOKENS_self + self_reflection_tokens
                    print(f"sum_tokens: {sum_tokens}")
                    
                    # Skip if empty dictionary is returned
                    if not ans_sum:
                        continue
                    for ans in ans_sum:
                        if ans["module"] == "Planning":
                            Planning_exp.append(ans)
                        elif ans["module"] == "Reflection":
                            Reflection_exp.append(ans)

                except Exception as e:
                    print(f"Error in Q{idx}: {e}")
                    continue

        print(f"Total TOKENS count: {TOKENS}")
        print(f"Total TOKENS_self: {TOKENS_self}")
        print("Score probability distribution statistics:")
        print("[Planning_scores]")
        print(Planning_scores)
        print("\n")
        print("[Reflection_scores]")
        print(Reflection_scores)

        # Save results
        # Planning experience
        with open(Planning_exp_path, 'w', encoding='utf-8') as f:
            json.dump(Planning_exp, f,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True)

        # Reflection experience
        with open(Reflection_exp_path, 'w', encoding='utf-8') as f:
            json.dump(Reflection_exp, f,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True)

    ###########################################################
    # narrativeqa
    ###########################################################
    elif datasets == "narrativeqa":
        with open(narrativeqa_base_dir, 'r', encoding='utf-8') as file:
            items = json.load(file)
        samples = items[0:60]   # Use 20% of the 300 sampled questions for experience construction
        
        for sample in samples:
            try:
                # Construct manually
                item = {}
                item["gold_answer"] = str(sample["gold_answers"])
                item["summary_answer"] = sample["pred"]
                json_path = sample["research_trace_file"]
                ans_sum, sum_tokens, self_reflection_tokens = self_reflection(json_path, item)
                
                TOKENS = TOKENS + sum_tokens
                print(f"sum_tokens: {sum_tokens}")
                
                # Skip if empty dictionary is returned
                if not ans_sum:
                    continue
                for ans in ans_sum:
                    if ans["module"] == "Planning":
                        Planning_exp.append(ans)
                    elif ans["module"] == "Reflection":
                        Reflection_exp.append(ans)
            
            except Exception as e:
                    print(f"Error: {e}")
                    continue
        
        print(f"Total TOKENS count: {TOKENS}")
        print("Score probability distribution statistics:")
        print("[Planning_scores]")
        print(Planning_scores)
        print("\n")
        print("[Reflection_scores]")
        print(Reflection_scores)

        # Save results
        # Planning experience
        with open(Planning_exp_path, 'w', encoding='utf-8') as f:
            json.dump(Planning_exp, f,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True)

        # Reflection experience
        with open(Reflection_exp_path, 'w', encoding='utf-8') as f:
            json.dump(Reflection_exp, f,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True)

    ###########################################################
    # hotpotqa
    ###########################################################
    elif datasets == "hotpotqa":

        # Each contains 128 examples, select 25 for exp_learning
        # samples = items[0:25]   # Take the first 25 samples (20%) for experience summarization (or replace with random sampling)
        samples = sample_hotpotqa_by_f1(results_file, base_dir, max_samples, RANDOM_SEED)

        samples_ids = []
        # Rewrite slightly
        for sample in samples:
            try:
                # Construct manually
                item = {}
                
                # Save sample ids
                samples_ids.append(sample["_id"])

                item["gold_answer"] = str(sample["gold_answers"])
                item["summary_answer"] = sample["pred"]
                json_path = sample["research_trace_file"]
                ans_sum, sum_tokens, self_reflection_tokens = self_reflection(json_path, item)
                
                TOKENS = TOKENS + sum_tokens
                print(f"sum_tokens: {sum_tokens}")
                
                # Skip if empty dictionary is returned
                if not ans_sum:
                    continue
                for ans in ans_sum:
                    if ans["module"] == "Planning":
                        Planning_exp.append(ans)
                    elif ans["module"] == "Reflection":
                        Reflection_exp.append(ans)
            
            except Exception as e:
                    print(f"Error: {e}")
                    continue
        
        with open(f'samples_ids_{hotpotqa_data}.json', 'w', encoding='utf-8') as f:
            json.dump(samples_ids, f, ensure_ascii=False, indent=2)
        
        print(f"Total TOKENS count: {TOKENS}")
        print("Score probability distribution statistics:")
        print("[Planning_scores]")
        print(Planning_scores)
        print("\n")
        print("[Reflection_scores]")
        print(Reflection_scores)

        # Save results
        # Planning experience
        with open(Planning_exp_path, 'w', encoding='utf-8') as f:
            json.dump(Planning_exp, f,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True)

        # Reflection experience
        with open(Reflection_exp_path, 'w', encoding='utf-8') as f:
            json.dump(Reflection_exp, f,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True)

    # embedding
    with open(Planning_exp_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    exp_bank(name_Planning, data)
    with open(Reflection_exp_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    exp_bank(name_Reflection, data)

if __name__ == "__main__":
    main()
