# exp_replay.py
import os
import json
import sys
import random

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))


from exp.eval.research_agent_exp_main import ResearchAgent_exp
from typing import Any, Dict, List, Optional, Tuple
from gam import (
    InMemoryMemoryStore,
    InMemoryPageStore,
    IndexRetriever,
    BM25Retriever,
    DenseRetriever,
    IndexRetrieverConfig,
    BM25RetrieverConfig,
    DenseRetrieverConfig,
    VLLMGenerator,
    VLLMGeneratorConfig,
)

model_size = "7B"

# One seed/source conversation per manual run. Replace with 42 or 44 for the
# other two runs; keep this value aligned with exp/self_reflection.py.
RANDOM_SEED = 43
# RANDOM_SEED = 42
# RANDOM_SEED = 44


import os
results_path = f"results_locomo_exp"
os.makedirs(results_path, exist_ok=True)

all_convs = [26, 30, 41, 42, 43, 44, 47, 48, 49, 50]
source_conv = random.Random(RANDOM_SEED).choice(all_convs)
convs = [conv for conv in all_convs if conv != source_conv]

for conv in convs:
    conv_folder = os.path.join(results_path, str(conv))
    os.makedirs(conv_folder, exist_ok=True)

DENSE_MODEL_PATH = "your_bge-m3_model"

# use GAM's memory pages can reduce the same (pages memories construction)
GAM_path = f"your_results_locomo"

base_model = "qwen2.5-7B-Instruct"
base_url = "http://localhost:8001/v1"

def Trace_str(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        trace_data = json.load(file)

    trace = ""
    for step in trace_data["iterations"]:
        trace += f"(step {step['step']})\n"
        trace += (
            "[planning]\n"
            f"plan: {step['plan']}\n"
            "[Reflection]\n"
            f"\'temp_memory\': {step['temp_memory']}\n"
            f"\'decision\': {step['decision']}\n\n"
        )
    trace += "[Answering]\n" + f"integrated_memory:{trace_data['integrated_memory']}"
    return trace_data["question"], trace

def make_summary_prompt_category2(summary: str, question: str) -> str:
    return f"""\
Based on the summary below, write an answer in the form of **a short phrase** for the following question, not a sentence. Answer with exact words from the context whenever possible.
For questions that require answering a date or time, strictly follow the format \"15 July 2023\" and provide a specific date whenever possible. For example, if you need to answer \"last year,\" give the specific year of last year rather than just saying \"last year.\" Only provide one year, date, or time, without any extra responses.
If the question is about the duration, answer in the form of several years, months, or days.

QUESTION:
{question}

SUMMARY:
{summary}

Short answer:
"""

def make_summary_prompt_category3(summary: str, question: str) -> str:
    return f"""\
Based on the summary below, write an answer in the form of **a short phrase** for the following question, not a sentence.
The question may need you to analyze and infer the answer from the summary.
    
QUESTION:
{question}

SUMMARY:
{summary}

Short answer:
"""

def make_summary_prompt(summary: str, question: str) -> str:
    return f"""\
Based on the summary below, write an answer in the form of **a short phrase** for the following question, not a sentence. Answer with exact words from the context whenever possible.
The question may need you to analyze and infer the answer from the summary.
    
QUESTION:
{question}

SUMMARY:
{summary}

Short answer:
"""

def answer_with_summary(category: Optional[int], summary: str, question: str, generator) -> str:
    if category == 2:
        prompt = make_summary_prompt_category2(summary, question)
    elif category == 3:
        prompt = make_summary_prompt_category3(summary, question)
    else:
        prompt = make_summary_prompt(summary, question)
    raw = generator.generate_single(prompt=prompt)
    return raw.get("text", "").strip()


class ReplayEngine:
    def __init__(self, workdir, dense_model_path):
        print("🚀 [Init] Loading heavy resources once...")
        self.workdir = workdir
        
        self.memory_store = InMemoryMemoryStore(dir_path=workdir)
        self.page_store = InMemoryPageStore(dir_path=workdir)
        
        self.retrievers = {
            "page_index": IndexRetriever(IndexRetrieverConfig(
                index_dir=os.path.join(workdir, "page_index")).__dict__),
            "keyword": BM25Retriever(BM25RetrieverConfig(
                index_dir=os.path.join(workdir, "bm25_index"), threads=1).__dict__),
            "vector": DenseRetriever(DenseRetrieverConfig(
                index_dir=os.path.join(workdir, "dense_index"), 
                model_name=dense_model_path).__dict__)
        }

        print("🔨 Building retrievers once...")
        for name, r in self.retrievers.items():
            try:
                r.build(self.page_store)
                print(f"✅ Built {name} retriever")
            except Exception as e:
                print(f"❌ Failed to build {name}: {e}")

        self.research_generator = VLLMGenerator(VLLMGeneratorConfig(
            model_name=base_model,
            api_key="empty",
            base_url=base_url,
            temperature=0.3,
            max_tokens=2048,
            use_schema=True
        ).__dict__)

    def run_replay_exp(self, question, conv, save_id):
        
        research_agent = ResearchAgent_exp(
            page_store=self.page_store,
            memory_store=self.memory_store,
            retrievers=self.retrievers,
            generator=self.research_generator,
            max_iters=3
        )
        self.current_agent = research_agent

        with open(f"{GAM_path}/conv-{conv}/qa_results.json", 'r', encoding='utf-8') as file:
            items = json.load(file)
        gold_answer = items[save_id-1]["gold_answer"]
        category = items[save_id-1]["category"]

        result, experience_log = research_agent.research(question, category)
        raw = result.raw_memory

        research_trace = {
            "question": question,
            "raw_memory": raw,
            "integrated_memory": result.integrated_memory,
            "iterations": raw.get("iterations", []),
            "search_plans": raw.get("search_plans", []),
            "reflections": raw.get("reflections", [])
        }

        trace_file = f"{results_path}/{conv}/replayed_research_trace_{save_id}.json"
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(research_trace, f, ensure_ascii=False, indent=2)
        
        summary_answer = answer_with_summary(category, result.integrated_memory, question, self.research_generator)
        print("===================")
        print(summary_answer)

        qa_result = {
            "question": question,
            "gold_answer": gold_answer,
            "category": category,
            "research_summary": result.integrated_memory,
            "summary_answer": summary_answer,
            "iterations": len(result.raw_memory.get("iterations", [])),
            "research_trace_file": trace_file
        }
        print(qa_result)
        
        total_tokens = research_agent.total_tokens
        return qa_result, experience_log, total_tokens


def main():
    
    for conv in convs:
        TOKENS = 0  # each conv tokens
        conv = str(conv)
        print(f"========= conv:{conv} =========")
        qa_results = []
        WORKDIR = f"{GAM_path}/conv-{conv}"
        replayer = ReplayEngine(WORKDIR, DENSE_MODEL_PATH)

        with open(f"{GAM_path}/conv-{conv}/qa_results.json", 'r', encoding='utf-8') as file:
            items = json.load(file)

        nums = 1
        for item in items:
            print(f"========= conv:{conv}({nums}) =========")
            replayer.current_agent = None
            try:
                question = item.get("question", "")
                # run one question + exp
                qa_result, experience, total_tokens = replayer.run_replay_exp(question, conv, nums)
                print("total_tokens:", total_tokens)
                TOKENS = TOKENS + total_tokens
                print("sum TOKENS:", TOKENS)
                qa_results.append(qa_result)
                nums = nums + 1
            except Exception as e:
                print(f"❌ Error at conv:{conv}({nums}) -> {e}")
                import traceback
                traceback.print_exc()
                partial_tokens = getattr(replayer.current_agent, "total_tokens", 0)
                TOKENS = TOKENS + partial_tokens
                qa_results.append({
                    "question": item.get("question", ""),
                    "gold_answer": item.get("gold_answer", ""),
                    "category": item.get("category"),
                    "research_summary": "",
                    "summary_answer": "",
                    "iterations": 0,
                    "error": str(e)
                })
                nums += 1
        
        with open(f"{results_path}/{conv}/qa_results.json", "w", encoding="utf-8") as f:
            json.dump(qa_results, f, ensure_ascii=False, indent=2)

        dict_tokens = {conv: TOKENS}
        with open(f"{results_path}/{conv}/TOKENS.json", "w", encoding="utf-8") as f:
            json.dump(dict_tokens, f, ensure_ascii=False, indent=2)

        print("TOKENS:", TOKENS)
        
if __name__ == "__main__":
    main()
        
