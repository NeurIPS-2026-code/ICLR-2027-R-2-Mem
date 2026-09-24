import json
from openai import OpenAI

import re
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

API_key = "your_api_key"

# trace to string
def Trace_str(json_path):
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

def response_json(response):
    raw_text = response['text']

    match = re.search(r'```json\n(.*?)\n```', raw_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = raw_text

    try:
        data = json.loads(json_str)
        print("json ok")
        return data, 1
    except json.JSONDecodeError as e:
        print(f"json error: {e}")
        return raw_text, 0



import json
from openai import OpenAI


def deepseek_judger(QUESTION, REFERENCE_ANSWER, MODEL_ANSWER, TRACE, API_key):
    client = OpenAI(
        api_key=API_key,
        base_url="https://api.deepseek.com"
    )

    system_prompt = """
    You are an expert evaluator for an AI Memory deep search system. Your task is to score EACH step in the TRACE.

    Each step contains two modules:
    - Planning: The goal is to create an **efficient retrieval strategy**. The system should develop a tailored plan for the question type, aiming to use the **fewest possible tools** to retrieve the exact information needed.
    - Reflection: The goal is to make **accurate sufficiency judgments**. The system must evaluate whether the currently gathered information already provides a satisfactory answer to the original question, and stop retrieval when it does to avoid unnecessary work.
    
    You must score BOTH modules strictly based on the ***Rubrics_List***.
    """

    Rubrics_List = """
    ***Rubrics_List***
    [Planning Quality] (0-3 for each) 
    - Info Needs Coverage: Evaluates whether the planned info_needs comprehensively cover all key information required to answer the question, and whether any critical aspects are missing.
    - Info Needs Non-Redundancy: Evaluates whether the info_needs contain semantic overlap or could be merged, and whether each info_need contributes a distinct and necessary information gain.
    - Tool–Info Alignment: Evaluates whether the selected retrieval tools (keyword, vector, page_index) and their generated fields are appropriate for the nature and focus of the corresponding info_needs.
    - Planning Efficiency: Evaluates whether the planning stage avoids generating too many unnecessary or irrelevant info_needs, tools, or retrieval fields that lead to unnecessary tool invocations, and whether there is excessive planning that yields low information gain relative to cost.

    [Reflection Quality] (0-3 for each)
    - Sufficiency Judgment Accuracy: Whether the system correctly judged if the current information was sufficient.
    - Minimal Sufficiency Recognition: Whether the system correctly recognized that a coarse-grained but valid answer already satisfied the QUESTION, and avoided unnecessary continued retrieval.
    - Follow-up Query Quality: Whether new_request precisely targets missing information and leads to new information, rather than repeating known content.
    - Answer Completeness Awareness: Whether the system correctly identifies if the QUESTION requires a **set, list, or multiple aspects** rather than a single fact, and avoids stopping prematurely after retrieving only partial information.
    """

    TRACE_explain = """
    ***Trace_explain***
    "step": A counter for the current iteration, used for logging and traceability only.(at most 0-2)

    [Planning]
    "plan": planning retrieval. Analyze the current question and determine what information is required and how it should be retrieved.
        - "info_needs": Concrete sub-questions or missing facts that must be resolved in order to fully answer the current QUESTION.
        - "tools": The retrieval strategy types selected for this plan, chosen from ["keyword", "vector", "page_index"].
        - "keyword_collection": Short, high-signal keywords or phrases for exact-match retrieval of specific entities, names, or attributes.
        - "vector_queries": Natural-language queries used for semantic retrieval of conceptually related information.
        - "page_index": Known page indices to be re-read in full

    [Reflection]
    "temp_memory": the current integrated factual summary about that QUESTION. it is intended to contain all useful known information so far
    "decision": reflecting on completeness and generating a follow-up request if needed
        - enough: Whether current content is sufficient to answer the QUESTION.
        - new_request: Follow-up retrieval request if enough is false.

    --------------------------------------------------

    ***Module Input Definition***
    At each step, define current_query:

    - If step = 0:
    current_query = QUESTION
    - If step > 0:
    current_query = the new_request from the previous step

    The current_query evolves across steps.

    --------------------------------------------------

    ***Module Logic***(IMPORTANT)
    - Planning:
    Takes current_query as input and generates a retrieval plan.
    - Reflection:
    Takes current_query and current temp_memory as input, and decides:
        - If sufficient → enough = True
        - If not sufficient → enough = False and generate a detailed new_request
    """

    user_prompt = f"""
    QUESTION:
    {QUESTION}

    REFERENCE_ANSWER:
    {REFERENCE_ANSWER}

    MODEL_ANSWER:
    {MODEL_ANSWER}

    TRACE:
    {TRACE}

    {TRACE_explain}

    {Rubrics_List}

    --------------------------------------------------
    
    YOUR TASK:
    1. Evaluate EACH step in TRACE based on rubrics list. For EACH step: Score Planning and Reflection. Each rubric dimension must be scored from 0 to 3.
    2. provide a short reason explaining the score(1–2 concise sentences) and give some advices(abstract the experience based on the ***Module Logic***)
        - If the score is HIGH → Provide guidance on how to maintain this good behavior
        - If the score is LOW → Provide guidance on how to improve and avoid this issue in future steps

    --------------------------------------------------

    JSON Output format:
    {{
    "results": [
        {{
        "step": 0,
        "module": "Planning",
        "rubrics": {{
            "Info Needs Coverage": 0,
            "Info Needs Non-Redundancy": 0,
            "Tool–Info Alignment": 0,
            "Planning Efficiency": 0
        }},
        "reason and advice": ""
        }},
        {{
        "step": 0,
        "module": "Reflection",
        "rubrics": {{
            "Sufficiency Judgment Accuracy": 0,
            "Minimal Sufficiency Recognition": 0,
            "Follow-up Query Quality": 0,
            "Answer Completeness Awareness": 0
        }},
        "reason and advice": ""
        }}...
    ]
    }}

    STRICT:
    - Output ONLY valid JSON
    - No extra text
    - No trailing commas
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2048,
            temperature=0.25,
            stream=False,
            response_format={'type': 'json_object'}
        )

        content = json.loads(response.choices[0].message.content)
        total_tokens = response.usage.total_tokens

        return content, total_tokens

    except Exception as e:
        return {
            "results": [],
            "error": str(e)
        }, 0

# use gpt-4o
def gpt_judger(QUESTION, REFERENCE_ANSWER, MODEL_ANSWER, TRACE, API_key):
    client = OpenAI(
        base_url="http://chatapi.littlewheat.com/v1",
        api_key=API_key
    )
    system_prompt = """
    You are an expert evaluator for an AI Memory deep search system. Your task is to score EACH step in the TRACE.

    Each step contains two modules:
    - Planning: The goal is to create an **efficient retrieval strategy**. The system should develop a tailored plan for the question type, aiming to use the **fewest possible tools** to retrieve the exact information needed.
    - Reflection: The goal is to make **accurate sufficiency judgments**. The system must evaluate whether the currently gathered information already provides a satisfactory answer to the original question, and stop retrieval when it does to avoid unnecessary work.
    
    You must score BOTH modules strictly based on the ***Rubrics_List***.
    """

    Rubrics_List = """
    ***Rubrics_List***
    [Planning Quality] (0-3 for each) 
    - Info Needs Coverage: Evaluates whether the planned info_needs comprehensively cover all key information required to answer the question, and whether any critical aspects are missing.
    - Info Needs Non-Redundancy: Evaluates whether the info_needs contain semantic overlap or could be merged, and whether each info_need contributes a distinct and necessary information gain.
    - Tool–Info Alignment: Evaluates whether the selected retrieval tools (keyword, vector, page_index) and their generated fields are appropriate for the nature and focus of the corresponding info_needs.
    - Planning Efficiency: Evaluates whether the planning stage avoids generating too many unnecessary or irrelevant info_needs, tools, or retrieval fields that lead to unnecessary tool invocations, and whether there is excessive planning that yields low information gain relative to cost.

    [Reflection Quality] (0-3 for each)
    - Sufficiency Judgment Accuracy: Whether the system correctly judged if the current information was sufficient.
    - Minimal Sufficiency Recognition: Whether the system correctly recognized that a coarse-grained but valid answer already satisfied the QUESTION, and avoided unnecessary continued retrieval.
    - Follow-up Query Quality: Whether new_request precisely targets missing information and leads to new information, rather than repeating known content.
    - Answer Completeness Awareness: Whether the system correctly identifies if the QUESTION requires a **set, list, or multiple aspects** rather than a single fact, and avoids stopping prematurely after retrieving only partial information.
    """

    TRACE_explain = """
    ***Trace_explain***
    "step": A counter for the current iteration, used for logging and traceability only.(at most 0-2)

    [Planning]
    "plan": planning retrieval. Analyze the current question and determine what information is required and how it should be retrieved.
        - "info_needs": Concrete sub-questions or missing facts that must be resolved in order to fully answer the current QUESTION.
        - "tools": The retrieval strategy types selected for this plan, chosen from ["keyword", "vector", "page_index"].
        - "keyword_collection": Short, high-signal keywords or phrases for exact-match retrieval of specific entities, names, or attributes.
        - "vector_queries": Natural-language queries used for semantic retrieval of conceptually related information.
        - "page_index": Known page indices to be re-read in full

    [Reflection]
    "temp_memory": the current integrated factual summary about that QUESTION. it is intended to contain all useful known information so far
    "decision": reflecting on completeness and generating a follow-up request if needed
        - enough: Whether current content is sufficient to answer the QUESTION.
        - new_request: Follow-up retrieval request if enough is false.

    --------------------------------------------------

    ***Module Input Definition***
    At each step, define current_query:

    - If step = 0:
    current_query = QUESTION
    - If step > 0:
    current_query = the new_request from the previous step

    The current_query evolves across steps.

    --------------------------------------------------

    ***Module Logic***(IMPORTANT)
    - Planning:
    Takes current_query as input and generates a retrieval plan.
    - Reflection:
    Takes current_query and current temp_memory as input, and decides:
        - If sufficient → enough = True
        - If not sufficient → enough = False and generate a detailed new_request
    """

    user_prompt = f"""
    QUESTION:
    {QUESTION}

    REFERENCE_ANSWER:
    {REFERENCE_ANSWER}

    MODEL_ANSWER:
    {MODEL_ANSWER}

    TRACE:
    {TRACE}

    {TRACE_explain}

    {Rubrics_List}

    --------------------------------------------------
    
    YOUR TASK:
    1. Evaluate EACH step in TRACE based on rubrics list. For EACH step: Score Planning and Reflection. Each rubric dimension must be scored from 0 to 3.
    2. provide a short reason explaining the score(1–2 concise sentences) and give some advices(abstract the experience based on the ***Module Logic***)
        - If the score is HIGH → Provide guidance on how to maintain this good behavior
        - If the score is LOW → Provide guidance on how to improve and avoid this issue in future steps

    --------------------------------------------------

    JSON Output format:
    {{
    "results": [
        {{
        "step": 0,
        "module": "Planning",
        "rubrics": {{
            "Info Needs Coverage": 0,
            "Info Needs Non-Redundancy": 0,
            "Tool–Info Alignment": 0,
            "Planning Efficiency": 0
        }},
        "reason and advice": ""
        }},
        {{
        "step": 0,
        "module": "Reflection",
        "rubrics": {{
            "Sufficiency Judgment Accuracy": 0,
            "Minimal Sufficiency Recognition": 0,
            "Follow-up Query Quality": 0,
            "Answer Completeness Awareness": 0
        }},
        "reason and advice": ""
        }}...
    ]
    }}

    STRICT:
    - Output ONLY valid JSON
    - No extra text
    - No trailing commas
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )

        content = json.loads(response.choices[0].message.content)
        total_tokens = response.usage.total_tokens

        return content, total_tokens

    except Exception as e:
        return {
            "results": [],
            "error": str(e)
        }, 0


# vllm
# Self-Evo calls this evaluator with the same backbone used by the search agent.
# REFERENCE_ANSWER is restricted to the offline experience-construction split;
# evaluation answers are not used when the fixed experience bank is applied.
def judger_vllm(QUESTION, REFERENCE_ANSWER, MODEL_ANSWER, TRACE, model):
    system_prompt = """
    You are an expert evaluator for an AI Memory deep search system. Your task is to score EACH step in the TRACE.

    Each step contains two modules:
    - Planning: The goal is to create an **efficient retrieval strategy**. The system should develop a tailored plan for the question type, aiming to use the **fewest possible tools** to retrieve the exact information needed.
    - Reflection: The goal is to make **accurate sufficiency judgments**. The system must evaluate whether the currently gathered information already provides a satisfactory answer to the original question, and stop retrieval when it does to avoid unnecessary work.
    
    You must score BOTH modules strictly based on the ***Rubrics_List***.
    """

    Rubrics_List = """
    ***Rubrics_List***
    [Planning Quality] (0-3 for each) 
    - Info Needs Coverage: Evaluates whether the planned info_needs comprehensively cover all key information required to answer the question, and whether any critical aspects are missing.
    - Info Needs Non-Redundancy: Evaluates whether the info_needs contain semantic overlap or could be merged, and whether each info_need contributes a distinct and necessary information gain.
    - Tool–Info Alignment: Evaluates whether the selected retrieval tools (keyword, vector, page_index) and their generated fields are appropriate for the nature and focus of the corresponding info_needs.
    - Planning Efficiency: Evaluates whether the planning stage avoids generating too many unnecessary or irrelevant info_needs, tools, or retrieval fields that lead to unnecessary tool invocations, and whether there is excessive planning that yields low information gain relative to cost.

    [Reflection Quality] (0-3 for each)
    - Sufficiency Judgment Accuracy: Whether the system correctly judged if the current information was sufficient.
    - Minimal Sufficiency Recognition: Whether the system correctly recognized that a coarse-grained but valid answer already satisfied the QUESTION, and avoided unnecessary continued retrieval.
    - Follow-up Query Quality: Whether new_request precisely targets missing information and leads to new information, rather than repeating known content.
    - Answer Completeness Awareness: Whether the system correctly identifies if the QUESTION requires a **set, list, or multiple aspects** rather than a single fact, and avoids stopping prematurely after retrieving only partial information.
    """

    TRACE_explain = """
    ***Trace_explain***
    "step": A counter for the current iteration, used for logging and traceability only.(at most 0-2)

    [Planning]
    "plan": planning retrieval. Analyze the current question and determine what information is required and how it should be retrieved.
        - "info_needs": Concrete sub-questions or missing facts that must be resolved in order to fully answer the current QUESTION.
        - "tools": The retrieval strategy types selected for this plan, chosen from ["keyword", "vector", "page_index"].
        - "keyword_collection": Short, high-signal keywords or phrases for exact-match retrieval of specific entities, names, or attributes.
        - "vector_queries": Natural-language queries used for semantic retrieval of conceptually related information.
        - "page_index": Known page indices to be re-read in full

    [Reflection]
    "temp_memory": the current integrated factual summary about that QUESTION. it is intended to contain all useful known information so far
    "decision": reflecting on completeness and generating a follow-up request if needed
        - enough: Whether current content is sufficient to answer the QUESTION.
        - new_request: Follow-up retrieval request if enough is false.

    --------------------------------------------------

    ***Module Input Definition***
    At each step, define current_query:

    - If step = 0:
    current_query = QUESTION
    - If step > 0:
    current_query = the new_request from the previous step

    The current_query evolves across steps.

    --------------------------------------------------

    ***Module Logic***(IMPORTANT)
    - Planning:
    Takes current_query as input and generates a retrieval plan.
    - Reflection:
    Takes current_query and current temp_memory as input, and decides:
        - If sufficient → enough = True
        - If not sufficient → enough = False and generate a detailed new_request
    """

    user_prompt = f"""
    QUESTION:
    {QUESTION}

    REFERENCE_ANSWER:
    {REFERENCE_ANSWER}

    MODEL_ANSWER:
    {MODEL_ANSWER}

    TRACE:
    {TRACE}

    {TRACE_explain}

    {Rubrics_List}

    --------------------------------------------------
    
    YOUR TASK:
    1. Evaluate EACH step in TRACE based on rubrics list. For EACH step: Score Planning and Reflection. Each rubric dimension must be scored from 0 to 3.
    2. provide a short reason explaining the score(1–2 concise sentences) and give some advices(abstract the experience based on the ***Module Logic***)
        - If the score is HIGH → Provide guidance on how to maintain this good behavior
        - If the score is LOW → Provide guidance on how to improve and avoid this issue in future steps

    --------------------------------------------------

    JSON Output format:
    {{
    "results": [
        {{
        "step": 0,
        "module": "Planning",
        "rubrics": {{
            "Info Needs Coverage": 0,
            "Info Needs Non-Redundancy": 0,
            "Tool–Info Alignment": 0,
            "Planning Efficiency": 0
        }},
        "reason and advice": ""
        }},
        {{
        "step": 0,
        "module": "Reflection",
        "rubrics": {{
            "Sufficiency Judgment Accuracy": 0,
            "Minimal Sufficiency Recognition": 0,
            "Follow-up Query Quality": 0,
            "Answer Completeness Awareness": 0
        }},
        "reason and advice": ""
        }}...
    ]
    }}

    STRICT:
    - Output ONLY valid JSON
    - No extra text
    - No trailing commas
    """

    SCHEMA = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "oneOf": [
                        # ✅ Planning
                        {
                            "properties": {
                                "step": {"type": "integer"},
                                "module": {"const": "Planning"},
                                "rubrics": {
                                    "type": "object",
                                    "properties": {
                                        "Info Needs Coverage": {"type": "integer"},
                                        "Info Needs Non-Redundancy": {"type": "integer"},
                                        "Tool–Info Alignment": {"type": "integer"},
                                        "Planning Efficiency": {"type": "integer"}
                                    },
                                    "required": [
                                        "Info Needs Coverage",
                                        "Info Needs Non-Redundancy",
                                        "Tool–Info Alignment",
                                        "Planning Efficiency"
                                    ],
                                    "additionalProperties": False
                                },
                                "reason and advice": {"type": "string"}
                            },
                            "required": ["step", "module", "rubrics", "reason and advice"]
                        },

                        # ✅ Reflection
                        {
                            "properties": {
                                "step": {"type": "integer"},
                                "module": {"const": "Reflection"},
                                "rubrics": {
                                    "type": "object",
                                    "properties": {
                                        "Sufficiency Judgment Accuracy": {"type": "integer"},
                                        "Minimal Sufficiency Recognition": {"type": "integer"},
                                        "Follow-up Query Quality": {"type": "integer"},
                                        "Answer Completeness Awareness": {"type": "integer"}
                                    },
                                    "required": [
                                        "Sufficiency Judgment Accuracy",
                                        "Minimal Sufficiency Recognition",
                                        "Follow-up Query Quality",
                                        "Answer Completeness Awareness"
                                    ],
                                    "additionalProperties": False
                                },
                                "reason and advice": {"type": "string"}
                            },
                            "required": ["step", "module", "rubrics", "reason and advice"]
                        }
                    ]
                }
            }
        },
        "required": ["results"]
    }

    try:
        response = model.generate_single(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            schema=SCHEMA
        )
        
        tokens = response["response"]["usage"]["total_tokens"]
        data = response.get("json") or json.loads(response["text"])

        return data, tokens

    except Exception as e:
        return {
            "results": [],
            "error": str(e)
        }
