from __future__ import annotations

import os
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

import os
os.environ["TOKEN_LAYERS_PARALLELISM"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Any, Dict, List, Optional, Tuple
import json
from exp.prompts.research_prompts_exp import Planning_PROMPT, Planning_PROMPT_exp, Integrate_PROMPT, Integrate_PROMPT_exp, InfoCheck_PROMPT, InfoCheck_PROMPT_exp, GenerateRequests_PROMPT, GenerateRequests_PROMPT_exp, Planning_PROMPT_exp_extra
from gam.schemas import (
    MemoryState, SearchPlan, Hit, Result, 
    ReflectionDecision, ResearchOutput, MemoryStore, PageStore, Retriever, 
    ToolRegistry, InMemoryMemoryStore,
    PLANNING_SCHEMA, INTEGRATE_SCHEMA, INFO_CHECK_SCHEMA, GENERATE_REQUESTS_SCHEMA
)
from gam.generator import AbsGenerator

from .experience_encoder import get_encoder
import faiss
import json
import numpy as np


# model_size settings
model_size = "7B"

# Experience bank path
exp_bank_path = "..."


def generate_situation_planning(vllm_generator, query):
    system_prompt = """
    You are an expert in abstracting situation.
    """

    user_prompt = f"""
    [question]: {query}

    ***YOUR TASK***
    abstract situation of the [question](Do NOT include concrete entities):
    Summarize the abstract planning situation, What is this question trying to ask?
    - Multi-entity query
    - Time-related query
    - list/set
    - single_fact
    - multi_hop
    ...
    
    Generate the standardized situation(one concise sentence).

    Output ONLY valid JSON:
    {{
        "situation": "<abstract situation>"
    }}

    """
    
    prompt = f"System Prompt: {system_prompt}\n\nUser Instructions: {user_prompt}"
    
    SCHEMA = {
        "type": "object",
        "properties": {
            "situation": {
                "type": "string"
            }
        },
        "required": ["situation"],
        "additionalProperties": False
    }

    response = vllm_generator.generate_single(prompt=prompt, schema=SCHEMA)
    tokens = response["response"]["usage"]["total_tokens"]
    data = response.get("json") or json.loads(response["text"])
    
    if data:
        return data["situation"], tokens
    else:
        return None, 0


def generate_situation_reflection(vllm_generator, query, temp_memory):
    system_prompt = """
    You are an expert in abstracting situation.
    """

    user_prompt = f"""
    [question]: {query}
    [temp_memory]: {temp_memory}

    ***YOUR TASK***
    abstract situation (Do NOT include concrete entities):
        Summarize the abstract situation(question and temp memory), considering:
        What is this question trying to ask?
        What does the temp_memory already contain?
        Whether the current information in temp_memory is already sufficient to solve the question?

    Generate the standardized situation.

    Output ONLY valid JSON:
    {{
        "situation": "<abstract situation>"
    }}

    """

    prompt = f"System Prompt: {system_prompt}\n\nUser Instructions: {user_prompt}"

    SCHEMA = {
        "type": "object",
        "properties": {
            "situation": {
                "type": "string"
            }
        },
        "required": ["situation"],
        "additionalProperties": False
    }

    response = vllm_generator.generate_single(prompt=prompt, schema=SCHEMA)
    tokens = response["response"]["usage"]["total_tokens"]
    data = response.get("json") or json.loads(response["text"])

    if data:
        return data["situation"], tokens
    else:
        return None, 0


def retrieve_experience(
    query,
    name,
    return_k=3,
    recall_k=10,
):
    model = get_encoder()

    index = faiss.read_index(f"{exp_bank_path}/{name}_{model_size}_situation.index")
    with open(f"{exp_bank_path}/{name}_{model_size}_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    query_emb = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    D, I = index.search(query_emb, k=recall_k)

    experiences = []

    for similarity, idx in zip(D[0], I[0]):
        if idx == -1:
            continue

        meta = metadata[idx]

        experiences.append({
            "id": int(idx),
            "similarity": float(similarity),

            "situation": meta["situation"],
            "summary": meta["summary"],
            "experience": meta["experience"],
        })

    experiences = sorted(
        experiences,
        key=lambda x: x["similarity"],
        reverse=True
    )

    return experiences[:return_k]


class ResearchAgent_exp:
    """
    Public API:
      - research(request) -> ResearchOutput
    Internal steps:
      - _planning(request, memory_state) -> SearchPlan
      - _search(plan) -> SearchResults  (calls keyword/vector/page_id + tools)
      - _integrate(search_results, temp_memory) -> TempMemory
      - _reflection(request, memory_state, temp_memory) -> ReflectionDecision

    Note: Uses MemoryStore to dynamically load current memory state.
    This allows ResearchAgent to access the latest memory updates from MemoryAgent.
    """

    def __init__(
        self,
        page_store: PageStore,
        memory_store: MemoryStore | None = None,
        tool_registry: Optional[ToolRegistry] = None,
        retrievers: Optional[Dict[str, Retriever]] = None,
        generator: AbsGenerator | None = None,
        max_iters: int = 3,
        dir_path: Optional[str] = None,
        system_prompts: Optional[Dict[str, str]] = None,
    ) -> None:
        if generator is None:
            raise ValueError("Generator instance is required for ResearchAgent")
        self.page_store = page_store
        self.memory_store = memory_store or InMemoryMemoryStore(dir_path=dir_path)
        self.tools = tool_registry
        self.retrievers = retrievers or {}
        self.generator = generator
        self.max_iters = max_iters

        # token
        self.total_tokens: int = 0
        
        default_system_prompts = {
            "planning": "",
            "integration": "",
            "reflection": ""
        }
        if system_prompts is None:
            self.system_prompts = default_system_prompts
        else:
            self.system_prompts = {**default_system_prompts, **system_prompts}
        

    # ---- Public ----
    def research(self, request: str, category: int) -> ResearchOutput:
        self._update_retrievers()
        
        temp = Result()
        iterations: List[Dict[str, Any]] = []
        next_request = request
        experience_log = []

        for step in range(self.max_iters):
            # Load current memory state dynamically
            memory_state = self.memory_store.load()
            
            # abstract situation
            planning_situation, tokens = generate_situation_planning(self.generator, next_request)
            print(f"planning_situation: {planning_situation}")
            self.total_tokens += tokens
            plan, experience_1 = self._planning(next_request, memory_state, planning_situation, 1)      # planning(planning tokens in the function)

            temp_before = temp.content
            temp = self._search(plan, temp, request, category, model_size)                              # searching


            reflection_situation, tokens = generate_situation_reflection(self.generator, request, temp)
            print(f"reflection_situation: {reflection_situation}")
            self.total_tokens += tokens
            decision, experience_2 = self._reflection(request, temp, reflection_situation, 1)           # reflection(reflection tokens in the function)


            print(f"decision: {decision}, type:{type(decision)}")
            
            iterations.append({
                "step": step,
                "plan": plan.__dict__,
                "temp_memory": temp.__dict__,
                "decision": decision.__dict__,
            })
            
            # experience log
            experience_log.append({"step":step, "Planning": experience_1, "Reflection": experience_2})

            if decision.enough:
                break

            if not decision.new_request:
                next_request = request
            else:
                next_request = decision.new_request

        raw = {
            "iterations": iterations,
            "temp_memory": temp.__dict__,
        }
        return ResearchOutput(integrated_memory=temp.content, raw_memory=raw), experience_log

    def _update_retrievers(self):
        current_page_count = len(self.page_store.load())
        
        if hasattr(self, '_last_page_count') and current_page_count != self._last_page_count:
            print(f"page nums change ({self._last_page_count} -> {current_page_count})，upgrate...")
            for name, retriever in self.retrievers.items():
                try:
                    retriever.update(self.page_store)
                    print(f"✅ Updated {name} retriever index")
                except Exception as e:
                    print(f"❌ Failed to update {name} retriever: {e}")
        
        self._last_page_count = current_page_count

    # ---- Internal ----
    def _planning(
        self, 
        request: str, 
        memory_state: MemoryState,
        planning_situation: str,
        exp_flag: int,
        planning_prompt: Optional[str] = None
    ) -> SearchPlan:
        """
        Produce a SearchPlan:
          - what specific info is needed
          - which tools are useful + inputs
          - keyword/vector/page_id payloads
        """

        if not memory_state.abstracts:
            memory_context = "No memory currently."
        else:
            memory_context_lines = []
            for i, abstract in enumerate(memory_state.abstracts):
                memory_context_lines.append(f"Page {i}: {abstract}")
            memory_context = "\n".join(memory_context_lines)
        
        system_prompt = self.system_prompts.get("planning")


        # retrieve with planning_situation
        # experiences = retrieve_experience(planning_situation, name="Planning")
        # experience = "\n".join(
        #     f"({i+1})experience:{e['experience']}\n\n"
        #     for i, e in enumerate(experiences)
        # )
        experiences = []

        if exp_flag:
            # situation + condition
            planning_situation = "(" + planning_situation + ")" + request
            print(f"planning_situation: {planning_situation}")
            experiences = retrieve_experience(planning_situation, name="Planning")
            experience = "\n".join(
                f"({i+1})experience:{e['experience']}\n\n"
                for i, e in enumerate(experiences)
            )

            template_prompt = Planning_PROMPT_exp.format(request=request, memory=memory_context, experience=experience)   # prompt 1
            # template_prompt = Planning_PROMPT_exp_extra.format(request=request, memory=memory_context, experience=experience)
        else:
            template_prompt = Planning_PROMPT.format(request=request, memory=memory_context)


        if system_prompt:
            prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_prompt}"
        else:
            prompt = template_prompt

        try:
            # print("============================== planning prompt ==============================")
            # print(prompt)
            # print("\n")

            response = self.generator.generate_single(prompt=prompt, schema=PLANNING_SCHEMA)
            tokens = response["response"]["usage"]["total_tokens"]
            print("Planning tokens:", tokens)
            self.total_tokens += tokens
            print("sum_tokens:", self.total_tokens)
            
            print("============================== planning response ==============================")
            print(response)
            print("\n")
            data = response.get("json") or json.loads(response["text"])
            return SearchPlan(
                info_needs=data.get("info_needs", []),
                tools=data.get("tools", []),
                # keyword_collection=[request],
                keyword_collection=data.get("keyword_collection", []),
                vector_queries=data.get("vector_queries", []),
                page_index=data.get("page_index", [])
            ), experiences
        except Exception as e:
            print(f"Error in planning: {e}")
            return SearchPlan(
                info_needs=[],
                tools=[],
                keyword_collection=[],
                vector_queries=[],
                page_index=[]
            ), {}
    
    # LLM(integrate)
    def _search(
        self, 
        plan: SearchPlan, 
        result: Result, 
        question: str,
        category: int,
        model_size: str,
        searching_prompt: Optional[str] = None
    ) -> Result:
        """
        Unified search with integration:
          1) Execute all search tools and collect all hits
          2) Deduplicate hits by page_id
          3) Integrate all deduplicated hits together with LLM
        Returns integrated Result.
        """
        all_hits: List[Hit] = []

        # Execute each planned tool and collect all hits
        for tool in plan.tools:
            hits: List[Hit] = []

            if tool == "keyword":
                if plan.keyword_collection:
                    # combine
                    combined_keywords = " ".join(plan.keyword_collection)
                    keyword_results = self._search_by_keyword([combined_keywords], top_k=5)
                    # Flatten the results if they come as List[List[Hit]]
                    if keyword_results and isinstance(keyword_results[0], list):
                        for result_list in keyword_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(keyword_results)
                    all_hits.extend(hits)
                    
            elif tool == "vector":
                if plan.vector_queries:
                    # search_by_vector
                    vector_results = self._search_by_vector(plan.vector_queries, top_k=5)
                    # Flatten the results if they come as List[List[Hit]]
                    if vector_results and isinstance(vector_results[0], list):
                        for result_list in vector_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(vector_results)
                    all_hits.extend(hits)
                    
            elif tool == "page_index":
                if plan.page_index:
                    page_results = self._search_by_page_index(plan.page_index)
                    # Flatten the results if they come as List[List[Hit]]
                    if page_results and isinstance(page_results[0], list):
                        for result_list in page_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(page_results)
                    all_hits.extend(hits)

        # Deduplicate hits by page_id
        if not all_hits:
            return result
        
        unique_hits: Dict[str, Hit] = {}  # page_id -> Hit
        hits_without_id: List[Hit] = []  # no page_id hits
        for hit in all_hits:
            if hit.page_id:
                if hit.page_id not in unique_hits:
                    unique_hits[hit.page_id] = hit
                else:
                    existing_hit = unique_hits[hit.page_id]
                    existing_score = existing_hit.meta.get("score", 0) if existing_hit.meta else 0
                    current_score = hit.meta.get("score", 0) if hit.meta else 0
                    if current_score > existing_score:
                        unique_hits[hit.page_id] = hit
            else:
                hits_without_id.append(hit)
        
        all_unique_hits = list(unique_hits.values()) + hits_without_id
        sorted_hits = sorted(all_unique_hits, 
                           key=lambda h: h.meta.get("score", 0) if h.meta else 0, 
                           reverse=True)
        
        return self._integrate(sorted_hits, result, question, category, model_size)

    def _search_no_integrate(self, plan: SearchPlan, result: Result, question: str) -> Result:
        """
        Search without integration:
          1) Execute search tools
          2) Collect all hits without LLM integration
          3) Format hits as plain text results
        Returns Result with raw search hits formatted as content.
        """
        all_hits: List[Hit] = []

        # Execute each planned tool and collect hits
        for tool in plan.tools:
            hits: List[Hit] = []

            if tool == "keyword":
                if plan.keyword_collection:
                    # combine
                    combined_keywords = " ".join(plan.keyword_collection)
                    keyword_results = self._search_by_keyword([combined_keywords], top_k=5)
                    # Flatten the results if they come as List[List[Hit]]
                    if keyword_results and isinstance(keyword_results[0], list):
                        for result_list in keyword_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(keyword_results)
                    all_hits.extend(hits)
                    
            elif tool == "vector":
                if plan.vector_queries:
                    # combine
                    vector_results = self._search_by_vector(plan.vector_queries, top_k=5)
                    # Flatten the results if they come as List[List[Hit]]
                    if vector_results and isinstance(vector_results[0], list):
                        for result_list in vector_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(vector_results)
                    all_hits.extend(hits)
                    
            elif tool == "page_index":
                if plan.page_index:
                    page_results = self._search_by_page_index(plan.page_index)
                    # Flatten the results if they come as List[List[Hit]]
                    if page_results and isinstance(page_results[0], list):
                        for result_list in page_results:
                            hits.extend(result_list)
                    else:
                        hits.extend(page_results)
                    all_hits.extend(hits)

        # Format all hits as text content without integration
        if not all_hits:
            return result
        
        unique_hits: Dict[str, Hit] = {}
        hits_without_id: List[Hit] = []
        for hit in all_hits:
            if hit.page_id:

                if hit.page_id not in unique_hits:
                    unique_hits[hit.page_id] = hit
                else:
                    existing_hit = unique_hits[hit.page_id]
                    existing_score = existing_hit.meta.get("score", 0) if existing_hit.meta else 0
                    current_score = hit.meta.get("score", 0) if hit.meta else 0
                    if current_score > existing_score:
                        unique_hits[hit.page_id] = hit
            else:
                hits_without_id.append(hit)
        
        evidence_text = []
        sources = []
        seen_sources = set()
        
        all_unique_hits = list(unique_hits.values()) + hits_without_id
        sorted_hits = sorted(all_unique_hits, 
                           key=lambda h: h.meta.get("score", 0) if h.meta else 0, 
                           reverse=True)
        
        for i, hit in enumerate(sorted_hits, 1):
            # Include page_id in evidence text if available
            source_info = f"[{hit.source}]"
            if hit.page_id:
                source_info = f"[{hit.source}]({hit.page_id})"
            evidence_text.append(f"{i}. {source_info} {hit.snippet}")
            
            # Collect unique sources
            if hit.page_id and hit.page_id not in seen_sources:
                sources.append(hit.page_id)
                seen_sources.add(hit.page_id)
        
        formatted_content = "\n".join(evidence_text)
        
        return Result(
            content=formatted_content if formatted_content else result.content,
            sources=sources if sources else result.sources
        )

    def _integrate(
        self, 
        hits: List[Hit], 
        result: Result, 
        question: str,
        category: int,
        model_size: str,
        integration_prompt: Optional[str] = None
    ) -> Result:
        """
        Integrate search hits with LLM to generate question-relevant result.
        """
        
        evidence_text = []
        sources = []
        for i, hit in enumerate(hits, 1):
            # Include page_id in evidence text if available
            source_info = f"[{hit.source}]"
            if hit.page_id:
                source_info = f"[{hit.source}]({hit.page_id})"
            evidence_text.append(f"{i}. {source_info} {hit.snippet}")

            if hit.page_id:
                sources.append(hit.page_id)
        
        evidence_context = "\n".join(evidence_text) if evidence_text else "no search outcome"
        

        max_chars = 29000 * 4
        if len(evidence_context) > max_chars:
            evidence_context = evidence_context[:max_chars]
        

        print(f"### evidence_text len: {len(evidence_text)}")
        system_prompt = self.system_prompts.get("integration")

        # if model_size == "14B" and category == 2:
        #     print("使用14B prompt")
        #     template_prompt = Integrate_PROMPT.format(question=question, evidence_context=evidence_context, result=result.content)
        # else:
        #     template_prompt = Integrate_PROMPT_exp.format(question=question, evidence_context=evidence_context, result=result.content)
        
        template_prompt = Integrate_PROMPT_exp.format(question=question, evidence_context=evidence_context, result=result.content)

        if system_prompt:
            prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_prompt}"
        else:
            prompt = template_prompt

        try:
            # print("============================== integrate prompt ==============================")
            # print(prompt)
            # print("\n")
            response = self.generator.generate_single(prompt=prompt, schema=INTEGRATE_SCHEMA)
            tokens = response["response"]["usage"]["total_tokens"]
            print("intergrate tokens:", tokens)
            self.total_tokens += tokens
            print("sum_tokens:", self.total_tokens)
            # print("============================== integrate response ==============================")
            # print(response)
            # print("\n")
            data = response.get("json") or json.loads(response["text"])
            
            llm_sources = data.get("sources", sources)
            if llm_sources:
                sources_list = []
                for s in llm_sources:
                    if s is not None:
                        sources_list.append(str(s))
                sources = sources_list if sources_list else sources
            else:
                sources = sources
            
            return Result(
                content=data.get("content", ""),
                sources=sources
            )
        except Exception as e:
            print(f"Error in integration: {e}")
            return result

    # ---- search channels ----
    def _search_by_keyword(self, query_list: List[str], top_k: int = 3) -> List[List[Hit]]:
        r = self.retrievers.get("keyword")
        if r is not None:
            try:
                # BM25Retriever return List[List[Hit]]
                return r.search(query_list, top_k=top_k)
            except Exception as e:
                print(f"Error in keyword search: {e}")
                return []
        # naive fallback: scan pages for substring
        out: List[List[Hit]] = []
        for query in query_list:
            query_hits: List[Hit] = []
            q = query.lower()
            for i, p in enumerate(self.page_store.load()):
                if q in p.content.lower() or q in p.header.lower():
                    snippet = p.content
                    query_hits.append(Hit(page_id=str(i), snippet=snippet, source="keyword", meta={}))
                    if len(query_hits) >= top_k:
                        break
            out.append(query_hits)
        return out

    def _search_by_vector(self, query_list: List[str], top_k: int = 3) -> List[List[Hit]]:
        r = self.retrievers.get("vector")
        if r is not None:
            try:
                return r.search(query_list, top_k=top_k)
            except Exception as e:
                print(f"Error in vector search: {e}")
                return []
        # fallback: none
        return []

    def _search_by_page_index(self, page_index: List[int]) -> List[List[Hit]]:
        r = self.retrievers.get("page_index")
        if r is not None:
            try:
                query_string = ",".join([str(idx) for idx in page_index])
                hits = r.search([query_string], top_k=len(page_index))
                return hits if hits else []
            except Exception as e:
                print(f"Error in page index search: {e}")
                return []

        out: List[Hit] = []
        for idx in page_index:
            p = self.page_store.get(idx)
            if p:
                out.append(Hit(page_id=str(idx), snippet=p.content, source="page_index", meta={}))
        return [out]  # List[List[Hit]]
        
    # ---- reflection & summarization ----
    # exp
    def _reflection(
        self, 
        request: str, 
        result: Result,
        reflection_situation: str,
        exp_flag: int,
        reflection_prompt: Optional[str] = None,
    ) -> ReflectionDecision:
        """
        - "whether information is enough" 
        - "if not, generate remaining information as a new request"  
        """
        experiences = []
        try:
            system_prompt = self.system_prompts.get("reflection")

            if exp_flag:    # exp
                reflection_situation = "(" + reflection_situation + ")" + f"question:{request} temp_memory:{result.content}"
                print(f"reflection_situation: {reflection_situation}")
                experiences = retrieve_experience(reflection_situation, name="Reflection")
                experience = "\n".join(
                    f"({i+1})experience:{e['experience']}\n\n"
                    for i, e in enumerate(experiences)
                )
                print("experience:")
                print(experience)

                template_check_prompt = InfoCheck_PROMPT_exp.format(request=request, result=result.content, experience=experience)
            else:
                template_check_prompt = InfoCheck_PROMPT.format(request=request, result=result.content)
            
            if system_prompt:
                check_prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_check_prompt}"
            else:
                check_prompt = template_check_prompt
            check_prompt_chars = len(check_prompt)
            estimated_check_tokens = check_prompt_chars // 4
            print(f"[DEBUG] Reflection check_prompt length: {check_prompt_chars} chars (~{estimated_check_tokens} tokens)")
            
            # print("============================== reflection prompt ==============================")
            # print(check_prompt)
            # print("\n")

            check_response = self.generator.generate_single(prompt=check_prompt, schema=INFO_CHECK_SCHEMA)
            tokens = check_response["response"]["usage"]["total_tokens"]
            print("reflection tokens:", tokens)
            self.total_tokens += tokens
            print("sum_tokens:", self.total_tokens)
            check_data = check_response.get("json") or json.loads(check_response["text"])
            
            if isinstance(check_data, list):
                check_data = check_data[0] if check_data else {}
            # print("============================== reflection response ==============================")
            # print(check_response)
            # print("\n")

            enough = check_data.get("enough", False)
            
            # If there is enough information, return directly
            if enough:
                return ReflectionDecision(enough=True, new_request=None), experiences
            
            # Step 2: Generate a list of new requests
            if exp_flag:
                template_generate_prompt = GenerateRequests_PROMPT_exp.format(
                    request=request, 
                    result=result.content,
                    experience=experience
                )
            else:
                template_generate_prompt = GenerateRequests_PROMPT.format(
                    request=request, 
                    result=result.content
                )

            if system_prompt:
                generate_prompt = f"User Instructions: {system_prompt}\n\n System Prompt: {template_generate_prompt}"
            else:
                generate_prompt = template_generate_prompt
            generate_prompt_chars = len(generate_prompt)
            estimated_generate_tokens = generate_prompt_chars // 4
            print(f"[DEBUG] Reflection generate_prompt length: {generate_prompt_chars} chars (~{estimated_generate_tokens} tokens)")
            
            generate_response = self.generator.generate_single(prompt=generate_prompt, schema=GENERATE_REQUESTS_SCHEMA)
            tokens = generate_response["response"]["usage"]["total_tokens"]
            print("generate tokens:", tokens)
            self.total_tokens += tokens
            print("sum_tokens:", self.total_tokens)
            generate_data = generate_response.get("json") or json.loads(generate_response["text"])

            if isinstance(generate_data, list):
                generate_data = generate_data[0] if generate_data else {}
            # Get the list of requests and convert to string
            new_requests_list = generate_data.get("new_requests", [])
            new_request = None
            
            if new_requests_list and isinstance(new_requests_list, list):
                new_request = " ".join(new_requests_list)
            
            return ReflectionDecision(
                enough=False,
                new_request=new_request
            ), experiences
            
        except Exception as e:
            print(f"Error in reflection: {e}")
            print("reflection ERROR")
            return ReflectionDecision(enough=True, new_request=None), experiences
