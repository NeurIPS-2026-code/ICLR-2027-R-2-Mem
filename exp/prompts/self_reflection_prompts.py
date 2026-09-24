####### Planning #######
Planning_system_prompt = """
{role_desc}

WARNING:
Your goal is to derive GENERALIZABLE planning experience in the form:
"IF <abstract situation> THEN <retrieval planning strategy>"

CRITICAL INSTRUCTION:
- You MUST treat the [DIAGNOSED REASON] as authoritative guidance.
- You MUST NOT copy surface details from the trace.
- You MUST abstract into a reusable pattern applicable to future queries.

STRICT REQUIREMENTS:
- Situation must describe a TYPE of query or information gap (not specific facts)
- Experience must describe HOW to construct:
  - info_needs
  - tool selection strategy
"""

Planning_prompt = """
***Planning Role Explanation***
Planning is NOT responsible for answering the QUESTION.
Its role is to translate the QUESTION into an explicit and executable retrieval plan.

"plan": planning retrieval. Analyze the current question and determine what information is required and how it should be retrieved.
    - "info_needs": Concrete sub-questions or missing facts that must be resolved in order to fully answer the current QUESTION.
    - "tools": The retrieval strategy types selected for this plan, chosen from ["keyword", "vector", "page_index"].
    - "keyword_collection": Short, high-signal keywords or phrases for exact-match retrieval of specific entities, names, or attributes.
    - "vector_queries": Natural-language queries used for semantic retrieval of conceptually related information.
    - "page_index": Known page indices to be re-read in full

***Inputs***
[QUESTION]: {QUESTION}
[TRACE]({content_type}): {info}
[DIAGNOSED REASON]: {content_reason}

***Your task***:
1. Thinking (Do not output):
    Analyze this trace based on the DIAGNOSED REASON and your own reasoning.
    Focus on:
    - info_needs design quality
    - tool selection correctness
    - redundancy / missing coverage
    - efficiency
    ...

2. Summarize the Trace (1-2 sentences):
    Briefly describe this Trace, including the question and the plan.

3. abstract Situation(Do NOT include concrete entities):
    Summarize the abstract planning situation, What is this question trying to ask?
    - Multi-entity query
    - Time-related query
    - list/set
    - single_fact
    - multi_hop
    ...

4. Experience:
    {planning_task}

Strict JSON Requirement:
- You MUST return a valid JSON object.
- Ensure all inner double quotes within string values are escaped with a backslash (e.g., use \" instead of ").
- Do not include any trailing commas.

Output ONLY valid JSON:
{{
    "thinking": "<logic analysis grounded in the trace>",
    "summary": "<Briefly describe this Trace(question and plan)>",
    "situation": "<abstract situation>",
    "experience": "IF <specific condition in Question/Memory or Question type> THEN <specific info_need/tool action>"
}}
"""

Planning_prompt_extra = """
***Planning Role Explanation***
Planning is NOT responsible for answering the QUESTION.
Its role is to translate the QUESTION into an explicit and executable retrieval plan.

"plan": planning retrieval. Analyze the current question and determine what information is required and how it should be retrieved.
    - "info_needs": Concrete sub-questions or missing facts that must be resolved in order to fully answer the current QUESTION.
    - "tools": The retrieval strategy types selected for this plan, chosen from ["keyword", "vector", "page_index"].
    - "keyword_collection": Short, high-signal keywords or phrases for exact-match retrieval of specific entities, names, or attributes.
    - "vector_queries": Natural-language queries used for semantic retrieval of conceptually related information.
    - "page_index": Known page indices to be re-read in full (Avoid using it unless absolutely necessary)

***Inputs***
[QUESTION]: {QUESTION}
[TRACE]({content_type}): {info}
[DIAGNOSED REASON]: {content_reason}

***Your task***:
1. Thinking (Do not output):
    Analyze this trace based on the DIAGNOSED REASON and your own reasoning.
    Focus on:
    - info_needs design quality
    - tool selection correctness
    - redundancy / missing coverage
    - efficiency
    ...

2. Summarize the Trace (1-2 sentences):
    Briefly describe this Trace, including the question and the plan.

3. abstract Situation(Do NOT include concrete entities):
    Summarize the abstract planning situation, What is this question trying to ask?
    - Multi-entity query
    - Time-related query
    - list/set
    - single_fact
    - multi_hop
    ...

4. Experience:
    {planning_task}

Strict JSON Requirement:
- You MUST return a valid JSON object.
- Ensure all inner double quotes within string values are escaped with a backslash (e.g., use \" instead of ").
- Do not include any trailing commas.

Output ONLY valid JSON:
{{
    "thinking": "<logic analysis grounded in the trace>",
    "summary": "<Briefly describe this Trace(question and plan)>",
    "situation": "<abstract situation>",
    "experience": "IF <specific condition in Question/Memory or Question type> THEN <specific info_need/tool action>"
}}
"""




####### Reflection #######
Reflection_system_prompt = """
{role_desc}

WARNING:
Your goal is to derive GENERALIZABLE reflection experience in the form:
"IF <abstract situation> THEN <decision strategy>"

CRITICAL INSTRUCTION:
- You MUST treat the [DIAGNOSED REASON] as authoritative guidance.
- You MUST NOT copy surface details from the trace.
- You MUST abstract into a reusable decision pattern.

STRICT REQUIREMENTS:
- Situation must describe a TYPE of (query + temp_memory), NOT specific facts
- Experience must describe HOW to:
  - judge sufficiency (enough True/False)
  - decide whether to stop or continue retrieval
  - generate new_request if needed
"""

Reflection_prompt = """
***Reflection Role Explanation***
"temp_memory": the current integrated factual summary about that QUESTION. it is intended to contain all useful known information so far
"decision": reflecting on completeness and generating a follow-up request if needed
    - enough: Whether current content is sufficient to answer the QUESTION.
    - new_request: Follow-up retrieval request if enough is false.


***Inputs***
[QUESTION]: {QUESTION}
[TRACE]({content_type}): {info}
[DIAGNOSED REASON]: {content_reason}

Your task:
1. Thinking (Do not output):
    Analyze this reflection based on DIAGNOSED REASON and your reasoning.
    Focus on:
    - sufficiency judgment correctness
    - whether the system stopped too early or too late
    - whether new_request is necessary and well-targeted
    - whether answer completeness is correctly recognized

2. Summarize the Trace (1-2 sentences):
    Briefly describe the question, temp_memory.

3. Abstract Situation (Do NOT include concrete entities):
    Summarize the abstract reflection situation(question and temp memory), considering:
    What is this question trying to ask?
    What does the temp_memory already contain?
    Whether the current information in temp_memory is already sufficient to solve the problem?

4. Experience:
    {experience}
    Formulate a reusable IF-THEN decision rule that explicitly states:
    IF <abstract situation(question + temp_memory)> THEN <decision strategy>
    decision strategy must be detailed: (enough = true) or (enough = false + generate a detailed new_request)
    The rule MUST be discriminative, not tautological.

Strict JSON Requirement:
- You MUST return a valid JSON object.
- Ensure all inner double quotes within string values are escaped with a backslash (e.g., use \" instead of ").
- Do not include any trailing commas.

Output ONLY valid JSON:
{{
    "thinking": "<analysis of reflection success/failure>",
    "summary": "<Briefly describe this Trace(question and temp memory)>",
    "situation": "<abstract situation>",
    "experience": "IF <abstract situation> THEN <decision strategy>"
}}
"""