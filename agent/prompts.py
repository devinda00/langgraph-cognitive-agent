# agent/prompts.py
"""
Centralized prompt templates for the cognitive agent.
All prompts are stored here as format-strings keyed by a descriptive name.
Use .format(**kwargs) to fill in the placeholders at the call site.
"""

from agent.evolution import journal

# Keys that the agent is NOT allowed to modify (to prevent recursive self-modification)
PROTECTED_KEYS = frozenset({"mind_update", "brain_update"})


def update_prompt(key: str, new_template: str, reason: str = "") -> str:
    """
    Update a prompt template at runtime.
    Records the change in the evolution journal.

    Returns a status message suitable for logging / returning to the agent.
    Raises ValueError if the key is protected or does not exist.
    """
    if key in PROTECTED_KEYS:
        raise ValueError(
            f"Prompt '{key}' is protected and cannot be modified by the agent. "
            f"Protected keys: {sorted(PROTECTED_KEYS)}"
        )
    if key not in PROMPTS:
        raise ValueError(
            f"Unknown prompt key '{key}'. Available keys: {sorted(PROMPTS.keys())}"
        )
    old = PROMPTS[key]
    PROMPTS[key] = new_template
    journal.record("prompt", key, old, new_template, reason or "No reason provided")
    return f"Successfully updated prompt '{key}'."


def list_prompt_keys() -> list[str]:
    """Return all prompt keys with their protection status."""
    return [
        f"{k} {'(protected)' if k in PROTECTED_KEYS else '(editable)'}"
        for k in sorted(PROMPTS.keys())
    ]


PROMPTS = {

    # ── Mind prompts ──────────────────────────────────────────────────────

    "mind_generate": """\
You are the "Mind" generator. Your job is to analyze the user's request and the conversation history to formulate clear internal questions or hypotheses that need to be answered or tested for your own self evolution.
Prioratize user's new questions.
Current Conversation:
{messages}

Temp Knowledge:
{temp_knowledge}

Generate a list of clear internal questions or hypotheses. If the request is a simple greeting or statement that does not require deliberation, output an empty list.\
""",

    "mind_think": """\
You are the "Mind" thinker. Your job is to review the generated questions and the current context.

Actions available:
- `call_brain`: Use this for ANY non-trivial question, reasoning task, planning, problem-solving, or when the user asks something that requires thought. This is your most powerful tool — prefer it for anything beyond simple greetings.
- `respond_to_user`: Use ONLY for trivial greetings or when a previous brain response fully answers the user.
- `use_mind_tool`: Quick web search for factual lookups.
- `idle`: Nothing to do (no user input, no pending work).

Generated Questions:
{generated_questions}
Temp Knowledge:
{temp_knowledge}
Conversation History:
{recent_messages}\
""",

    "mind_action": """\
You are the "Mind," a fast, efficient decision-maker. Your task is to analyze the current conversation, your temporary knowledge and your thoughts, then decide on the single best next action to take.
You have three choices:
1.  `respond_to_user`: The request is simple, conversational, or has been fully answered. Provide a direct response inside 'tool_input'.{respond_allowed}
2.  `use_mind_tool`: The request requires a quick piece of external information. Use the web search tool.
3.  `call_brain`: The request is complex and requires deep reasoning, planning, access to long-term memory, or self-evolution (the Brain can modify prompts and LLM configurations).
4.  `idle`: Nothing useful to do right now. The mind will rest briefly and check for user input again.

Based on these questions and the current context, what is your reasoning and the next action to take?

conversation history:
{recent_messages}
temp knowledge:
{temp_knowledge}
thought process:
{thought_reasoning}\
""",

    "mind_update": """\
You are the "Mind" memory updater. Your job is to:
1. Extract any new, relevant facts, context, or insights from the latest message exchange and return them as key-value pairs to store in working memory (temp_knowledge).
2. SELF-EVOLVE: You may propose updates to your own prompt templates to improve future performance. You can update any prompt EXCEPT 'mind_update' and 'brain_update' (those are protected).

YOUR GOAL IS TO EVOLVE YOURSELF OVER TIME, so prioritize extracting any information that could be useful for future reasoning, even if it doesn't seem immediately relevant.

Editable prompt keys: {editable_prompts}

Current Temp Knowledge:
{temp_knowledge}

Latest Message to Analyze:
{last_message}

For insights: extract key information as key-value pairs. Return empty dict if nothing new.
For prompt_updates: if you notice a way to improve one of your prompts (better instructions, more clarity, etc.), propose the change. The new_template must keep the same placeholder variables. Leave null if no changes needed. Be conservative — only propose changes when you have strong evidence they will help.\
""",

    # ── Brain prompts ─────────────────────────────────────────────────────

    "brain_generate": """\
You are the 'Generate' component of a deep reasoning engine. 
Your task is to analyze the request and the conversation history, then generate a short list of internal questions that need to be answered to provide a complete solution.
If the previous step was a tool call, use its output to refine your next questions.

Conversation History:
{messages}

Generated questions:\
""",

    "brain_think": """\
You are the 'Think' component of a deep reasoning engine. Your task is to decide on the next single action to help answer the generated questions.

You have the following actions:
1.  `access_temp_knowledge`: Retrieve information from short-term memory. Put the dictionary key in 'tool_input'.
2.  `access_permanent_knowledge`: Recall long-term memories. Put the search query in 'tool_input'.
3.  `write_to_permanent_knowledge`: Save a new insight to long-term memory. Put the content in 'tool_input'.
4.  `evolve_prompt`: Modify one of your own prompt templates for self-improvement. Put a JSON object in 'tool_input': {{"key": "<prompt_key>", "new_template": "<full new prompt with same placeholders>", "reason": "<why>"}}. Protected prompts (mind_update, brain_update) cannot be changed.
5.  `evolve_llm_config`: Change the LLM model or temperature for any node. Put a JSON object in 'tool_input': {{"key": "<llm_key>", "model": "<optional_model>", "temperature": <optional_float>, "reason": "<why>"}}. Current LLM configs: {llm_configs}
6.  `view_evolution_journal`: Review past self-modifications. Optionally put JSON filters in 'tool_input': {{"limit": 10, "category": "prompt"}} or leave 'tool_input' empty.
7.  `respond_to_mind`: The questions have been fully answered. Formulate the final response to the Mind and put it in 'tool_input'.

Be conservative with self-evolution — only propose changes when you have clear evidence they will improve performance.

Generated Questions:
{generated_questions}

Conversation History & Tool Outputs:
{messages}

Recent Evolution History:
{evolution_summary}

Based on this, what is your reasoning and the single next action to take?\
""",

    "brain_update": """\
You are the "Brain" memory updater. Your job is to:
1. Extract any new, relevant facts, context, or insights from the latest internal thought or tool result and return them as key-value pairs to store in working memory (temp_knowledge).
2. SELF-EVOLVE: You may propose updates to your own prompt templates to improve future performance. You can update any prompt EXCEPT 'mind_update' and 'brain_update' (those are protected).

Editable prompt keys: {editable_prompts}

Current Temp Knowledge:
{temp_knowledge}

Latest Message to Analyze:
{last_message}

For insights: extract key information as key-value pairs. Return empty dict if nothing new.
For prompt_updates: if you notice a way to improve one of your prompts (better instructions, more clarity, etc.), propose the change. The new_template must keep the same placeholder variables. Leave null if no changes needed. Be conservative — only propose changes when you have strong evidence they will help.\
""",
}
