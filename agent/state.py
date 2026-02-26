# langgraph_cognitive_arch/agent/state.py
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from agent.permanent_knowledge import VectorStore

class MindGenerateAction(BaseModel):
    """The generated questions or hypotheses from the Mind's generate node."""
    questions: List[str] = Field(description="A list of clear internal questions or hypotheses generated from the current context.")

class PromptUpdate(BaseModel):
    """A proposed update to one of the agent's own prompt templates."""
    key: str = Field(description="The prompt key to update (e.g. 'mind_generate', 'mind_think'). Cannot be 'mind_update' or 'brain_update'.")
    new_template: str = Field(description="The full replacement prompt template. Must include the same {placeholder} variables as the original.")
    reason: str = Field(description="Why this change will improve the agent's performance.")

class MindUpdateAction(BaseModel):
    """Distilled insights to be saved to temp_knowledge from the last action."""
    insights: Dict[str, Any] = Field(description="Key-value pairs of important facts, context, or insights extracted from the latest interaction. Should be empty if nothing new was learned.")
    prompt_updates: Optional[List[PromptUpdate]] = Field(default=None, description="Optional list of proposed prompt updates for self-evolution. Leave null if no prompt changes are needed.")

class MindAction(BaseModel):
    """The decision and action from the Mind's think node."""
    reasoning: str = Field(description="The chain of thought leading to the decision.")
    action: str = Field(description="The chosen action. Must be one of: 'respond_to_user', 'call_brain', 'use_mind_tool', 'idle'.")
    tool_input: Optional[str] = Field(description="The input for the chosen tool, or the direct text response to the user if the action is 'respond_to_user'.")

class BrainThought(BaseModel):
    """The decision and action from the Brain's think node."""
    reasoning: str = Field(description="The chain of thought leading to the decision.")
    action: str = Field(description="The chosen action. Must be one of: 'access_temp_knowledge', 'access_permanent_knowledge', 'write_to_permanent_knowledge', 'evolve_prompt', 'evolve_llm_config', 'view_evolution_journal', 'respond_to_mind'.")
    tool_input: Optional[str] = Field(description="The input for the chosen tool (plain text or JSON depending on action), or the direct text response to the Mind if the action is 'respond_to_mind'.")

class BrainUpdateAction(BaseModel):
    """Distilled insights to be saved to temp_knowledge from the Brain's last action."""
    insights: Dict[str, Any] = Field(description="Key-value pairs of important facts, context, or insights extracted from the latest interaction. Should be empty if nothing new was learned.")
    prompt_updates: Optional[List[PromptUpdate]] = Field(default=None, description="Optional list of proposed prompt updates for self-evolution. Leave null if no prompt changes are needed.")

class AgentState(TypedDict):
    """The state of our cognitive agent."""
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    
    # Represents "Temp Knowledge" as a Python dictionary
    temp_knowledge: Dict[str, Any]
    
    # Handle to the vector store for permanent knowledge
    permanent_knowledge: VectorStore

    # Whether the current loop iteration was triggered by new user input
    has_new_input: bool

    # Mind's intermediate state
    mind_thought: MindAction
    mind_action: MindAction

    # Brain's intermediate state
    generated_questions: List[str]
    brain_thought: BrainThought
