# langgraph_cognitive_arch/agent/state.py
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from agent.permanent_knowledge import VectorStore

class MindGenerateAction(BaseModel):
    """The generated questions or hypotheses from the Mind's generate node."""
    questions: List[str] = Field(description="A list of clear internal questions or hypotheses generated from the current context.")

class MindUpdateAction(BaseModel):
    """Distilled insights to be saved to temp_knowledge from the last action."""
    insights: Dict[str, Any] = Field(description="Key-value pairs of important facts, context, or insights extracted from the latest interaction. Should be empty if nothing new was learned.")

class MindAction(BaseModel):
    """The decision and action from the Mind's think node."""
    reasoning: str = Field(description="The chain of thought leading to the decision.")
    action: str = Field(description="The chosen action. Must be one of: 'respond_to_user', 'call_brain', 'use_mind_tool', 'idle'.")
    tool_input: Optional[str] = Field(description="The input for the chosen tool, or the direct text response to the user if the action is 'respond_to_user'.")

class BrainThought(BaseModel):
    """The decision and action from the Brain's think node."""
    reasoning: str = Field(description="The chain of thought leading to the decision.")
    action: str = Field(description="The chosen action. Must be one of: 'access_temp_knowledge', 'access_permanent_knowledge', 'write_to_permanent_knowledge', 'respond_to_mind'.")
    tool_input: Optional[str] = Field(description="The input for the chosen tool, or the direct text response to the Mind if the action is 'respond_to_mind'.")

class BrainUpdateAction(BaseModel):
    """Distilled insights to be saved to temp_knowledge from the Brain's last action."""
    insights: Dict[str, Any] = Field(description="Key-value pairs of important facts, context, or insights extracted from the latest interaction. Should be empty if nothing new was learned.")

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
    mind_action: MindAction

    # Brain's intermediate state
    generated_questions: List[str]
    brain_thought: BrainThought
