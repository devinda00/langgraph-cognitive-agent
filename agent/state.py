# langgraph_cognitive_arch/agent/state.py
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langchain_core.pydantic_v1 import BaseModel, Field

class Thought(BaseModel):
    """
    The output of the 'think' node. It contains the reasoning process
    and a decision on how to proceed. For the Mind, this includes the
    decision to escalate.
    """
    reasoning: str = Field(description="The chain of thought leading to the decision.")
    decision: str = Field(description="The decision made. Can be 'act', 'escalate', or 'respond'.")

class AgentState(TypedDict):
    """The state of our cognitive agent."""
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    
    # Intermediate products of the cognitive loops
    generated_question: str
    thought: Thought
    
    # This field will signal escalation from the Mind to the Router
    escalate_to_brain: bool
    
    # Represents the "Temp Knowledge" from your diagram
    knowledge_base: str
