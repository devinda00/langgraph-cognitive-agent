# langgraph_cognitive_arch/agent/state.py
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field

class MindDecision(BaseModel):
    """
    The structured output from the Mind's reasoning process.
    The Mind must decide whether the user's query is simple enough
    for an immediate answer or if it requires deeper thought from the Brain.
    """
    reasoning: str = Field(description="The thought process and reasoning behind the decision.")
    should_escalate: bool = Field(description="True if the query is complex and requires the Brain's full reasoning power. False otherwise.")
    simple_response: str = Field(description="If should_escalate is False, provide a direct, simple answer here. Otherwise, this can be empty.", default="")

class AgentState(TypedDict):
    """The state of our cognitive agent."""
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    mind_decision: Optional[MindDecision]
