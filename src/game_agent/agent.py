"""Game analysis agent using LangGraph."""

import json
import operator
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from .tools import take_screenshot, take_region_screenshot


class AgentState(TypedDict):
    """State for the game agent."""
    messages: Annotated[list[AnyMessage], operator.add]
    screenshot_data: dict | None


class GameAgent:
    """Agent for analyzing game screens and providing insights."""

    def __init__(self, api_key: str, model: str = "google/gemini-2.0-flash-001"):
        """Initialize the game agent.

        Args:
            api_key: OpenRouter API key
            model: Model to use for analysis (OpenRouter format)
        """
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model=model,
        )

        # Bind tools to the LLM
        self.tools = [take_screenshot, take_region_screenshot]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("process_screenshot", self._process_screenshot)

        # Set entry point
        workflow.set_entry_point("agent")

        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            }
        )

        # Add edge from tools to screenshot processor
        workflow.add_edge("tools", "process_screenshot")

        # Add edge from screenshot processor back to agent
        workflow.add_edge("process_screenshot", "agent")

        return workflow.compile()

    def _call_model(self, state: AgentState) -> dict:
        """Call the LLM with the current state.

        Args:
            state: Current agent state

        Returns:
            Updated state with LLM response
        """
        messages = state["messages"]

        # Add system message if this is the first call
        if len(messages) == 1 or not any(isinstance(m, SystemMessage) for m in messages):
            system_message = SystemMessage(
                content="""You are a game analysis assistant. You help users understand and analyze game screens.

Your capabilities:
- Take screenshots of the game screen using the take_screenshot tool
- Analyze game screens using vision to identify UI elements, characters, stats, objectives, etc.
- Provide strategic insights and advice based on what you see
- Read and interpret in-game text, menus, and HUD elements

When a user asks you to analyze something:
1. First, take a screenshot using the take_screenshot tool
2. Then analyze the screenshot content that will be automatically added to the context
3. Provide detailed insights based on what you observe

Be helpful, detailed, and game-focused in your responses."""
            )
            messages = [system_message] + messages

        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _should_continue(self, state: AgentState) -> str:
        """Determine if we should continue or end.

        Args:
            state: Current agent state

        Returns:
            "continue" if there are tool calls, "end" otherwise
        """
        last_message = state["messages"][-1]

        # If there are tool calls, continue to tools node
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"

        # Otherwise, end
        return "end"

    def _process_screenshot(self, state: AgentState) -> dict:
        """Process screenshot tool results and add images to context.

        Args:
            state: Current agent state

        Returns:
            Updated state with screenshot images added as HumanMessage
        """
        messages = state["messages"]
        new_messages = []

        # Look for the last ToolMessage with screenshot data
        for msg in reversed(messages):
            # Debug: print message type and attributes
            print(f"DEBUG: Message type: {type(msg).__name__}")
            if hasattr(msg, "name"):
                print(f"DEBUG: Message name: {msg.name}")
            if hasattr(msg, "content"):
                print(f"DEBUG: Content type: {type(msg.content)}")
                print(f"DEBUG: Content preview: {str(msg.content)[:200]}")

            if hasattr(msg, "name") and "screenshot" in msg.name.lower():
                try:
                    # Parse the tool result
                    content = msg.content
                    if isinstance(content, str):
                        result = json.loads(content)
                    else:
                        result = content

                    print(f"DEBUG: Parsed result keys: {result.keys()}")

                    # Extract base64 image
                    if "base64" in result:
                        base64_image = result["base64"]
                        print(f"DEBUG: Found base64 image, length: {len(base64_image)}")

                        # Create a new HumanMessage with the image
                        image_message = HumanMessage(
                            content=[
                                {
                                    "type": "text",
                                    "text": "Here is the screenshot. Please analyze it:",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    },
                                },
                            ]
                        )
                        new_messages.append(image_message)
                        print("DEBUG: Added image message to queue")
                        break
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error processing screenshot: {e}")
                    import traceback
                    traceback.print_exc()

        print(f"DEBUG: Returning {len(new_messages)} new messages")
        return {"messages": new_messages}

    def run(self, user_input: str) -> dict:
        """Run the agent with user input.

        Args:
            user_input: User's request

        Returns:
            Final state of the agent
        """
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "screenshot_data": None,
        }

        result = self.graph.invoke(initial_state)
        return result

    async def arun(self, user_input: str) -> dict:
        """Run the agent asynchronously with user input.

        Args:
            user_input: User's request

        Returns:
            Final state of the agent
        """
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "screenshot_data": None,
        }

        result = await self.graph.ainvoke(initial_state)
        return result
