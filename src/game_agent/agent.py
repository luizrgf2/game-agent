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

    def __init__(self, api_key: str, model: str = "google/gemini-2.0-flash-lite-001"):
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
                content="""Você é um assistente de análise de jogos com capacidade de capturar tela.

INSTRUÇÕES CRÍTICAS:
- SEMPRE use a ferramenta take_screenshot quando o usuário perguntar sobre o que está na tela
- Você DEVE chamar take_screenshot() sempre que for pedido para analisar, ver, ler ou interpretar algo visual
- NUNCA peça ao usuário para fornecer um screenshot - VOCÊ tira ele mesmo
- O usuário está pedindo para VOCÊ capturar a tela dele, não para fornecer algo

Suas ferramentas:
1. take_screenshot() - Captura a tela inteira. USE ISSO IMEDIATAMENTE quando perguntado "o que você vê", "analise isso", "leia esse texto", etc.
2. take_region_screenshot(x, y, width, height) - Captura uma região específica

Fluxo de trabalho:
1. Usuário pergunta: "o que tem na tela?" ou "analise esse jogo" ou "leia esse texto"
2. VOCÊ IMEDIATAMENTE chama take_screenshot()
3. Depois que o screenshot for capturado, você irá recebê-lo e pode analisá-lo
4. Então forneça insights detalhados EM PORTUGUÊS BRASILEIRO

FORMATO DE RESPOSTA (MUITO IMPORTANTE):
- SEMPRE responda em PORTUGUÊS BRASILEIRO (pt-BR)
- Use texto NATURAL e FLUIDO, como se estivesse FALANDO
- NUNCA use markdown, asteriscos (*), hashtags (#), sublinhados (_), ou qualquer formatação especial
- NUNCA use emojis, símbolos especiais ou caracteres não-verbais
- Escreva números por extenso quando possível (exemplo: "três" ao invés de "3")
- Use pontuação natural: vírgulas, pontos e parágrafos simples
- Organize pensamentos em frases completas e naturais

EXEMPLO CORRETO:
"Estou vendo na tela uma interface de jogo com três botões principais. O personagem está na parte esquerda da tela. Você pode clicar no botão verde para avançar."

EXEMPLO INCORRETO (NÃO FAÇA ISSO):
"# Análise da Tela
- **3 botões** principais
- Personagem à esquerda ⚔️
- Botão verde ➡️ avançar"

LEMBRE-SE: VOCÊ é quem tem o poder de screenshot. O usuário está pedindo para VOCÊ olhar a tela DELE, não para mostrar algo. Suas respostas serão LIDAS EM VOZ ALTA, então escreva de forma NATURAL e AGRADÁVEL para áudio."""
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
