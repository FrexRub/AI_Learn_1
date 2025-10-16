from typing import TypedDict

from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from core.config import setting


class ChatSate(TypedDict):
    messages: list[BaseMessage]
    should_continue: bool


class ChatAgent:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        """Инициализация агента"""
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=setting.llm.openrouter_api_key.get_secret_value(),
            base_url="https://api.aitunnel.ru/v1/",
            temperature=temperature,
        )
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> CompiledStateGraph:
        graph = StateGraph(ChatSate)

        graph.add_node("user_input", self._input_user_node)
        graph.add_node("llm_response", self._llm_response_node)

        graph.add_edge(START, "user_input")
        graph.add_edge("user_input", "llm_response")
        graph.add_conditional_edges(
            "llm_response",
            self._should_continue,
            {"continue": "user_input", "end": END},
        )

        return graph.compile()

    def _input_user_node(self, state: ChatSate):
        user_input: str = input("Вы: ")
        if user_input in ("by", "exit", "пока"):
            return {"should_continue": False}

        new_messages = state["messages"] + [HumanMessage(content=user_input)]

        return {"messages": new_messages, "should_continue": True}

    def _llm_response_node(self, state: ChatSate):
        response = self.llm.invoke(state["messages"])
        msg_content = response.content

        print("ИИ: ", msg_content)
        print("type message: ", type(response))
        new_message = state["messages"] + [AIMessage(content=msg_content)]
        return {"messages": new_message}

    def _should_continue(self, state: ChatSate) -> str:
        return "continue" if state.get("should_continue", True) else "end"

    def classify(self):
        """Основной метод для классификации вакансии/услуги"""
        initial_state = {
            "messages": [
                SystemMessage(
                    content="Ты дружелюбный помощник. Отвечай коротко и по делу"
                )
            ],
            "should_continue": True,
        }

        result = self.workflow.invoke(initial_state)
        return result


if __name__ == "__main__":
    print("Добро пожаловать в чат с ИИ!")
    print("Для выхода используйте слова: by, exit, пока")
    print("-" * 50)
    try:
        app = ChatAgent()
        res = app.classify()
    except KeyboardInterrupt:
        print("Чат прерван пользователем")
    else:
        print("До свидания")

