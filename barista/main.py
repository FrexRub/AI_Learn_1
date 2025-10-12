from langchain_openai import ChatOpenAI

from core.config import setting


def get_openrouter_llm(model="gpt-4o-mini"):
    return ChatOpenAI(
        model=model,
        api_key=setting.llm.openrouter_api_key.get_secret_value(),
        base_url="https://api.aitunnel.ru/v1/",
        temperature=0,
    )


if __name__ == "__main__":
    llm = get_openrouter_llm(model="gpt-4o-mini")
    response = llm.invoke("Кто ты?")
    print(response.content)
