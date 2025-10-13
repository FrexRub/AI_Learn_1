import asyncio
import json
from langchain_openai import ChatOpenAI

from core.config import setting
from llm.agent import VacancyClassificationAgent


def get_openrouter_llm(model="gpt-4o-mini"):
    return ChatOpenAI(
        model=model,
        api_key=setting.llm.openrouter_api_key.get_secret_value(),
        base_url="https://api.aitunnel.ru/v1/",
        temperature=0,
    )


async def main():
    """Демонстрация работы агента"""
    agent = VacancyClassificationAgent()

    # Тестовые примеры
    test_cases = [
        "Требуется Python разработчик для создания веб-приложения на Django. Постоянная работа, полный рабочий день.",
        "Ищу заказы на создание логотипов и фирменного стиля. Работаю в Adobe Illustrator.",
        "Нужен 3D-аниматор для краткосрочного проекта создания рекламного ролика.",
        "Резюме: опытный маркетолог, ищу удаленную работу в сфере digital-маркетинга",
        "Ищем фронтенд-разработчика React в нашу команду на постоянную основе",
    ]

    print("🤖 Демонстрация работы агента классификации вакансий\n")

    for i, description in enumerate(test_cases, 1):
        print(f"📋 Тест {i}:")
        print(f"Описание: {description}")

        try:
            result = await agent.classify(description)
            print("Результат классификации:")
            print(json.dumps(result, ensure_ascii=False, indent=2))

        except Exception as e:
            print(f"❌ Ошибка: {e}")

        print("-" * 80)


if __name__ == "__main__":
    # llm = get_openrouter_llm(model="gpt-4o-mini")
    # response = llm.invoke("Кто ты?")
    # print(response.content)

    asyncio.run(main())
