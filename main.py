from openai import OpenAI

from core.config import setting

client = OpenAI(
    api_key=setting.llm.openrouter_api_key.get_secret_value(), # Ключ из нашего сервиса
    base_url="https://api.aitunnel.ru/v1/",
)

chat_result = client.chat.completions.create(
    messages=[{"role": "user", "content": "Скажи интересный факт"}],
    model="gpt-4o-mini",
    max_tokens=50000, # Старайтесь указывать для более точного расчёта цены
)

print(chat_result.choices[0].message)