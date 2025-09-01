from openai import OpenAI

client = OpenAI(
    api_key="sk-aitunnel-xxx", # Ключ из нашего сервиса
    base_url="https://api.aitunnel.ru/v1/",
)

chat_result = client.chat.completions.create(
    messages=[{"role": "user", "content": "Скажи интересный факт"}],
    model="llama-4-scout",
    max_tokens=50000, # Старайтесь указывать для более точного расчёта цены
)
print(chat_result.choices[0].message)