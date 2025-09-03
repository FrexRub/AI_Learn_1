from pathlib import Path
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent.parent


class ConfigLLM(BaseSettings):
    openrouter_api_key: SecretStr

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", env_file_encoding="utf8", extra="ignore")


class Setting(BaseSettings):
    llm: ConfigLLM = ConfigLLM()


setting = Setting()