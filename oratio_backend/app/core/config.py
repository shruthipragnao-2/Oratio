from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    env: str = Field(default="dev")
    use_gpu: bool = Field(default=False)
    spacy_model: str = Field(default="en_core_web_lg")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()


