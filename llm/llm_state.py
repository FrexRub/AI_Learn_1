from typing import TypedDict
from enum import Enum

# Категории профессий
CATEGORIES = [
    "2D-аниматор",
    "3D-аниматор",
    "3D-моделлер",
    "Бизнес-аналитик",
    "Блокчейн-разработчик",
    ...,
]


class JobType(Enum):
    PROJECT = "проектная работа"
    PERMANENT = "постоянная работа"


class SearchType(Enum):
    LOOKING_FOR_WORK = "поиск работы"
    LOOKING_FOR_PERFORMER = "поиск исполнителя"


class State(TypedDict):
    """Состояние агента для хранения информации о процессе классификации"""

    description: str
    job_type: str
    category: str
    search_type: str
    confidence_scores: Dict[str, float]
    processed: bool
