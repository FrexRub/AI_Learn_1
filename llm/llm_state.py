from typing import TypedDict


class State(TypedDict):
    """Состояние агента для хранения информации о процессе классификации"""

    description: str
    job_type: str
    category: str
    search_type: str
    confidence_scores: dict[str, float]
    processed: bool
