# src/retrievers/base.py
from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Повертає список кортежів (текст чанку, score)
        """
        pass
