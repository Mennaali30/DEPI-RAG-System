from abc import ABC, abstractmethod

from .Document import Document


class DocumentLoader(ABC):
    @abstractmethod
    def load(self) -> Document:
        pass