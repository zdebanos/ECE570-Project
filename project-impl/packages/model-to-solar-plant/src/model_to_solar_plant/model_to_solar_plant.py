import numpy as np
from abc import ABC, abstractmethod

# The mapper interface
class Model2SolarPlantMapper(ABC):
    @abstractmethod
    def map(self, timestamps: np.ndarray, model_outputs: np.ndarray) -> np.ndarray:
        pass

# Creator with Factory method
class Model2SolarPlantMapperCreator(ABC):
    @abstractmethod
    def create_mapper(self, *args, **kwargs) -> Model2SolarPlantMapper:
        """The Factory Method for mapper creation"""
        pass
