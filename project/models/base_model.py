from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict

class BaseBettingModel(ABC):
    """Interfaz estándar para todos los algoritmos de predicción."""
    
    @abstractmethod
    def train(self, df_train: pd.DataFrame) -> None:
        """Entrena el modelo con datos históricos."""
        pass

    @abstractmethod
    def predict_proba(self, home_team: str, away_team: str) -> Dict[str, float]:
        """Retorna un diccionario con las probabilidades { '1': x, 'X': y, '2': z }."""
        pass
