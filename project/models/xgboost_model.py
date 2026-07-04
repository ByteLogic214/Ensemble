import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Dict, List
from sklearn.preprocessing import LabelEncoder
from models.base_model import BaseBettingModel
from logs.logger import get_logger

logger = get_logger(__name__)

class XGBoost1X2Model(BaseBettingModel):
    """Modelo XGBoost para predicción multiclase de resultados de fútbol (1X2)."""
    
    def __init__(self, features: List[str]):
        self.features = features
        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            learning_rate=0.05,
            max_depth=4,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.target_encoder = LabelEncoder()
        
    def train(self, df_train: pd.DataFrame) -> None:
        """
        Entrena el modelo usando las features tabulares.
        El target debe ser 'result' ('H' para Home, 'D' para Draw, 'A' para Away).
        """
        logger.info("Preparando datos para entrenar XGBoost...")
        
        X = df_train[self.features]
        # Codificamos [A, D, H] a [0, 1, 2]
        y = self.target_encoder.fit_transform(df_train['result'])
        
        logger.info(f"Iniciando entrenamiento con {len(X)} muestras y {len(self.features)} features.")
        self.model.fit(X, y)
        logger.info("Entrenamiento de XGBoost completado exitosamente.")

    def predict_proba(self, match_features: pd.DataFrame) -> Dict[str, float]:
        """
        Infiere la probabilidad para un partido específico.
        match_features debe ser una fila (o DataFrame) con las mismas columnas de self.features.
        """
        if self.model is None:
            raise ValueError("El modelo XGBoost no ha sido entrenado.")
            
        X_pred = match_features[self.features]
        probs = self.model.predict_proba(X_pred)[0]
        
        # Mapear las probabilidades de vuelta a las clases originales (H, D, A -> 1, X, 2)
        classes = self.target_encoder.classes_
        prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
        
        # Estandarizar la salida a formato de apuestas (1, X, 2)
        return {
            "1": prob_dict.get('H', 0.0),
            "X": prob_dict.get('D', 0.0),
            "2": prob_dict.get('A', 0.0)
        }
