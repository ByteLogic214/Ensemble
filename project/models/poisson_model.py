import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson
from typing import Dict
from models.base_model import BaseBettingModel
from logs.logger import get_logger

logger = get_logger(__name__)

class PoissonModel(BaseBettingModel):
    """Modelo de Regresión de Poisson para predecir goles esperados (xG)."""
    
    def __init__(self):
        self.model = None

    def train(self, df_train: pd.DataFrame) -> None:
        """
        Requiere un DataFrame con: Team, Opponent, Goals, Venue (Home/Away).
        Aplica un GLM (Generalized Linear Model) de la familia Poisson.
        """
        logger.info("Entrenando Modelo de Poisson Bivariado...")
        
        # Concatenar datos de local y visitante para estructurar el panel
        goal_model_data = pd.concat([
            df_train[['home_team', 'away_team', 'home_goals']].assign(home=1).rename(
                columns={'home_team': 'team', 'away_team': 'opponent', 'home_goals': 'goals'}),
            df_train[['away_team', 'home_team', 'away_goals']].assign(home=0).rename(
                columns={'away_team': 'team', 'home_team': 'opponent', 'away_goals': 'goals'})
        ])
        
        self.model = smf.glm(
            formula="goals ~ home + team + opponent",
            data=goal_model_data,
            family=sm.families.Poisson()
        ).fit()
        logger.info("Modelo de Poisson entrenado correctamente.")

    def predict_proba(self, home_team: str, away_team: str) -> Dict[str, float]:
        """Calcula probabilidades 1X2 construyendo la matriz bivariada de goles."""
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado.")
            
        home_xg = self.model.predict(pd.DataFrame(data={'team': home_team, 'opponent': away_team, 'home': 1})).values[0]
        away_xg = self.model.predict(pd.DataFrame(data={'team': away_team, 'opponent': home_team, 'home': 0})).values[0]
        
        # Limitar matriz a 5 goles para optimizar cálculo
        max_goals = 5
        prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
        
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob_matrix[i][j] = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
                
        # 1X2 Sumas Diagonales y Triangulares
        prob_home = np.sum(np.tril(prob_matrix, -1))
        prob_draw = np.sum(np.diag(prob_matrix))
        prob_away = np.sum(np.triu(prob_matrix, 1))
        
        # Normalizar para evitar sesgo de truncamiento
        total = prob_home + prob_draw + prob_away
        
        return {
            "1": prob_home / total,
            "X": prob_draw / total,
            "2": prob_away / total
        }
