import pandas as pd
from typing import Dict, Tuple
from logs.logger import get_logger

logger = get_logger(__name__)

class EloCalculator:
    """Motor de cálculo de Elo Ratings para equipos de fútbol."""
    
    def __init__(self, base_rating: float = 1500.0, k_factor: float = 20.0):
        self.ratings: Dict[str, float] = {}
        self.base_rating = base_rating
        self.k_factor = k_factor

    def get_rating(self, team_name: str) -> float:
        return self.ratings.get(team_name, self.base_rating)

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calcula la probabilidad de victoria usando la curva logística del Elo."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, team_home: str, team_away: str, score_home: int, score_away: int) -> Tuple[float, float]:
        """Actualiza el rating de ambos equipos tras un partido."""
        rating_home = self.get_rating(team_home)
        rating_away = self.get_rating(team_away)
        
        exp_home = self._expected_score(rating_home, rating_away)
        exp_away = self._expected_score(rating_away, rating_home)
        
        # Determinar el resultado real (S)
        if score_home > score_away:
            s_home, s_away = 1.0, 0.0
        elif score_home < score_away:
            s_home, s_away = 0.0, 1.0
        else:
            s_home, s_away = 0.5, 0.5
            
        # Calcular nuevos ratings
        new_rating_home = rating_home + self.k_factor * (s_home - exp_home)
        new_rating_away = rating_away + self.k_factor * (s_away - exp_away)
        
        self.ratings[team_home] = new_rating_home
        self.ratings[team_away] = new_rating_away
        
        return new_rating_home, new_rating_away
