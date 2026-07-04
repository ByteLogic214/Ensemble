import pandas as pd
import numpy as np
from typing import List
from logs.logger import get_logger

logger = get_logger(__name__)

class MatchDataPreprocessor:
    """Pipeline para generar features de Machine Learning desde datos históricos."""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.features = [
            'elo_diff', 'xg_diff_rolling', 'points_per_game_diff',
            'asian_handicap_line', 'rest_days_diff'
        ]

    def _calculate_rolling_stats(self, df: pd.DataFrame, team_col: str) -> pd.DataFrame:
        """Calcula medias móviles (ej. últimos 5 partidos) para un equipo."""
        df = df.sort_values(by=['date'])
        
        # Goles Esperados (xG) a favor y en contra
        df['xg_for_rolling'] = df.groupby(team_col)['xg_for'].transform(
            lambda x: x.rolling(self.window_size, min_periods=1).mean().shift()
        )
        df['xg_against_rolling'] = df.groupby(team_col)['xg_against'].transform(
            lambda x: x.rolling(self.window_size, min_periods=1).mean().shift()
        )
        return df

    def transform(self, df_matches: pd.DataFrame) -> pd.DataFrame:
        """
        Ejecuta el pipeline completo de transformaciones.
        df_matches debe contener: date, home_team, away_team, home_xg, away_xg, ah_line, etc.
        """
        logger.info(f"Iniciando preprocesamiento para {len(df_matches)} partidos...")
        
        # 1. Diferencia de descanso (Rest Days)
        df_matches['date'] = pd.to_datetime(df_matches['date'])
        # (Lógica de cálculo de días de descanso omitida por brevedad, requiere panel iterativo)
        df_matches['rest_days_diff'] = 0 # Placeholder para la diferencia de días
        
        # 2. Medias Móviles de Rendimiento
        df_matches = self._calculate_rolling_stats(df_matches, 'home_team')
        
        # 3. Diferenciales (Feature clave para el modelo)
        df_matches['xg_diff_rolling'] = df_matches['xg_for_rolling'] - df_matches['xg_against_rolling']
        
        # 4. Líneas de Mercado (Si se tiene el histórico de la línea principal de Asian Handicap)
        # Ayuda al modelo a entender la expectativa del mercado vs el rendimiento real
        if 'ah_line_home' in df_matches.columns:
            df_matches['asian_handicap_line'] = df_matches['ah_line_home']
        else:
            df_matches['asian_handicap_line'] = 0.0
            
        # Limpiar NaNs producto del rolling window
        df_clean = df_matches.dropna(subset=self.features)
        logger.info(f"Preprocesamiento finalizado. Shape resultante: {df_clean.shape}")
        
        return df_clean
