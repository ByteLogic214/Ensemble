import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss
from typing import Dict
from logs.logger import get_logger

logger = get_logger(__name__)

class QuantitativeBacktester:
    """Motor de evaluación rigurosa para modelos de apuestas."""
    
    @staticmethod
    def evaluate_brier_score(y_true: pd.Series, y_prob: pd.Series) -> float:
        """
        Calcula el Brier Score. Un valor más cercano a 0 indica mayor precisión.
        y_true debe ser un vector binario (1 si ocurrió el evento, 0 si no).
        """
        score = brier_score_loss(y_true, y_prob)
        logger.info(f"Brier Score Calculado: {score:.4f}")
        return score

    @staticmethod
    def evaluate_log_loss(y_true: pd.Series, y_prob: pd.Series) -> float:
        """Calcula la pérdida logarítmica (Log Loss/Cross-Entropy)."""
        loss = log_loss(y_true, y_prob)
        logger.info(f"Log Loss Calculado: {loss:.4f}")
        return loss

    @staticmethod
    def calculate_clv(placed_odds: pd.Series, closing_odds: pd.Series) -> pd.DataFrame:
        """
        Calcula el Closing Line Value (CLV).
        Vencer a la cuota de cierre es el indicador principal de rentabilidad a largo plazo.
        """
        df = pd.DataFrame({
            'placed_odds': placed_odds,
            'closing_odds': closing_odds
        })
        
        # Ratio simple de ventaja sobre la línea de cierre
        df['clv_ratio'] = df['placed_odds'] / df['closing_odds']
        
        # CLV porcentual
        df['clv_percent'] = (df['clv_ratio'] - 1) * 100
        
        avg_clv = df['clv_percent'].mean()
        logger.info(f"CLV Promedio del portafolio: {avg_clv:.2f}%")
        
        return df

    @staticmethod
    def run_portfolio_simulation(bets_df: pd.DataFrame, initial_bankroll: float = 1000.0) -> Dict[str, float]:
        """
        Simula el crecimiento del Bankroll utilizando el histórico de apuestas (bets_df).
        Requiere las columnas: 'odds', 'kelly_fraction', 'won' (booleano).
        """
        bankroll = initial_bankroll
        peak_bankroll = bankroll
        drawdown = 0.0
        
        for _, row in bets_df.iterrows():
            stake = bankroll * row['kelly_fraction']
            
            if row['won']:
                profit = stake * (row['odds'] - 1)
                bankroll += profit
            else:
                bankroll -= stake
                
            # Cálculo de Drawdown (Caída máxima desde el pico)
            if bankroll > peak_bankroll:
                peak_bankroll = bankroll
            
            current_drawdown = (peak_bankroll - bankroll) / peak_bankroll
            if current_drawdown > drawdown:
                drawdown = current_drawdown
                
        roi = ((bankroll - initial_bankroll) / initial_bankroll) * 100
        
        metrics = {
            "initial_bank": initial_bankroll,
            "final_bank": bankroll,
            "total_roi_%": roi,
            "max_drawdown_%": drawdown * 100
        }
        
        logger.info(f"Simulación Finalizada: ROI {roi:.2f}% | Max DD {drawdown*100:.2f}%")
        return metrics
