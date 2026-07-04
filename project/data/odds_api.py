import requests
from typing import Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential
from config.settings import settings
from logs.logger import get_logger

logger = get_logger(__name__)

class OddsAPIClient:
    """Cliente para extraer cuotas de mercado en tiempo real desde The Odds API."""
    
    BASE_URL = "https://api.the-odds-api.com/v4/sports"
    
    def __init__(self):
        self.api_key = settings.ODDS_API_KEY

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_market_odds(self, sport: str, region: str = "eu", market: str = "h2h") -> List[Dict[str, Any]]:
        """Descarga las cuotas de las casas de apuestas para una liga específica."""
        logger.info(f"Consultando Odds API para el deporte: {sport}, mercado: {market}")
        
        url = f"{self.BASE_URL}/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": region,
            "markets": market,
            "oddsFormat": "decimal"
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        odds_data = response.json()
        logger.info(f"Se encontraron cuotas para {len(odds_data)} eventos.")
        return odds_data
