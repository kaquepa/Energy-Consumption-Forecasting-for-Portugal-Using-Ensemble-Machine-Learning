from datetime import datetime
from typing import Optional, AnyStr
from datetime import datetime

class Formatters():
    def __init__(self):
        pass
    def format_gwh(self,value: float, decimals: int = 2) -> AnyStr:
        """Format value as GWh with specified decimals."""
        return f"{value:.{decimals}f} GWh"
    def format_percentage(self,value: float, decimals: int = 2, include_sign: bool = True) -> str:
        """Format value as percentage with optional sign."""
        sign = "+" if value > 0 and include_sign else ""
        return f"{sign}{value:.{decimals}f}%"
    def format_date(self,date_str: str, format: str = "%a, %b %d") -> str:
        """Format date string to human-readable format."""
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return date_obj.strftime(format)
        except:
            return date_str
    def get_confidence_emoji(self,level: str) -> str:
        """Get emoji for confidence level."""
        emojis = {
            'high': '✅',
            'medium': '🟡',
            'low': '🔴'
        }
        return emojis.get(level.lower(), '❓')
   
