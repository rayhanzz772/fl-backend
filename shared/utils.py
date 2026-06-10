import logging
import sys
from datetime import datetime

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'logs/fl_{datetime.now().strftime("%Y%m%d")}.log')
        ]
    )
    return logging.getLogger(__name__)

def validate_data(df) -> dict:
    """Validate data quality"""
    issues = {}
    
    # Check required columns
    required = ['JK', 'BB Lahir', 'TB Lahir', 'ZS TB/U']
    missing = [col for col in required if col not in df.columns]
    if missing:
        issues['missing_columns'] = missing
    
    # Check missing values
    missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
    high_missing = {k: v for k, v in missing_pct.items() if v > 30}
    if high_missing:
        issues['high_missing'] = high_missing
    
    # Check data ranges
    if 'ZS TB/U' in df.columns:
        zs = pd.to_numeric(df['ZS TB/U'], errors='coerce')
        if zs.min() < -5 or zs.max() > 5:
            issues['extreme_zs'] = {'min': zs.min(), 'max': zs.max()}
    
    return issues