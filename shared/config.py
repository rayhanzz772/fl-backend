# shared/config.py
import os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class FederatedConfig:
    """
    Central configuration for Federated Learning Stunting Detection
    """
    
    # ==========================================
    # DATA CONFIGURATION
    # ==========================================
    
    # Columns to drop (identitas & tidak relevan)
    DROP_COLUMNS: List[str] = field(default_factory=lambda: [
        'No', 'NIK', 'Nama', 'Nama Ortu', 'Alamat', 'Detail',
        'RW', 'RT', 'Posyandu', 'Tanggal Pengukuran', 'Cara Ukur', 'LiLA'
    ])
    
    # Redundant columns (high correlation)
    REDUNDANT_COLUMNS: List[str] = field(default_factory=lambda: [
        'BB/U', 'ZS BB/U', 'TB/U', 'Usia Saat Ukur'
    ])
    
    # Core features for stunting prediction
    CORE_FEATURES: List[str] = field(default_factory=lambda: [
        'JK',               # Jenis Kelamin (L/P)
        'BB Lahir',         # Berat Badan Lahir (kg)
        'TB Lahir',         # Tinggi Badan Lahir (cm)
        'umur_bulan',       # Age in months (calculated)
        'Berat',            # Current weight (kg)
        'Tinggi',           # Current height (cm)
        'BB/TB',            # Weight/Height ratio (categorical)
        'ZS BB/TB',         # Z-Score Weight for Height (numeric)
        'Naik Berat Badan', # Weight gain status (Y/T/N)
        'Jml Vit A'         # Vitamin A intake count
    ])
    
    # Additional features (if available)
    ADDITIONAL_FEATURES: List[str] = field(default_factory=lambda: [
        'KPSP',              # Development screening result
        'KIA',               # Maternal and child health
        'Kelas Ibu Balita',  # Mother's education class
        'MBG'                # Supplementary feeding program
    ])
    
    # Geographic features (will be encoded)
    GEOGRAPHIC_FEATURES: List[str] = field(default_factory=lambda: [
        'Prov', 'Kab/Kota', 'Kec', 'Desa/Kel'
    ])
    
    # Target column
    TARGET_COLUMN: str = 'ZS TB/U'
    
    # Stunting threshold (ZS < -2 = stunting)
    STUNTING_THRESHOLD: float = -2.0
    
    # ==========================================
    # FEDERATED LEARNING CONFIGURATION
    # ==========================================
    
    # Number of federated rounds
    NUM_ROUNDS: int = 20
    
    # Number of local training epochs per round
    LOCAL_EPOCHS: int = 3
    
    # Minimum clients required for aggregation
    MIN_CLIENTS: int = 2
    
    # ==========================================
    # MODEL CONFIGURATION
    # ==========================================
    
    # Model type: 'logistic_regression', 'random_forest', 'xgboost'
    MODEL_TYPE: str = 'logistic_regression'
    
    # Random seed for reproducibility
    RANDOM_STATE: int = 42
    
    # Logistic Regression specific
    LR_MAX_ITER: int = 1000
    LR_C: float = 1.0  # Regularization strength
    LR_SOLVER: str = 'lbfgs'
    
    # Random Forest specific (if used)
    RF_N_ESTIMATORS: int = 100
    RF_MAX_DEPTH: int = 10
    RF_MIN_SAMPLES_SPLIT: int = 5
    
    # ==========================================
    # SERVER CONFIGURATION
    # ==========================================
    
    # Server host and port
    SERVER_HOST: str = '0.0.0.0'
    SERVER_PORT: int = 5000
    
    # ==========================================
    # CLIENT CONFIGURATION
    # ==========================================
    
    # Default client port
    CLIENT_PORT: int = 5001
    
    # Client names
    CLIENT_NAMES: List[str] = field(default_factory=lambda: ['waru', 'kemiri', 'nangsri'])
    
    # ==========================================
    # DATA PROCESSING CONFIGURATION
    # ==========================================
    
    # Outlier handling method: 'iqr', 'zscore', 'none'
    OUTLIER_METHOD: str = 'iqr'
    
    # IQR multiplier for outlier detection
    IQR_MULTIPLIER: float = 1.5
    
    # Z-Score threshold for outlier detection
    ZSCORE_THRESHOLD: float = 3.0
    
    # Missing value handling: 'median', 'mean', 'mode', 'drop'
    MISSING_STRATEGY: str = 'median'
    
    # Maximum missing percentage allowed per column
    MAX_MISSING_PERCENT: float = 30.0
    
    # Scaling method: 'standard', 'minmax', 'robust'
    SCALING_METHOD: str = 'standard'
    
    # ==========================================
    # MONITORING CONFIGURATION
    # ==========================================
    
    # Enable detailed logging
    DEBUG_MODE: bool = False
    
    # Save model checkpoints
    SAVE_CHECKPOINTS: bool = True
    
    # Checkpoint interval (rounds)
    CHECKPOINT_INTERVAL: int = 5
    
    # ==========================================
    # PATH CONFIGURATION
    # ==========================================
    
    # Model save path
    MODEL_SAVE_PATH: str = '/app/models'
    
    # Log path
    LOG_PATH: str = '/app/logs'
    
    # Data path (relative to client)
    DATA_PATH: str = '/app/data'
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        
        # Validate MODEL_TYPE
        valid_models = ['logistic_regression', 'random_forest', 'xgboost']
        if self.MODEL_TYPE not in valid_models:
            raise ValueError(f"MODEL_TYPE must be one of {valid_models}")
        
        # Validate OUTLIER_METHOD
        valid_methods = ['iqr', 'zscore', 'none']
        if self.OUTLIER_METHOD not in valid_methods:
            raise ValueError(f"OUTLIER_METHOD must be one of {valid_methods}")
        
        # Validate SCALING_METHOD
        valid_scaling = ['standard', 'minmax', 'robust']
        if self.SCALING_METHOD not in valid_scaling:
            raise ValueError(f"SCALING_METHOD must be one of {valid_scaling}")
        
        # Validate numeric ranges
        if self.NUM_ROUNDS < 1:
            raise ValueError("NUM_ROUNDS must be at least 1")
        
        if self.LOCAL_EPOCHS < 1:
            raise ValueError("LOCAL_EPOCHS must be at least 1")
        
        if self.MIN_CLIENTS < 1:
            raise ValueError("MIN_CLIENTS must be at least 1")
        
        if self.STUNTING_THRESHOLD < -5 or self.STUNTING_THRESHOLD > 0:
            raise ValueError("STUNTING_THRESHOLD should be between -5 and 0")
    
    def get_all_features(self) -> List[str]:
        """Get all features combined"""
        features = self.CORE_FEATURES.copy()
        features.extend(self.ADDITIONAL_FEATURES)
        features.extend(self.GEOGRAPHIC_FEATURES)
        return list(set(features))  # Remove duplicates
    
    def get_numeric_features(self) -> List[str]:
        """Get numeric features only"""
        numeric = ['BB Lahir', 'TB Lahir', 'umur_bulan', 'Berat', 'Tinggi', 
                   'ZS BB/TB', 'Jml Vit A']
        return [f for f in numeric if f in self.get_all_features()]
    
    def get_categorical_features(self) -> List[str]:
        """Get categorical features only"""
        categorical = ['JK', 'BB/TB', 'Naik Berat Badan', 'KPSP', 'KIA', 
                       'Kelas Ibu Balita', 'MBG']
        categorical.extend(self.GEOGRAPHIC_FEATURES)
        return [f for f in categorical if f in self.get_all_features()]
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            'num_rounds': self.NUM_ROUNDS,
            'local_epochs': self.LOCAL_EPOCHS,
            'model_type': self.MODEL_TYPE,
            'features': self.get_all_features(),
            'n_features': len(self.get_all_features()),
            'stunting_threshold': self.STUNTING_THRESHOLD,
            'clients': self.CLIENT_NAMES
        }
    
    def display(self):
        """Display configuration in readable format"""
        print("="*60)
        print("FEDERATED LEARNING CONFIGURATION")
        print("="*60)
        print(f"\n📊 DATA CONFIGURATION:")
        print(f"   Core features: {len(self.CORE_FEATURES)}")
        print(f"   Additional features: {len(self.ADDITIONAL_FEATURES)}")
        print(f"   Geographic features: {len(self.GEOGRAPHIC_FEATURES)}")
        print(f"   Total features: {len(self.get_all_features())}")
        print(f"   Target: {self.TARGET_COLUMN}")
        print(f"   Stunting threshold: ZS < {self.STUNTING_THRESHOLD}")
        
        print(f"\n🤖 FEDERATED LEARNING:")
        print(f"   Total rounds: {self.NUM_ROUNDS}")
        print(f"   Local epochs: {self.LOCAL_EPOCHS}")
        print(f"   Min clients: {self.MIN_CLIENTS}")
        
        print(f"\n🧠 MODEL:")
        print(f"   Type: {self.MODEL_TYPE}")
        print(f"   Random state: {self.RANDOM_STATE}")
        
        print(f"\n🖥️ SERVERS:")
        print(f"   Server: {self.SERVER_HOST}:{self.SERVER_PORT}")
        print(f"   Clients: {', '.join(self.CLIENT_NAMES)}")
        
        print(f"\n⚙️ PROCESSING:")
        print(f"   Outlier method: {self.OUTLIER_METHOD}")
        print(f"   Missing strategy: {self.MISSING_STRATEGY}")
        print(f"   Scaling: {self.SCALING_METHOD}")
        
        print("="*60)


# ==========================================
# ENVIRONMENT VARIABLES OVERRIDE
# ==========================================

def load_config_from_env(config: FederatedConfig) -> FederatedConfig:
    """
    Override config values from environment variables
    """
    
    # Numeric configs
    if os.getenv('NUM_ROUNDS'):
        config.NUM_ROUNDS = int(os.getenv('NUM_ROUNDS'))
    
    if os.getenv('LOCAL_EPOCHS'):
        config.LOCAL_EPOCHS = int(os.getenv('LOCAL_EPOCHS'))
    
    if os.getenv('MIN_CLIENTS'):
        config.MIN_CLIENTS = int(os.getenv('MIN_CLIENTS'))
    
    if os.getenv('SERVER_PORT'):
        config.SERVER_PORT = int(os.getenv('SERVER_PORT'))
    
    if os.getenv('CLIENT_PORT'):
        config.CLIENT_PORT = int(os.getenv('CLIENT_PORT'))
    
    # String configs
    if os.getenv('MODEL_TYPE'):
        config.MODEL_TYPE = os.getenv('MODEL_TYPE')
    
    if os.getenv('OUTLIER_METHOD'):
        config.OUTLIER_METHOD = os.getenv('OUTLIER_METHOD')
    
    if os.getenv('SCALING_METHOD'):
        config.SCALING_METHOD = os.getenv('SCALING_METHOD')
    
    # Boolean configs
    if os.getenv('DEBUG_MODE'):
        config.DEBUG_MODE = os.getenv('DEBUG_MODE').lower() == 'true'
    
    if os.getenv('SAVE_CHECKPOINTS'):
        config.SAVE_CHECKPOINTS = os.getenv('SAVE_CHECKPOINTS').lower() == 'true'
    
    return config


# ==========================================
# DEFAULT CONFIG INSTANCE
# ==========================================

# Create default config
config = FederatedConfig()

# Override with environment variables
config = load_config_from_env(config)

# Display config if debug mode
if config.DEBUG_MODE:
    config.display()


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_config_summary() -> dict:
    """Get config summary for API responses"""
    return {
        'num_rounds': config.NUM_ROUNDS,
        'local_epochs': config.LOCAL_EPOCHS,
        'model_type': config.MODEL_TYPE,
        'features': config.get_all_features(),
        'n_features': len(config.get_all_features()),
        'clients': config.CLIENT_NAMES,
        'stunting_threshold': config.STUNTING_THRESHOLD
    }


def update_config(**kwargs):
    """Update config values dynamically"""
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
            print(f"Config updated: {key} = {value}")
        else:
            print(f"Warning: {key} not found in config")


# ==========================================
# TEST CONFIG (run this file directly)
# ==========================================

if __name__ == "__main__":
    # Test the configuration
    print("\n🧪 TESTING CONFIGURATION")
    print("-"*40)
    
    # Display config
    config.display()
    
    # Test getters
    print("\n📋 Feature Categories:")
    print(f"   All features: {config.get_all_features()[:5]}...")  # First 5 only
    print(f"   Numeric features: {config.get_numeric_features()}")
    print(f"   Categorical features: {config.get_categorical_features()[:5]}...")
    
    # Test to_dict
    print("\n📦 Config as dict:")
    print(f"   {config.to_dict()}")
    
    print("\n✅ Config loaded successfully!")