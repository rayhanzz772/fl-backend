from dataclasses import dataclass

@dataclass
class FederatedConfig:
    # Training parameters
    NUM_ROUNDS: int = 20
    LOCAL_EPOCHS: int = 5
    MIN_CLIENTS: int = 2
    
    # Model parameters
    RANDOM_STATE: int = 42
    MODEL_TYPE: str = 'logistic_regression'
    
    # Data parameters
    STUNTING_THRESHOLD: float = -2.0
    
    # Server
    SERVER_HOST: str = '0.0.0.0'
    SERVER_PORT: int = 5000

config = FederatedConfig()