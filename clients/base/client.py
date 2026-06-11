import os
import sys
import logging
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Add shared modules
sys.path.append('/app')
from shared.data_processor import StuntingDataProcessor
from shared.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class FederatedLearningClient:
    def __init__(self, client_name: str, data_path: str):
        self.client_name = client_name
        self.data_path = data_path
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=config.RANDOM_STATE,
            class_weight='balanced'
        )
        self.processor = StuntingDataProcessor()
        self.X = None
        self.y = None
        self.num_samples = 0
        self.is_trained = False
        
        # Load dan proses data LOKAL (tidak dicampur dengan desa lain!)
        self.load_and_preprocess()
    
    def load_and_preprocess(self):
        """
        Load data dari CSV lokal client
        Data diproses sendiri, TIDAK dicampur dengan client lain
        Ini adalah esensi Federated Learning!
        """
        logger.info(f"="*50)
        logger.info(f"LOADING LOCAL DATA FOR {self.client_name.upper()}")
        logger.info(f"="*50)
        
        if not os.path.exists(self.data_path):
            logger.error(f"Data file not found: {self.data_path}")
            logger.warning(f"Using dummy data for {self.client_name}")
            # Generate dummy data (only for testing)
            np.random.seed(hash(self.client_name) % 2**32)
            self.X = np.random.rand(50, 10)
            self.y = np.random.randint(0, 2, 50)
            self.num_samples = 50
            return
        
        # Load raw CSV
        df = pd.read_csv(self.data_path, sep=";", encoding="latin1")
        logger.info(f"Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Process data LOKAL (no merge with other clients!)
        self.X, self.y = self.processor.process(df, fit=True)
        self.num_samples = len(self.X)
        
        logger.info(f"Preprocessed data: {self.X.shape}")
        logger.info(f"Stunting rate: {self.y.mean()*100:.1f}%")
        logger.info(f"✅ Local data ready for {self.client_name}")
    
    def get_parameters(self):
        """Get current model parameters"""
        if not self.is_trained:
            # Initialize model with local data
            self.model.fit(self.X[:min(10, len(self.X))], self.y[:min(10, len(self.y))])
            self.is_trained = True
        
        return {
            'coef_': self.model.coef_.tolist(),
            'intercept_': self.model.intercept_.tolist()
        }
    
    def set_parameters(self, params):
        """Set model parameters from server"""
        try:
            self.model.coef_ = np.array(params['coef_'])
            self.model.intercept_ = np.array(params['intercept_'])
            self.is_trained = True
            logger.info(f"Global model received for {self.client_name}")
        except Exception as e:
            logger.error(f"Error setting parameters: {e}")
    
    def train_local(self, epochs=3):
        """
        Train model dengan data LOKAL
        Data tidak pernah离开 client!
        """
        history = []
        
        # Ensure model initialized
        if not self.is_trained:
            self.model.fit(self.X[:min(10, len(self.X))], self.y[:min(10, len(self.y))])
            self.is_trained = True
        
        for epoch in range(epochs):
            # Local training with LOCAL data only!
            self.model.fit(self.X, self.y)
            acc = self.model.score(self.X, self.y)
            history.append(acc)
            logger.debug(f"Epoch {epoch+1}: acc={acc:.3f}")
        
        logger.info(f"Local training completed for {self.client_name}")
        return self.get_parameters(), history
    
    def evaluate(self):
        """Evaluate local model on local data"""
        if not self.is_trained:
            return 0.0
        return self.model.score(self.X, self.y)

# Initialize client
CLIENT_NAME = os.getenv('CLIENT_NAME', 'client')
DATA_PATH = f'/app/data/{CLIENT_NAME}.csv'
LOCAL_EPOCHS = getattr(config, 'LOCAL_EPOCHS', 3)

client = FederatedLearningClient(CLIENT_NAME, DATA_PATH)

# Flask endpoints
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'client': client.client_name,
        'samples': client.num_samples,
        'n_features': client.X.shape[1] if client.X is not None else 0,
        'is_trained': client.is_trained
    })

@app.route('/api/client_info', methods=['GET'])
def client_info():
    return jsonify({
        'name': client.client_name,
        'num_samples': client.num_samples,
        'stunting_rate': float(client.y.mean()) if client.y is not None else 0,
        'features': client.processor.get_feature_names()
    })

@app.route('/api/update_model', methods=['POST'])
def update_model():
    try:
        data = request.json
        global_params = data.get('global_parameters')
        
        if global_params:
            client.set_parameters(global_params)
        
        local_params, history = client.train_local(epochs=LOCAL_EPOCHS)
        local_accuracy = client.evaluate()
        
        logger.info(f"{client.client_name} - Accuracy: {local_accuracy:.3f}")
        
        return jsonify({
            'success': True,
            'client_name': client.client_name,
            'parameters': local_params,
            'num_samples': client.num_samples,
            'accuracy': local_accuracy,
            'training_history': history
        })
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/evaluate', methods=['GET'])
def evaluate():
    accuracy = client.evaluate()
    return jsonify({
        'client_name': client.client_name,
        'accuracy': accuracy,
        'num_samples': client.num_samples
    })

if __name__ == '__main__':
    port = int(os.getenv('CLIENT_PORT', 5001))
    logger.info(f"Starting {CLIENT_NAME} client on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)