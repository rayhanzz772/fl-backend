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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class FederatedLearningClient:
    def __init__(self, client_name: str, data_path: str):
        self.client_name = client_name
        self.data_path = data_path
        self.n_features = 7  # FIXED: 7 features, matching server
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
        
        self.load_and_preprocess()
    
    def load_and_preprocess(self):
        """Load data and apply preprocessing"""
        logger.info(f"Loading data for {self.client_name}")
        
        if not os.path.exists(self.data_path):
            logger.error(f"Data file not found: {self.data_path}")
            # Generate dummy data with EXACTLY 7 features
            np.random.seed(42)
            self.X = np.random.rand(50, self.n_features)  # 7 features, not 10
            self.y = np.random.randint(0, 2, 50)
            self.num_samples = 50
            logger.warning(f"Using dummy data: {self.num_samples} samples, {self.n_features} features")
            return
        
        df = pd.read_csv(self.data_path)
        logger.info(f"Raw data shape: {df.shape}")
        
        # Process data
        self.X, self.y = self.processor.process(df, fit=True)
        self.num_samples = len(self.X)
        
        # Ensure X has exactly 7 features
        if self.X.shape[1] != self.n_features:
            logger.warning(f"Processor returned {self.X.shape[1]} features, adjusting to {self.n_features}")
            if self.X.shape[1] > self.n_features:
                self.X = self.X[:, :self.n_features]  # Take first 7
            else:
                # Pad with zeros
                padded = np.zeros((self.num_samples, self.n_features))
                padded[:, :self.X.shape[1]] = self.X
                self.X = padded
        
        logger.info(f"Preprocessed data shape: {self.X.shape}")
        logger.info(f"Stunting rate: {self.y.mean()*100:.1f}%")
    
    def get_parameters(self):
        """Get current model parameters with EXACTLY 7 features"""
        if not self.is_trained:
            # Fit once to initialize parameters
            self.model.fit(self.X[:min(10, len(self.X))], self.y[:min(10, len(self.y))])
            self.is_trained = True
        
        # Ensure coef has correct shape (1,7)
        coef = self.model.coef_
        if coef.shape[1] != self.n_features:
            logger.warning(f"Coef shape {coef.shape}, reshaping to (1,{self.n_features})")
            if coef.shape[1] > self.n_features:
                coef = coef[:, :self.n_features]
            else:
                padded = np.zeros((1, self.n_features))
                padded[:, :coef.shape[1]] = coef
                coef = padded
            self.model.coef_ = coef
        
        return {
            'coef_': self.model.coef_.tolist(),
            'intercept_': self.model.intercept_.tolist()
        }
    
    def set_parameters(self, params):
        """Set model parameters from server"""
        try:
            coef = np.array(params['coef_'])
            intercept = np.array(params['intercept_'])
            
            # Ensure correct shape (1,7)
            if coef.shape[1] != self.n_features:
                logger.warning(f"Received {coef.shape[1]} features, adjusting to {self.n_features}")
                if coef.shape[1] > self.n_features:
                    coef = coef[:, :self.n_features]
                else:
                    padded = np.zeros((1, self.n_features))
                    padded[:, :coef.shape[1]] = coef
                    coef = padded
            
            self.model.coef_ = coef
            self.model.intercept_ = intercept
            self.is_trained = True
            logger.info(f"Parameters set for {self.client_name}, coef shape: {self.model.coef_.shape}")
        except Exception as e:
            logger.error(f"Error setting parameters: {e}")
    
    def train_local(self, epochs=3):
        """Perform local training"""
        history = []
        
        # Ensure model has parameters
        if not self.is_trained:
            self.model.fit(self.X[:min(10, len(self.X))], self.y[:min(10, len(self.y))])
            self.is_trained = True
        
        for epoch in range(epochs):
            self.model.fit(self.X, self.y)
            acc = self.model.score(self.X, self.y)
            history.append(acc)
            logger.debug(f"Epoch {epoch+1}/{epochs}: accuracy={acc:.3f}")
        
        # Ensure coef has correct shape after training
        if self.model.coef_.shape[1] != self.n_features:
            logger.warning(f"After training, coef shape is {self.model.coef_.shape}, fixing...")
            if self.model.coef_.shape[1] > self.n_features:
                self.model.coef_ = self.model.coef_[:, :self.n_features]
            else:
                padded = np.zeros((1, self.n_features))
                padded[:, :self.model.coef_.shape[1]] = self.model.coef_
                self.model.coef_ = padded
        
        return self.get_parameters(), history
    
    def evaluate(self):
        """Evaluate local model"""
        if not self.is_trained:
            return 0.0
        return self.model.score(self.X, self.y)

# Initialize client
CLIENT_NAME = os.getenv('CLIENT_NAME', 'client')
DATA_PATH = f'/app/data/stunting_{CLIENT_NAME}.csv'

# Get LOCAL_EPOCHS from config with fallback
LOCAL_EPOCHS = getattr(config, 'LOCAL_EPOCHS', 3)

client = FederatedLearningClient(CLIENT_NAME, DATA_PATH)

# Flask endpoints
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'client': client.client_name,
        'samples': client.num_samples,
        'n_features': client.n_features,
        'model_ready': client.model is not None,
        'is_trained': client.is_trained
    })

@app.route('/api/client_info', methods=['GET'])
def client_info():
    return jsonify({
        'name': client.client_name,
        'num_samples': client.num_samples,
        'n_features': client.n_features,
        'stunting_rate': float(client.y.mean()) if client.y is not None else 0,
        'features': client.processor.get_feature_names() if hasattr(client.processor, 'get_feature_names') else []
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
        
        logger.info(f"Client {client.client_name} - Accuracy: {local_accuracy:.3f}")
        
        return jsonify({
            'success': True,
            'client_name': client.client_name,
            'parameters': local_params,
            'num_samples': client.num_samples,
            'accuracy': local_accuracy,
            'training_history': history
        })
    except Exception as e:
        logger.error(f"Error in update_model: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/evaluate', methods=['GET'])
def evaluate():
    """Evaluate local model"""
    try:
        accuracy = client.evaluate()
        return jsonify({
            'client_name': client.client_name,
            'accuracy': accuracy,
            'num_samples': client.num_samples
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/preprocessing_info', methods=['GET'])
def preprocessing_info():
    try:
        info = client.processor.get_preprocessing_info() if hasattr(client.processor, 'get_preprocessing_info') else {}
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset_model', methods=['POST'])
def reset_model():
    """Reset local model"""
    client.model = LogisticRegression(
        max_iter=1000,
        random_state=config.RANDOM_STATE,
        class_weight='balanced'
    )
    client.is_trained = False
    logger.info(f"Model reset for {client.client_name}")
    return jsonify({'success': True, 'message': 'Model reset successfully'})

if __name__ == '__main__':
    port = int(os.getenv('CLIENT_PORT', 5001))
    logger.info(f"Starting client {CLIENT_NAME} on port {port} with {client.n_features} features")
    app.run(host='0.0.0.0', port=port, debug=False)