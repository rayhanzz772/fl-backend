import os
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import FederatedModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize model
federated_model = FederatedModel(n_features=7)

# Training state
training_state = {
    'current_round': 0,
    'rounds': [],
    'start_time': None,
    'end_time': None,
    'clients': {}
}

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_ready': federated_model.model is not None,
        'current_round': training_state['current_round'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/global_model', methods=['GET'])
def get_global_model():
    """Get current global model parameters"""
    try:
        params = federated_model.get_parameters()
        return jsonify({
            'success': True,
            'parameters': params,
            'round': training_state['current_round'],
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting global model: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/aggregate', methods=['POST'])
def aggregate():
    """Aggregate client updates"""
    try:
        data = request.json
        client_params = data.get('client_params', [])
        client_samples = data.get('client_samples', [])
        client_names = data.get('client_names', [])
        client_accuracies = data.get('client_accuracies', [])
        
        if not client_params:
            return jsonify({'success': False, 'error': 'No client parameters'}), 400
        
        # Perform FedAvg aggregation
        new_global_params = federated_model.aggregate_fedavg(client_params, client_samples)
        
        # Update training state
        training_state['current_round'] += 1
        round_info = {
            'round': training_state['current_round'],
            'clients': client_names,
            'total_samples': sum(client_samples),
            'client_accuracies': client_accuracies,
            'timestamp': datetime.now().isoformat()
        }
        training_state['rounds'].append(round_info)
        
        # Update client info
        for name, samples, acc in zip(client_names, client_samples, client_accuracies):
            if name not in training_state['clients']:
                training_state['clients'][name] = {'samples': samples, 'accuracies': []}
            training_state['clients'][name]['accuracies'].append(acc)
        
        # Save model checkpoint
        federated_model.save(f'/app/models/global_model_round_{training_state["current_round"]}.pkl')
        
        logger.info(f"Round {training_state['current_round']} completed. Clients: {client_names}")
        
        return jsonify({
            'success': True,
            'global_parameters': new_global_params,
            'round': training_state['current_round'],
            'message': f'Aggregated {len(client_params)} clients'
        })
        
    except Exception as e:
        logger.error(f"Error in aggregation: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/training_status', methods=['GET'])
def training_status():
    """Get training status"""
    return jsonify({
        'success': True,
        'state': training_state,
        'current_round': training_state['current_round']
    })

@app.route('/api/reset_training', methods=['POST'])
def reset_training():
    """Reset training state"""
    global training_state
    training_state = {
        'current_round': 0,
        'rounds': [],
        'start_time': None,
        'end_time': None,
        'clients': {}
    }
    return jsonify({'success': True, 'message': 'Training state reset'})

@app.route('/api/export_model', methods=['GET'])
def export_model():
    """Export final model"""
    try:
        model_path = '/app/models/final_model.pkl'
        federated_model.save(model_path)
        return jsonify({
            'success': True,
            'model_path': model_path,
            'rounds': training_state['current_round']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)