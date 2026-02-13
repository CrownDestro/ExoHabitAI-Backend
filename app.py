# app.py - Flask Backend API (FIXED IMPORTS FOR RENDER)
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os

# FIXED: Import from root level, not from backend module
from utils import (
    validate_input,
    prepare_features,
    format_prediction_response
)

app = Flask(__name__)
CORS(app)

# Load model and ranking data
MODEL_PATH = 'models/final_model_scientific.pkl'
RANKING_PATH = 'data/processed/habitability_ranking_final.csv'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

try:
    ranking_df = pd.read_csv(RANKING_PATH)
    print(f"✅ Ranking data loaded: {len(ranking_df)} candidates")
except Exception as e:
    print(f"❌ Error loading ranking data: {e}")
    ranking_df = None

# Root endpoint
@app.route('/')
def root():
    """API documentation endpoint"""
    return jsonify({
        'name': 'ExoHabitAI API',
        'version': '1.0.0',
        'endpoints': {
            '/health': 'GET - Health check',
            '/predict': 'POST - Single planet prediction',
            '/rank': 'GET - Get ranked candidates',
            '/batch_predict': 'POST - Batch predictions (max 100)',
            '/examples': 'GET - Example payloads'
        },
        'documentation': 'https://github.com/CrownDestro/ExoHabitAI-Backend'
    })

# Health check
@app.route('/health', methods=['GET'])
def health():
    """Check if API and model are operational"""
    return jsonify({
        'status': 'healthy' if model is not None else 'degraded',
        'model_loaded': model is not None,
        'ranking_loaded': ranking_df is not None
    })

# Single prediction
@app.route('/predict', methods=['POST'])
def predict():
    """Predict habitability for a single exoplanet"""
    try:
        data = request.get_json()
        
        # Validate input
        is_valid, error_msg = validate_input(data)
        if not is_valid:
            return jsonify({'status': 'error', 'message': error_msg}), 400
        
        # Prepare features
        features = prepare_features(data)
        
        # Make prediction
        if model is None:
            return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
            
        probability = model.predict_proba(features)[0][1]
        prediction = model.predict(features)[0]
        
        # Format response
        response = format_prediction_response(
            planet_name=data.get('planet_name', 'Unknown'),
            probability=probability,
            prediction=prediction
        )
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}'
        }), 500

# Get rankings
@app.route('/rank', methods=['GET'])
def get_ranking():
    """Retrieve pre-computed habitability rankings"""
    try:
        if ranking_df is None:
            return jsonify({'status': 'error', 'message': 'Ranking data not available'}), 500
        
        # Get query parameters
        top_n = int(request.args.get('top', 10))
        threshold = float(request.args.get('threshold', 0.0))
        
        # Validate parameters
        if top_n < 1 or top_n > 100:
            return jsonify({'status': 'error', 'message': 'top must be between 1 and 100'}), 400
        if threshold < 0.0 or threshold > 1.0:
            return jsonify({'status': 'error', 'message': 'threshold must be between 0.0 and 1.0'}), 400
        
        # Filter and get top N
        filtered = ranking_df[ranking_df['habitability_probability'] >= threshold]
        top_candidates = filtered.head(top_n)
        
        # Format response
        candidates = []
        for _, row in top_candidates.iterrows():
            candidates.append({
                'rank': int(row['rank']),
                'planet_name': row['planet_name'],
                'habitability_probability': float(row['habitability_probability']),
                'predicted_habitable': bool(row['predicted_habitable']),
                'disc_year': int(row['discovery_year']) if pd.notna(row['discovery_year']) else None
            })
        
        return jsonify({
            'status': 'success',
            'count': len(candidates),
            'threshold': threshold,
            'candidates': candidates
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to retrieve rankings: {str(e)}'
        }), 500

# Batch prediction
@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict habitability for multiple exoplanets"""
    try:
        data = request.get_json()
        planets = data.get('planets', [])
        
        if not planets:
            return jsonify({'status': 'error', 'message': 'No planets provided'}), 400
        
        if len(planets) > 100:
            return jsonify({'status': 'error', 'message': 'Maximum 100 planets per batch'}), 400
        
        results = []
        failed = []
        
        for planet in planets:
            try:
                # Validate
                is_valid, error_msg = validate_input(planet)
                if not is_valid:
                    failed.append({
                        'planet_name': planet.get('planet_name', 'Unknown'),
                        'error': error_msg
                    })
                    continue
                
                # Predict
                features = prepare_features(planet)
                probability = model.predict_proba(features)[0][1]
                prediction = model.predict(features)[0]
                
                response = format_prediction_response(
                    planet_name=planet.get('planet_name', 'Unknown'),
                    probability=probability,
                    prediction=prediction
                )
                results.append(response)
                
            except Exception as e:
                failed.append({
                    'planet_name': planet.get('planet_name', 'Unknown'),
                    'error': str(e)
                })
        
        return jsonify({
            'status': 'success',
            'total': len(planets),
            'successful': len(results),
            'failed': len(failed),
            'results': results,
            'errors': failed if failed else None
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Batch prediction failed: {str(e)}'
        }), 500

# Example payloads
@app.route('/examples', methods=['GET'])
def examples():
    """Get example input payloads"""
    return jsonify({
        'examples': [
            {
                'planet_name': 'Kepler-442b',
                'pl_orbper': 112.3,
                'pl_orbsmax': 0.409,
                'pl_bmasse': 2.34,
                'st_met': 0.0,
                'st_logg': 4.48,
                'disc_year': 2015,
                'st_type': 'K',
                'pl_type': 'super_earth'
            },
            {
                'planet_name': 'Proxima Centauri b',
                'pl_orbper': 11.2,
                'pl_orbsmax': 0.0485,
                'pl_bmasse': 1.27,
                'st_met': 0.21,
                'st_logg': 5.2,
                'disc_year': 2016,
                'st_type': 'M',
                'pl_type': 'rocky'
            }
        ]
    })

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

if __name__ == '__main__':
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get('PORT', 5000))
    # Bind to 0.0.0.0 to allow external connections
    app.run(host='0.0.0.0', port=port, debug=False)