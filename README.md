# 🚀 ExoHabitAI Backend

Flask REST API for exoplanet habitability prediction using calibrated Logistic Regression model with scientific feature engineering.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Overview

This API provides machine learning-powered predictions for exoplanet habitability based on orbital, physical, and stellar characteristics. The model achieves **87% cross-validated recall** and uses **Platt scaling** for calibrated probability estimates.

### Key Features

- ✅ **Single planet prediction** via `/predict` endpoint
- ✅ **Pre-computed rankings** via `/rank` endpoint  
- ✅ **Batch processing** up to 100 planets via `/batch_predict`
- ✅ **Comprehensive validation** with physical constraints
- ✅ **Scientific categorization** (High/Moderate/Low priority)
- ✅ **Confidence metrics** based on prediction certainty

---

## 🏗️ Architecture

```
backend/
├── app.py                    # Flask application & routes
├── utils.py                  # Validation & feature engineering
├── generate_ranking.py       # Pre-compute habitability rankings
├── requirements.txt          # Python dependencies
├── models/
│   └── final_model_scientific.pkl    # Trained model (included)
└── data/
    └── processed/
        └── habitability_ranking.csv  # Pre-computed rankings
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/CrownDestro/ExoHabitAI-Backend.git
   cd ExoHabitAI-Backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server**:
   ```bash
   python app.py
   ```

The API will be available at `http://localhost:5000`

---

## 📡 API Endpoints

### 1. Health Check
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "ranking_loaded": true
}
```

---

### 2. Single Prediction
```http
POST /predict
Content-Type: application/json
```

**Request Body**:
```json
{
  "planet_name": "Kepler-442b",
  "pl_orbper": 112.3,
  "pl_orbsmax": 0.409,
  "pl_bmasse": 2.34,
  "st_met": 0.0,
  "st_logg": 4.48,
  "disc_year": 2015,
  "st_type": "K",
  "pl_type": "super_earth"
}
```

**Response**:
```json
{
  "status": "success",
  "planet_name": "Kepler-442b",
  "habitability_prediction": {
    "is_habitable": true,
    "probability": 0.8234,
    "score": 82.34,
    "category": "High Priority",
    "description": "Strong habitability candidate - recommended for immediate spectroscopic follow-up"
  },
  "confidence": {
    "level": "High",
    "explanation": "Model is 64.7% confident in this classification"
  },
  "recommendation": {
    "observe": true,
    "priority_rank": "Top 10%",
    "suggestion": "Immediate spectroscopic analysis recommended"
  }
}
```

---

### 3. Get Rankings
```http
GET /rank?top=20&threshold=0.5
```

**Query Parameters**:
- `top` (optional): Number of results (default: 10, max: 100)
- `threshold` (optional): Minimum probability (default: 0.0, range: 0.0-1.0)

**Response**:
```json
{
  "status": "success",
  "count": 20,
  "threshold": 0.5,
  "candidates": [
    {
      "rank": 1,
      "planet_name": "TOI-700 d",
      "habitability_probability": 0.8456,
      "predicted_habitable": true,
      "disc_year": 2020
    },
    ...
  ]
}
```

---

### 4. Batch Prediction
```http
POST /batch_predict
Content-Type: application/json
```

**Request Body** (max 100 planets):
```json
{
  "planets": [
    {
      "planet_name": "Planet-1",
      "pl_orbper": 365.25,
      ...
    },
    {
      "planet_name": "Planet-2",
      "pl_orbper": 687.0,
      ...
    }
  ]
}
```

**Response**:
```json
{
  "status": "success",
  "total": 2,
  "successful": 2,
  "failed": 0,
  "results": [...]
}
```

---

### 5. Example Payloads
```http
GET /examples
```

Returns valid example inputs for testing.

---

## 📊 Input Parameters

### Required Fields

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `planet_name` | string | - | Custom identifier |
| `pl_orbper` | float | 0.1 - 100,000 | Orbital period (days) |
| `pl_orbsmax` | float | 0.001 - 1,000 | Semi-major axis (AU) |
| `pl_bmasse` | float | 0.01 - 13,000 | Planet mass (Earth masses) |
| `st_met` | float | -3.0 - 1.0 | Stellar metallicity [Fe/H] |
| `st_logg` | float | 0.0 - 6.0 | Surface gravity (log g) |
| `disc_year` | integer | 1990 - 2030 | Discovery year |
| `st_type` | string | F, G, K, M, Other | Stellar spectral type |
| `pl_type` | string | rocky, super_earth, neptune, jupiter | Planet classification |

---

## 🔬 Model Details

### Algorithm
- **Classifier**: Logistic Regression with Platt scaling
- **Features**: 15 total (6 numerical + 9 categorical one-hot encoded)
- **Training**: Star system-aware cross-validation (prevents data leakage)

### Performance Metrics
- **Recall**: 87.0% (cross-validated)
- **Recall@10**: 42.9% (3/7 habitable planets in top 10)
- **Recall@50**: 85.7% (6/7 habitable planets in top 50)
- **Average Precision**: 0.467

### Feature Engineering
Removed all direct habitability indicators to force the model to learn from indirect correlations:
- Equilibrium temperature (direct indicator)
- Habitable zone flags (derived feature)
- Earth similarity index (composite score)

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file (optional):
```bash
FLASK_ENV=production
PORT=5000
HOST=0.0.0.0
```

### CORS Configuration

The API allows cross-origin requests from all domains. For production, update `app.py`:

```python
CORS(app, resources={
    r"/*": {
        "origins": ["https://your-frontend-domain.com"]
    }
})
```

---

## 🧪 Testing

### Using cURL

**Health check**:
```bash
curl http://localhost:5000/health
```

**Single prediction**:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "planet_name": "Test-1",
    "pl_orbper": 365.25,
    "pl_orbsmax": 1.0,
    "pl_bmasse": 1.0,
    "st_met": 0.0,
    "st_logg": 4.5,
    "disc_year": 2020,
    "st_type": "G",
    "pl_type": "rocky"
  }'
```

**Get rankings**:
```bash
curl "http://localhost:5000/rank?top=10&threshold=0.7"
```

### Using Python

```python
import requests

# Single prediction
response = requests.post('http://localhost:5000/predict', json={
    "planet_name": "Kepler-442b",
    "pl_orbper": 112.3,
    "pl_orbsmax": 0.409,
    "pl_bmasse": 2.34,
    "st_met": 0.0,
    "st_logg": 4.48,
    "disc_year": 2015,
    "st_type": "K",
    "pl_type": "super_earth"
})

print(response.json())
```

---

## 🚨 Error Handling

### Error Codes

| Code | Meaning | Example |
|------|---------|---------|
| 400 | Bad Request | Invalid input parameters |
| 404 | Not Found | Endpoint doesn't exist |
| 500 | Internal Server Error | Model inference failure |

### Error Response Format

```json
{
  "status": "error",
  "message": "pl_orbper must be between 0.1 and 100000.0",
  "error_type": "ValidationError"
}
```

---

## 📦 Deployment

### Production Checklist

- [ ] Set `FLASK_ENV=production`
- [ ] Disable debug mode: `app.run(debug=False)`
- [ ] Configure CORS for specific domain
- [ ] Use production WSGI server (Gunicorn/uWSGI)
- [ ] Set up HTTPS
- [ ] Monitor with logging

### Deploy to Render

1. Connect GitHub repository to Render
2. Configure build command: `pip install -r requirements.txt`
3. Configure start command: `python app.py`
4. Set environment variables
5. Deploy!

---

## 📚 Scientific Background

### Habitability Criteria

The model is trained on simplified habitable zone criteria based on:
- Orbital configuration (period, semi-major axis)
- Planetary characteristics (mass, type)
- Stellar properties (metallicity, surface gravity, spectral type)

**Note**: Predictions are based on statistical patterns, not confirmed observations of liquid water or biological activity.

### Limitations

- **Class imbalance**: Only 0.67% of planets are labeled habitable
- **Simplified criteria**: Based on temperature/orbital zones, not atmospheric composition
- **Data constraints**: Training data limited to discovered exoplanets with complete measurements

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👨‍💻 Author

**Satya Sri Dheeraj M**

---

## 🔗 Related Projects

- [ExoHabitAI Frontend](https://github.com/CrownDestro/ExoHabitAI-Frontend) - Interactive 3D visualization dashboard

---

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Email: [motupallidheeraj@gmail.com]

---

Built with ❤️ for exoplanet science
