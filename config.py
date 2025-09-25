"""
Configuration settings for Oratio Backend
"""

import os

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Database (MySQL)
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "oratio")

# Construct MySQL URL
DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"

# Model Configuration - Multiple models for comprehensive bias detection
BIAS_MODELS = {
    "toxicity": os.getenv("TOXICITY_MODEL", "unitary/toxic-bert"),
    "hate_speech": os.getenv("HATE_SPEECH_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"),
    "gender_bias": os.getenv("GENDER_BIAS_MODEL", "PriyaPatel/bias_identificaiton45"),
    "sentiment": os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")
}

# Fallback to single model if needed
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "unitary/toxic-bert")

# CORS
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173"
]
