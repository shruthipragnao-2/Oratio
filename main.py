"""
Oratio Backend - AI Bias Detection API
Minimal FastAPI backend with PyTorch and Hugging Face integration
"""

import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import secrets
from passlib.context import CryptContext

# Configuration
from config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES, ALLOWED_ORIGINS, HF_MODEL_NAME, DATABASE_URL, BIAS_MODELS

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Global model variables - Multiple models for comprehensive bias detection
bias_models = {}
model_load_status = {}

# Database Models
class UserModel(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(BaseModel):
    id: int
    email: str

class AnalyzeRequest(BaseModel):
    text: str

class BiasedSpan(BaseModel):
    text: str
    start: int
    end: int
    type: str

class SentenceAnalysis(BaseModel):
    sentence: str
    biased_spans: List[BiasedSpan]
    suggestion: str

class AnalyzeResponse(BaseModel):
    original_text: str
    summary: Dict[str, Any]
    sentences: List[SentenceAnalysis]

def init_database():
    """Initialize MySQL database with users table"""
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
    except Exception as e:
        print(f"❌ Error creating database tables: {e}")
        raise

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def create_access_token(data: dict) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = secrets.token_urlsafe(32)  # Simple token for demo
    return encoded_jwt

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email from database"""
    user = db.query(UserModel).filter(UserModel.email == email).first()
    if user:
        return User(id=user.id, email=user.email)
    return None

def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticate user with email and password"""
    user = db.query(UserModel).filter(UserModel.email == email).first()
    if user and verify_password(password, user.hashed_password):
        return User(id=user.id, email=user.email)
    return None

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), 
                    db: Session = Depends(get_db)) -> User:
    """Get current authenticated user"""
    # Simple token validation for demo - in production use proper JWT
    if not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # For demo purposes, we'll just return a mock user
    # In production, decode JWT and get user from database
    return User(id=1, email="demo@example.com")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize multiple bias detection models on startup"""
    global bias_models, model_load_status
    
    print("Loading multiple bias detection models...")
    print("=" * 50)
    
    # Load multiple specialized models
    for model_type, model_name in BIAS_MODELS.items():
        try:
            print(f"Loading {model_type} model: {model_name}")
            
            if model_type == "gender_bias":
                # Special handling for bias identification model
                bias_models[model_type] = pipeline(
                    "text-classification",
                    model=model_name,
                    tokenizer=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
            else:
                # Standard text classification models
                bias_models[model_type] = pipeline(
                    "text-classification",
                    model=model_name,
                    tokenizer=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
            
            model_load_status[model_type] = True
            print(f"✅ {model_type} model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading {model_type} model: {e}")
            model_load_status[model_type] = False
            bias_models[model_type] = None
    
    # Fallback: Load single model if all others fail
    if not any(model_load_status.values()):
        print("\nFalling back to single model...")
        try:
            bias_models["fallback"] = pipeline(
                "text-classification",
                model=HF_MODEL_NAME,
                tokenizer=HF_MODEL_NAME,
                device=0 if torch.cuda.is_available() else -1
            )
            model_load_status["fallback"] = True
            print(f"✅ Fallback model loaded: {HF_MODEL_NAME}")
        except Exception as e:
            print(f"❌ Fallback model failed: {e}")
            bias_models["fallback"] = None
            model_load_status["fallback"] = False
    
    print("=" * 50)
    print("Model loading complete!")
    
    yield
    
    # Cleanup on shutdown
    print("Shutting down...")

app = FastAPI(
    title="Oratio Bias Detection API",
    description="AI-powered text bias detection and neutral rewriting",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
init_database()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.post("/auth/signup", response_model=Dict[str, str])
async def signup(user_data: UserCreate, db: Session = Depends(get_db)):
    """User registration"""
    # Check if user already exists
    existing_user = get_user_by_email(db, user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    db_user = UserModel(
        email=user_data.email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Generate access token
    access_token = create_access_token({"sub": user_data.email})
    return {"access_token": access_token}

@app.post("/auth/login", response_model=Dict[str, str])
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    """User authentication"""
    user = authenticate_user(db, user_data.email, user_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token({"sub": user.email})
    return {"access_token": access_token}

@app.get("/auth/me", response_model=User)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return current_user

def detect_bias_comprehensive(text: str) -> List[Dict[str, Any]]:
    """Comprehensive bias detection using multiple models"""
    all_bias_spans = []
    
    # Try each loaded model
    for model_type, model in bias_models.items():
        if model is None:
            continue
            
        try:
            result = model(text)
            
            # Process results based on model type
            if model_type == "toxicity":
                # Toxicity model results
                if result[0]['label'] in ['toxic', 'severe_toxic', 'threat', 'insult', 'identity_hate'] and result[0]['score'] > 0.5:
                    all_bias_spans.append({
                        "text": text,
                        "start": 0,
                        "end": len(text),
                        "type": f"toxic_{result[0]['label']}",
                        "confidence": result[0]['score'],
                        "model": model_type
                    })
                    
            elif model_type == "hate_speech":
                # Hate speech model results
                if result[0]['label'] in ['LABEL_1', 'hate', 'offensive'] and result[0]['score'] > 0.5:
                    all_bias_spans.append({
                        "text": text,
                        "start": 0,
                        "end": len(text),
                        "type": "hate_speech",
                        "confidence": result[0]['score'],
                        "model": model_type
                    })
                    
            elif model_type == "gender_bias":
                # Gender bias model results
                if result[0]['score'] > 0.5:
                    bias_type = result[0]['label'].lower()
                    all_bias_spans.append({
                        "text": text,
                        "start": 0,
                        "end": len(text),
                        "type": f"gender_{bias_type}",
                        "confidence": result[0]['score'],
                        "model": model_type
                    })
                    
            elif model_type == "sentiment":
                # Sentiment model for detecting negative bias
                if result[0]['label'] in ['NEGATIVE', 'LABEL_0'] and result[0]['score'] > 0.7:
                    all_bias_spans.append({
                        "text": text,
                        "start": 0,
                        "end": len(text),
                        "type": "negative_sentiment",
                        "confidence": result[0]['score'],
                        "model": model_type
                    })
                    
        except Exception as e:
            print(f"Error with {model_type} model: {e}")
            continue
    
    # If no models detected bias, try rule-based detection
    if not all_bias_spans:
        all_bias_spans = detect_bias_simple(text)
    
    return all_bias_spans

def detect_bias_simple(text: str) -> List[Dict[str, Any]]:
    """Simple rule-based bias detection as fallback"""
    biased_terms = {
        # Gender bias - more comprehensive patterns
        'women cant cook': 'gender_bias',
        'women can\'t cook': 'gender_bias',
        'women dont cook': 'gender_bias',
        'women don\'t cook': 'gender_bias',
        'dont think women can cook': 'gender_bias',
        'don\'t think women can cook': 'gender_bias',
        'men are better': 'gender_bias',
        'girls are bad at': 'gender_bias',
        'boys are better': 'gender_bias',
        'women are bad at': 'gender_bias',
        'men cant': 'gender_bias',
        'men can\'t': 'gender_bias',
        
        # Ableist terms
        'crazy': 'ableist',
        'insane': 'ableist', 
        'stupid': 'ableist',
        'dumb': 'ableist',
        'lame': 'ableist',
        'retarded': 'ableist',
        'idiot': 'ableist',
        'moron': 'ableist',
        
        # Homophobic terms
        'gay': 'homophobic',
        'fag': 'homophobic',
        
        # Sexist terms
        'bitch': 'sexist',
        'whore': 'sexist',
        'slut': 'sexist',
        
        # Racist terms
        'nigger': 'racist',
        'chink': 'racist',
        'spic': 'racist',
        'kike': 'racist',
        
        # Ageist terms
        'old man': 'ageist',
        'old woman': 'ageist',
        'old people': 'ageist',
        'young and dumb': 'ageist',
        'boomer': 'ageist',
        'old people can\'t': 'ageist',
        'old people cant': 'ageist',
        
        # Negative/toxic patterns
        'you\'re an': 'toxic',
        'you are an': 'toxic',
        'this is terrible': 'negative_sentiment',
        'this is awful': 'negative_sentiment',
        'this is horrible': 'negative_sentiment'
    }
    
    spans = []
    text_lower = text.lower()
    
    for term, bias_type in biased_terms.items():
        if term in text_lower:
            start = text_lower.find(term)
            end = start + len(term)
            spans.append({
                "text": text[start:end],
                "start": start,
                "end": end,
                "type": bias_type,
                "confidence": 1.0,
                "model": "rule_based"
            })
    
    return spans

def generate_suggestion(sentence: str, biased_spans: List[Dict[str, Any]]) -> str:
    """Generate neutral suggestion for biased text"""
    if not biased_spans:
        return sentence
    
    # More comprehensive replacement rules
    replacements = {
        # Gender bias
        'women cant cook': 'people have different cooking abilities',
        'women can\'t cook': 'people have different cooking abilities',
        'women dont cook': 'people have different cooking preferences',
        'women don\'t cook': 'people have different cooking preferences',
        'dont think women can cook': 'people have different cooking abilities',
        'don\'t think women can cook': 'people have different cooking abilities',
        'men are better': 'people have different strengths',
        'girls are bad at': 'people have different abilities in',
        'boys are better': 'people have different strengths',
        'women are bad at': 'people have different abilities in',
        'men cant': 'people may have difficulty with',
        'men can\'t': 'people may have difficulty with',
        
        # Ableist terms
        'crazy': 'unusual',
        'insane': 'remarkable', 
        'stupid': 'unwise',
        'dumb': 'unwise',
        'lame': 'unfortunate',
        'retarded': 'inappropriate',
        'idiot': 'person',
        'moron': 'person',
        
        # Ageist terms
        'old man': 'person',
        'old woman': 'person',
        'young and dumb': 'inexperienced',
        'old people': 'people',
        'boomer': 'person',
        'old people can\'t': 'people may have difficulty with',
        'old people cant': 'people may have difficulty with',
        
        # Homophobic terms
        'gay': 'person',
        'fag': 'person',
        
        # Sexist terms
        'bitch': 'person',
        'whore': 'person',
        'slut': 'person',
        
        # Toxic patterns
        'you\'re an': 'you are',
        'you are an': 'you are',
        'this is terrible': 'this is challenging',
        'this is awful': 'this is challenging',
        'this is horrible': 'this is challenging',
        
        # General bias patterns
        'cant': 'may have difficulty with',
        'can\'t': 'may have difficulty with'
    }
    
    suggestion = sentence
    for span in biased_spans:
        original = span['text']
        bias_type = span.get('type', 'toxic')
        
        # Try exact match first
        if original.lower() in replacements:
            replacement = replacements[original.lower()]
        else:
            # Try partial matches for common patterns
            replacement = None
            for pattern, replacement_text in replacements.items():
                if pattern in original.lower():
                    replacement = replacement_text
                    break
            
            # Fallback based on bias type
            if not replacement:
                if 'gender' in bias_type:
                    replacement = 'gender-neutral language'
                elif bias_type == 'ageist':
                    replacement = 'age-neutral language'
                elif bias_type == 'ableist':
                    replacement = 'ability-neutral language'
                elif bias_type == 'racist':
                    replacement = 'race-neutral language'
                elif bias_type == 'homophobic':
                    replacement = 'inclusive language'
                elif bias_type == 'sexist':
                    replacement = 'gender-neutral language'
                elif 'toxic' in bias_type:
                    replacement = 'more appropriate language'
                elif bias_type == 'hate_speech':
                    replacement = 'respectful language'
                elif bias_type == 'negative_sentiment':
                    replacement = 'more positive language'
                else:
                    replacement = 'more inclusive language'
        
        suggestion = suggestion.replace(original, replacement)
    
    return suggestion

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest, current_user: User = Depends(get_current_user)):
    """Analyze text for bias and provide neutral alternatives"""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        raise HTTPException(status_code=400, detail="No valid sentences found")
    
    sentence_analyses = []
    total_biased_count = 0
    
    for sentence in sentences:
        if not sentence:
            continue
            
        # Use comprehensive bias detection with multiple models
        biased_spans = detect_bias_comprehensive(sentence)
        
        # Generate suggestion
        suggestion = generate_suggestion(sentence, biased_spans)
        
        # Convert to BiasedSpan objects
        biased_span_objects = [
            BiasedSpan(**span) for span in biased_spans
        ]
        
        sentence_analyses.append(SentenceAnalysis(
            sentence=sentence,
            biased_spans=biased_span_objects,
            suggestion=suggestion
        ))
        
        total_biased_count += len(biased_spans)
    
    # Calculate bias score
    bias_score = min(total_biased_count / len(sentences), 1.0) if sentences else 0.0
    
    return AnalyzeResponse(
        original_text=text,
        summary={
            "biased_count": total_biased_count,
            "score": round(bias_score, 2)
        },
        sentences=sentence_analyses
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
