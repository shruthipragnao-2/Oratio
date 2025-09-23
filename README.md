Oratio Backend - AI Bias Detection API
======================================

An AI-powered backend that analyzes text to detect biased sentences and suggests neutral, fair alternatives, helping users create more inclusive and responsible content.

Tech stack
----------
- FastAPI (Python)
- spaCy
- PyTorch
- TensorFlow

Project layout
--------------

```
oratio_backend/
  app/
    __init__.py
    main.py
    api/
      __init__.py
      routes.py
    core/
      __init__.py
      config.py
    services/
      __init__.py
      bias_detector.py
    models/
      __init__.py
      tf_model.py
      torch_model.py
  tests/
    __init__.py
    test_health.py
requirements.txt
uvicorn.ini
```

Quickstart
----------

1) Create and activate a virtual environment

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: . .venv/Scripts/Activate.ps1
```

2) Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

3) Run the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --use-colors
```

API
---

- GET /health: Liveness check
- POST /analyze: Analyze text and return biased spans and neutral rewrites

Example request
---------------

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"The crazy old man couldn't understand simple tech."}'
```

Example response
----------------

```json
{
  "original_text": "The crazy old man couldn't understand simple tech.",
  "summary": {
    "biased_count": 1,
    "score": 0.61
  },
  "sentences": [
    {
      "sentence": "The crazy old man couldn't understand simple tech.",
      "biased_spans": [
        {"text": "crazy", "start": 4, "end": 9, "type": "ableist"}
      ],
      "suggestion": "The man had difficulty with the technology."
    }
  ]
}
```

Notes
-----
- This project includes lightweight placeholder models using both PyTorch and TensorFlow to demonstrate integration. You can swap in trained models later.
- spaCy is used for sentence splitting, tokenization, and simple lexical heuristics.
