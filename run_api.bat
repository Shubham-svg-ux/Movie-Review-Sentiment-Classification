@echo off
echo ========================================================
echo Starting FastAPI Sentiment Classification REST API Server
echo Docs will be accessible at http://127.0.0.1:8000/docs
echo ========================================================
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
pause
