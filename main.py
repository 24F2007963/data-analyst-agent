import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, Response
import json
from pydantic import ValidationError

from typing import Dict, List, Any
from settings import SETTINGS
from llm_client import generate_json_response, llm_call_async
from prompts import PLANNER_SYSTEM, CODER_SYSTEM, OUTPUT_FORMATTER_SYSTEM, VERIFY_FORMATTER_SYSTEM, FINAL_FORMATTER_SYSTEM, make_coder_prompt
from models import Plan, AnalysisSpec
from helpers import  validate_final_result
from analysis_pipeline import run_analysis_spec

app = FastAPI(title="Universal Data Analyst", version="5.0.0-refactored")

def make_formatter_prompt(question: str, final_result: Dict[str, Any]) -> str:
    return f"Question: {question}\nData: {json.dumps(final_result, default=str)[:2000]}" 

@app.post("/api/")
async def analyze_question(
    file: UploadFile = File(...)
):
    question = (await file.read()).decode("utf-8").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question file")

    try:
        plan = await generate_json_response(
            PLANNER_SYSTEM, question, SETTINGS.planner_model, Plan
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Planner failed: {e}")


    try:
        spec_prompt = make_coder_prompt(plan)
        spec = await generate_json_response(
            CODER_SYSTEM, spec_prompt, SETTINGS.coder_model, AnalysisSpec
        )
        print('spec : ', spec)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Spec generation failed: {e}")


    try:
        final_structured = await run_analysis_spec(spec)
        
        final_structured = validate_final_result(final_structured)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {e}")


    try:
        print('will format now')
        fmt_prompt = make_formatter_prompt(question, final_structured)
        
        formatted = await llm_call_async(
            OUTPUT_FORMATTER_SYSTEM, fmt_prompt, SETTINGS.formatter_model
        )
    except Exception:
        formatted = f"Analysis complete. Found {len(final_structured.get('tables', {}))} tables, {len(final_structured.get('images', []))} charts."

    try:
        print('validating with gemini now')
        
        formatted_gemini = await llm_call_async(
            VERIFY_FORMATTER_SYSTEM,  f"Question statement: {question}", SETTINGS.formatter_model
        )

        finalstr = f"Question statement: {question}\nData1: {formatted} \nData2: {formatted_gemini}"
        print('finalstr:', finalstr)
        final_formatted = await llm_call_async(
            FINAL_FORMATTER_SYSTEM,  finalstr, SETTINGS.verifier_model
        )
    except Exception:
        final_formatted = f"Analysis complete. Found {len(final_structured.get('tables', {}))} tables, {len(final_structured.get('images', []))} charts."

    return Response(final_formatted, media_type="text/plain")


@app.get("/")
async def root():
    return {
        "message": "Universal Data Analyst API v5.0 (Refactored)",
        "usage": "curl -s -X POST 'http://localhost:8000/api/?debug=1' -F 'file=@question.txt'",
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)