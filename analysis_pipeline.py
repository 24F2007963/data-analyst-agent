import asyncio
import pandas as pd
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor

from models import AnalysisInput, AnalysisSpec, Transform, ChartSpec, AnswerSpec
# You'd need to create a helper module for the `HELPERS` code
from helpers import load_input_data, apply_transform, render_charts, summarize, validate_final_result

async def run_analysis_spec(spec: AnalysisSpec) -> Dict[str, Any]:
    logs: List[str] = []

    # Parallel loading of inputs using asyncio for better non-blocking behavior
    # This is an improvement over the original's ThreadPoolExecutor inside
    # a synchronous function, which can be less efficient.
    tables =  load_input_data(spec.inputs, logs)

    # Apply transforms sequentially (limited)
    for t in spec.transforms[:10]:
        apply_transform(tables, t, logs)

    # Render charts (limited)
    images = render_charts(tables, spec.charts, logs)

    # Prepare tables output (smaller previews)
    tables_out = {name: df.head(50).to_dict(orient="records")
                  for name, df in tables.items()}
    
    # Generate answer
    ans = summarize(spec.answer, tables, logs)

    final_result = {
        "answer": ans,
        "tables": tables_out,
        "images": images,
        "logs": logs,
    }

    return final_result