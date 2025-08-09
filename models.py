from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class Plan(BaseModel):
    question: str
    parameters: Dict[str, Any]
    steps: List[str]
    final_variables: List[str]


class AnalysisInput(BaseModel):
    name: str
    source: str  # "html" | "csv" | "json" | "inline"
    url: Optional[str] = None
    data: Optional[Any] = None
    table_index: Optional[int] = None

class Transform(BaseModel):
    target: str
    op: str
    args: Dict[str, Any]

class ChartSpec(BaseModel):
    table: str
    kind: str
    x: Optional[str] = None
    y: Optional[Any] = None
    title: Optional[str] = None
    bins: Optional[int] = None

class AnswerSpec(BaseModel):
    type: str  # "text_summary" | "basic_stats" | "none"
    table: Optional[str] = None
    columns: Optional[List[str]] = None

class AnalysisSpec(BaseModel):
    inputs: List[AnalysisInput]
    transforms: List[Transform]
    charts: List[ChartSpec]
    answer: AnswerSpec
    result_table: str
