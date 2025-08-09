from typing import Dict, List, Any

PLANNER_SYSTEM = """You are a planning engine. Output STRICT JSON that validates this schema:

{
  "type": "object",
  "properties": {
    "question": { "type": "string" },
    "parameters": { "type": "object" },
    "steps": { "type": "array", "items": { "type": "string" } },
    "final_variables": { "type": "array", "items": { "type": "string" } }
  },
  "required": ["question", "parameters", "steps", "final_variables"],
  "additionalProperties": false
}

Keep steps minimal (max 3). Return JSON only. No commentary."""



CODER_SYSTEM = """You write ONLY STRICT JSON analysis specs. Be concise and efficient.

Schema (must validate exactly):
{
  "type": "object",
  "properties": {
    "inputs": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type":"string"},
          "source": {"type":"string", "enum": ["html","csv","json","inline"]},
          "url": {"type":["string","null"]},
          "data": {"type":["string","object","array","null"]},
          "table_index": {"type":["integer","null"]}
        },
        "required": ["name","source"],
        "additionalProperties": false
      }
    },
    "transforms": {
      "type": "array",
      "items": {
        "type":"object",
        "properties": {
          "target": {"type":"string"},
          "op": {"type":"string", "enum": [
            "select_columns","rename","dropna","head","sort_values",
            "filter_query","groupby_agg","join","add_column","parse_dates"
          ]},
          "args": {"type":"object"}
        },
        "required": ["target","op","args"],
        "additionalProperties": false
      }
    },
    "charts": {
      "type":"array",
      "items":{
        "type":"object",
        "properties": {
          "table": {"type":"string"},
          "kind": {"type":"string","enum":["line","bar","scatter","hist"]},
          "x": {"type":["string","null"]},
          "y": {"type":["string","array","null"]},
          "title": {"type":["string","null"]},
          "bins": {"type":["integer","null"]}
        },
        "required": ["table","kind"],
        "additionalProperties": false
      }
    },
    "answer": {
      "type":"object",
      "properties": {
        "type": {"type":"string","enum":["text_summary","basic_stats","none"]},
        "table": {"type":["string","null"]},
        "columns": {"type":["array","null"], "items":{"type":"string"}}
      },
      "required": ["type"],
      "additionalProperties": false
    },
    "result_table": {"type":"string"}
  },
  "required": ["inputs","transforms","charts","answer","result_table"],
  "additionalProperties": false
}

Minimize transforms. Prefer simple operations. Return JSON only."""

# Simplified formatter for speed
OUTPUT_FORMATTER_SYSTEM = """ Return ONE string only. Use categorical answer only if question suggestes.
no need for answer number. make all the calculations twice to make sure the answer is correct. discard the data gien to you. Focus on the Question and analyze yourself.
For CSV: header+rows. For JSON: compact. For markdown: brief summary.
No extra formatting or explanations."""

VERIFY_FORMATTER_SYSTEM = """You are a data analyst. Analyze the given question statements and context/url provided and try to answer the questions in Return ONE string only.
No extra explanation to be given. Provide only answers and nothing else. No extra formatting or explanations.
""" 

FINAL_FORMATTER_SYSTEM = """Analyze the given Question and the data from 2 different sources. Carefully observe both the data and return Answer(s) to the questions as a single string only.
Return ONE string only. Use categorical answer only if question suggests.
no need for answer number. make all the calculations twice to make sure the answer is correct. discard the data gien to you. Focus on the Question and analyze yourself.
For CSV: header+rows. For JSON: compact. For markdown: brief summary.
No extra formatting or explanations. Return as  JSON """



def make_coder_prompt(plan: 'Plan') -> str:
    return (
        f"Question: {plan.question}\n"
        f"Plan: {plan.model_dump_json()}\n"
        "Return analysis_spec JSON. Be minimal and efficient."
    )

def make_formatter_prompt(question: str, final_result: Dict[str, Any]) -> str:
    print(f"Question: {question}\nData: {json.dumps(final_result, default=str)[:2000]}")
    return (f"Question: {question}\nData: {json.dumps(final_result, default=str)[:2000]}")