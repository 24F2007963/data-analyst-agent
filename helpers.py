# helpers.py

import json, base64, io, time, math, statistics, re, datetime
import requests, pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

# The original HELPERS code block starts here
def fetch_text(url, timeout=8, retries=1):
    last = None
    headers = {"User-Agent": "Mozilla/5.0 (compatible; UDA/4.0)"}
    for _ in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last = e
            if _ < retries:
                time.sleep(0.5)  # Reduced sleep
    raise last

def fetch_json(url, timeout=8):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; UDA/4.0)"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()

def read_table_html(html):
    try:
        dfs = pd.read_html(html)
        return dfs
    except Exception:
        soup = BeautifulSoup(html, "lxml")
        tables = soup.find_all("table")
        if not tables:
            return []
        out = []
        for t in tables[:3]:  # Limit to first 3 tables for speed
            try:
                out.append(pd.read_html(str(t))[0])
            except Exception:
                pass
        return out

def df_to_records(df):
    return df.replace({pd.NA: None, np.nan: None}).to_dict(orient="records")

def fig_to_data_uri(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=72)  # Lower DPI for speed
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

# The original executor functions start here
from concurrent.futures import ThreadPoolExecutor, as_completed
from models import AnalysisInput, Transform, ChartSpec, AnswerSpec

ALLOWED_OPS = {
    "select_columns","rename","dropna","head","sort_values",
    "filter_query","groupby_agg","join","add_column","parse_dates"
}

def load_input_data(inputs: List[AnalysisInput], logs: List[str]) -> Dict[str, Any]:
    tables: Dict[str, Any] = {}
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_input = {}
        
        for inp in inputs:
            if inp.source == "html" and inp.url:
                future = executor.submit(fetch_text, inp.url, 8, 1)
                future_to_input[future] = inp
            elif inp.source == "csv" and inp.url:
                future = executor.submit(fetch_text, inp.url, 8, 1)
                future_to_input[future] = inp
            elif inp.source == "json" and inp.url:
                future = executor.submit(fetch_json, inp.url, 8)
                future_to_input[future] = inp
            else:
                _load_single_input(inp, tables, logs)
        
        for future in as_completed(future_to_input, timeout=15):
            inp = future_to_input[future]
            try:
                data = future.result()
                _process_fetched_data(inp, data, tables, logs)
            except Exception as e:
                logs.append(f"[inputs] {inp.name}: fetch error {e}")
    
    return tables

def _load_single_input(inp: AnalysisInput, tables: Dict[str, Any], logs: List[str]):
    from io import StringIO
    if inp.source == "csv" and isinstance(inp.data, str):
        tables[inp.name] = pd.read_csv(StringIO(inp.data))
        logs.append(f"[inputs] {inp.name}: inline csv shape={tables[inp.name].shape}")
    elif inp.source == "json" and inp.data:
        tables[inp.name] = pd.json_normalize(inp.data)
        logs.append(f"[inputs] {inp.name}: inline json shape={tables[inp.name].shape}")
    elif inp.source == "inline":
        tables[inp.name] = pd.DataFrame(inp.data or [])
        logs.append(f"[inputs] {inp.name}: inline records shape={tables[inp.name].shape}")

def _process_fetched_data(inp: AnalysisInput, data: Any, tables: Dict[str, Any], logs: List[str]):
    from io import StringIO
    if inp.source == "html":
        dfs = read_table_html(data)
        if dfs:
            idx = min(inp.table_index or 0, len(dfs) - 1)
            tables[inp.name] = dfs[idx].head(1000)
            logs.append(f"[inputs] {inp.name}: html table shape={tables[inp.name].shape}")
    elif inp.source == "csv":
        df = pd.read_csv(StringIO(data))
        tables[inp.name] = df.head(1000)
        logs.append(f"[inputs] {inp.name}: csv shape={tables[inp.name].shape}")
    elif inp.source == "json":
        df = pd.json_normalize(data)
        tables[inp.name] = df.head(1000)
        logs.append(f"[inputs] {inp.name}: json shape={tables[inp.name].shape}")

def apply_transform(df_map: Dict[str, Any], t: Transform, logs: List[str]) -> None:
    if t.op not in ALLOWED_OPS:
        logs.append(f"[transform] unsupported op {t.op}")
        return
    if t.target not in df_map:
        logs.append(f"[transform] missing target {t.target}")
        return
    
    df = df_map[t.target]
    
    try:
        if t.op == "select_columns":
            cols = t.args.get("columns", [])
            df_map[t.target] = df[cols]
        elif t.op == "rename":
            mapping = t.args.get("map", {})
            df_map[t.target] = df.rename(columns=mapping)
        elif t.op == "dropna":
            subset = t.args.get("subset", None)
            df_map[t.target] = df.dropna(subset=subset)
        elif t.op == "head":
            n = min(int(t.args.get("n", 10)), 100)
            df_map[t.target] = df.head(n)
        elif t.op == "sort_values":
            by = t.args.get("by")
            asc = bool(t.args.get("ascending", True))
            df_map[t.target] = df.sort_values(by=by, ascending=asc)
        elif t.op == "filter_query":
            q = t.args.get("query", "")
            df_map[t.target] = df.query(q)
        elif t.op == "groupby_agg":
            by = t.args.get("by", [])
            aggs = t.args.get("aggs", {})
            df_map[t.target] = df.groupby(by, dropna=False).agg(aggs).reset_index()
        elif t.op == "join":
            right = t.args.get("right")
            how = t.args.get("how", "left")
            on = t.args.get("on")
            if right not in df_map:
                logs.append(f"[transform] join: right table {right} missing")
            else:
                df_map[t.target] = df.merge(df_map[right], how=how, on=on)
        elif t.op == "add_column":
            name = t.args.get("name")
            expr = t.args.get("expr")
            if name and expr:
                df_map[t.target][name] = pd.eval(expr, engine="python", parser="pandas", target=df_map[t.target])
        elif t.op == "parse_dates":
            cols = t.args.get("columns", [])
            for c in cols:
                df_map[t.target][c] = pd.to_datetime(df_map[t.target][c], errors="coerce")
        logs.append(f"[transform] {t.op} on {t.target}: shape={df_map[t.target].shape}")
    except Exception as e:
        logs.append(f"[transform] {t.op} error: {e}")

def render_charts(df_map: Dict[str, Any], charts: List[ChartSpec], logs: List[str]) -> List[str]:
    if not charts:
        return []
    
    import matplotlib
    matplotlib.use('Agg')
    
    images: List[str] = []
    
    for ch in charts[:3]:
        if ch.table not in df_map:
            logs.append(f"[chart] missing table {ch.table}")
            continue
        df = df_map[ch.table].head(200)
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            
            if ch.kind == "line":
                if isinstance(ch.y, list):
                    for col in ch.y[:3]:
                        ax.plot(df[ch.x], df[col], label=col)
                    ax.legend()
                else:
                    ax.plot(df[ch.x], df[ch.y])
            elif ch.kind == "bar":
                ax.bar(df[ch.x], df[ch.y] if isinstance(ch.y, str) else df[ch.y[0]])
            elif ch.kind == "scatter":
                ycol = ch.y if isinstance(ch.y, str) else (ch.y[0] if ch.y else None)
                ax.scatter(df[ch.x], df[ycol], alpha=0.6, s=20)
            elif ch.kind == "hist":
                ycol = ch.y if isinstance(ch.y, str) else (ch.y[0] if ch.y else None)
                bins = min(ch.bins or 20, 20)
                ax.hist(df[ycol], bins=bins)
            
            if ch.title: 
                ax.set_title(ch.title)
            
            images.append(fig_to_data_uri(fig))
            logs.append(f"[chart] {ch.kind} on {ch.table}")
        except Exception as e:
            logs.append(f"[chart] error: {e}")
    
    return images

def summarize(answer: AnswerSpec, df_map: Dict[str, Any], logs: List[str]) -> Any:
    if answer.type == "none":
        return None
    
    if answer.type == "basic_stats":
        tname = answer.table or next(iter(df_map.keys()), None)
        if not tname or tname not in df_map:
            return {"note": "no table available for stats"}
        df = df_map[tname]
        cols = (answer.columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])[:5]
        out = {}
        for c in cols:
            try:
                s = df[c].dropna()
                out[c] = {
                    "count": int(s.count()),
                    "mean": float(s.mean()) if len(s) else None,
                    "min": float(s.min()) if len(s) else None,
                    "max": float(s.max()) if len(s) else None,
                }
            except Exception:
                out[c] = {"error": "stat failed"}
        logs.append(f"[answer] basic_stats on {tname}")
        return out
    
    if answer.type == "text_summary":
        tname = answer.table or next(iter(df_map.keys()), None)
        if not tname or tname not in df_map:
            return "No data available."
        df = df_map[tname]
        return f"Rows: {len(df)}, Columns: {len(df.columns)}. Sample cols: {list(df.columns)[:3]}"
    
    return None

REQUIRED_FINAL_KEYS = {"answer", "tables", "images", "logs"}
def validate_final_result(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Output is not a JSON object.")
    missing = REQUIRED_FINAL_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Output missing required keys: {sorted(missing)}")
    return data