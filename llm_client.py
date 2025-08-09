import asyncio
import json
import re
from typing import Any, Dict
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI, APIError
from fastapi import HTTPException

from settings import SETTINGS

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
client = OpenAI(api_key=SETTINGS.api_key, base_url=SETTINGS.base_url)
thread_pool = ThreadPoolExecutor(max_workers=3) # Global thread pool

def extract_payload(text: str) -> str:
    """Extracts JSON payload from a markdown code fence."""
    match = _CODE_FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()

def coerce_json(text: str) -> Dict[str, Any]:
    """Tries to parse JSON, even if it has extra text around it."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            return json.loads(snippet)
        raise

async def llm_call_async(system_prompt: str, user_prompt: str, model: str, temperature: float = 0, max_tokens: int = 1000) -> str:
    """
    Makes a non-blocking call to the LLM API using a thread pool.
    This is an improvement over the original's ThreadPoolExecutor in the async
    function, as it uses a global pool.
    """
    if not SETTINGS.api_key:
        raise HTTPException(status_code=500, detail="LLM_API_KEY is not set.")

    def _sync_call():
        try:
            # First attempt with JSON mode for stricter output
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content.strip()
        except APIError:
            # Fallback without JSON mode on API error
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content.strip()

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(thread_pool, _sync_call)

async def generate_json_response(system_prompt: str, user_prompt: str, model: str, schema: Any, retries: int = 1) -> Any:
    """
    Generates a Pydantic-validated JSON object from an LLM call.
    This encapsulates the retry and validation logic.
    """
    for _ in range(retries + 1):
        try:
            raw_response = await llm_call_async(system_prompt, user_prompt, model)
            payload = coerce_json(extract_payload(raw_response))
            return schema(**payload)
        except Exception as e:
            if _ == retries:
                raise ValueError(f"Failed to generate valid JSON after {retries+1} attempts: {e}")
    
    raise ValueError("Unexpected error in LLM JSON generation loop.")