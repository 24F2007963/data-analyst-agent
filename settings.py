import os
from pydantic import BaseModel, Field

class LLMSettings(BaseModel):
    """Configuration for LLM API access."""
    api_key: str = Field(..., description="API key for the LLM service.")
    base_url: str = Field(
        os.getenv("LLM_BASE_URL", "https://aipipe.org/openrouter/v1"),
        description="Base URL for the OpenAI-compatible API."
    )
    planner_model: str = Field(
        os.getenv("LLM_PLANNER_MODEL", "gpt-4o-mini"),
        description="Model name for the planning engine."
    )
    coder_model: str = Field(
        os.getenv("LLM_CODER_MODEL", "gpt-4o-mini"),
        description="Model name for the code generation engine."
    )
    formatter_model: str = Field(
        os.getenv("LLM_FORMATTER_MODEL", "gpt-4o-mini"),
        description="Model name for the output formatter."
    )
    verifier_model: str = Field(
        os.getenv("LLM_VERIFY_MODEL", "google/gemini-2.0-flash-lite-001"),
        description="Model name for the output VERIFY."
    )
    


    def __post_init__(self):
        # Your hardcoded API key, but good to check from env
        if self.api_key == "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6InNtcml0aS50dXJhbkBnbWFpbC5jb20ifQ.9pqId92evbHHNgH8I5FWqTHYqdMSmo-bM2TnLi2XEgg":
            print("Warning: Using hardcoded API key. Consider using an environment variable.")

# Initialize settings
SETTINGS = LLMSettings(api_key=os.getenv("LLM_API_KEY", "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6InNtcml0aS50dXJhbkBnbWFpbC5jb20ifQ.9pqId92evbHHNgH8I5FWqTHYqdMSmo-bM2TnLi2XEgg"))