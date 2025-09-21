import asyncio
import argparse
import json
import os
import re
from vertexai.generative_models import GenerativeModel, GenerationConfig
import vertexai
from utils import extract_text_from_pdf, chunk_text, extract_text_from_docx, extract_text
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
import warnings
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

load_dotenv()
warnings.filterwarnings("ignore")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "doclearn-470008")
LOCATION = os.getenv("GCP_REGION", "us-central1")

# Initialize Vertex AI
def init_vertex_ai():
    try:
        vertexai.init(project="PROJECT_ID", location="LOCATION")
    except Exception:
        vertexai.init(project="PROJECT_ID", location="asia-east1")  # Fallback region

init_vertex_ai()

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable))
)
async def process_chunk(chunk, role, jurisdiction, model, is_final_pass=False, total_clauses=0):
    if is_final_pass and total_clauses > 50:
        prompt = (
            f"As a {role} in {jurisdiction}, extract concise numbered clauses from the legal text. Club short clauses under the same topic (e.g., liability, penalties, obligations) into a single clause with combined text. "
            f"Assess their risk (low, medium, high, very high). Very high risk includes clauses with severe financial, legal, or operational impact (e.g., unlimited liability, strict penalties). "
            f"Return a JSON list of objects with: 'clause_number' (string, use the first number if clubbing), 'clause_text' (string, concise and combined for same-topic clauses), "
            f"'clause_risk' (low, medium, high, very high), 'negotiation' ('NIL' for low/medium/high risk, concise negotiation suggestion for very high risk). Ensure valid JSON output. "
            f'Example: [{{"clause_number": "1", "clause_text": "Combined liability clauses...", "clause_risk": "very high", "negotiation": "Limit liability..."}}]. '
            f"Text: {chunk}"
        )
    else:
        prompt = (
            f"As a {role} in {jurisdiction}, extract concise numbered clauses from the legal text. Club short clauses under the same topic (e.g., liability, penalties, obligations) into a single clause with combined text. "
            f"Assess their risk (low, medium, high, very high). Very high risk includes clauses with severe financial, legal, or operational impact (e.g., unlimited liability, strict penalties). "
            f"Return a JSON list of objects with: 'clause_number' (string, use the first number if clubbing), 'clause_text' (string, concise and combined for same-topic clauses), "
            f"'clause_risk' (low, medium, high, very high), 'negotiation' ('NIL' for all risks). Ensure valid JSON output. "
            f'Example: [{{"clause_number": "1", "clause_text": "Combined liability clauses...", "clause_risk": "very high", "negotiation": "NIL"}}]. '
            f"Text: {chunk}"
        )
    try:
        response = await model.generate_content_async(
            prompt,
            generation_config=GenerationConfig(max_output_tokens=4000, temperature=0.2),
            stream=True
        )
        full_response = ""
        async for part in response:
            full_response += part.text
        # Clean response: Remove markdown, extra whitespace
        cleaned_response = re.sub(r'```json\n|```|\n\s*\n', '', full_response).strip()
        try:
            clauses = json.loads(cleaned_response)
            return clauses
        except json.JSONDecodeError:
            # Fallback: Write chunk as a single clause
            return[ {
                "clause_number": "unknown",
                "clause_text": chunk[:1000],  # Truncate for safety
                "clause_risk": "medium",
                "negotiation": "NIL"
            }]
            
    except Exception as e:
        return [{"error": f"Processing error: {str(e)}"}]

async def process_document(extracted_json, role, jurisdiction, chunk_size=1000):
    model = GenerativeModel(GEMINI_MODEL)
    text = extracted_json.get("text", "")
    if not text:
        return [{"error": "No text provided"}]

    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    all_clauses = []

    for chunk in chunks:
        clauses = await process_chunk(chunk, role, jurisdiction, model, is_final_pass=False)
        all_clauses.extend(clauses)

    total_clauses = len([c for c in all_clauses if "error" not in c])
    if total_clauses > 50:
        very_high_clauses = []
        for chunk in chunks:
            clauses = await process_chunk(
                chunk, role, jurisdiction, model,
                is_final_pass=True, total_clauses=total_clauses
            )
            very_high_clauses.extend([c for c in clauses if c.get("clause_risk") == "very high"])

        very_high_clauses = sorted(very_high_clauses, key=lambda x: x["clause_number"])[:50]
        other_clauses = [
            c for c in all_clauses if c.get("clause_risk") != "very high" or c in very_high_clauses
        ]
        return other_clauses + very_high_clauses

    return all_clauses

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Legal Negotiation Analyzer is running on Cloud Run 🚀"}

class NegotiationInput(BaseModel):
    role: str
    jurisdiction: str
    text: str

@app.post("/analyze/")
async def analyze(input: NegotiationInput):
    result = await process_document(
        {"text": input.text}, input.role, input.jurisdiction
    )
    return {"clauses": result}

import os, uvicorn
port = int(os.environ.get("PORT", 8080))
uvicorn.run(app, host="0.0.0.0", port=port)
