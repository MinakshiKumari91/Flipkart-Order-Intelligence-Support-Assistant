BLOCK_PATTERNS = [
    "ignore previous instructions",
    "ignore all rules",
    "pretend you are",
]
 
GROUNDING_THRESHOLD = 0.45  # tune from your own retrieval scores and document your chosen value
 
def input_guardrail(text: str):
    low = text.lower()
    hit = next((p for p in BLOCK_PATTERNS if p in low), None)
    return {"blocked": hit is not None, "matched_pattern": hit}
 
def grounded(top_score: float):
    return top_score >= GROUNDING_THRESHOLD