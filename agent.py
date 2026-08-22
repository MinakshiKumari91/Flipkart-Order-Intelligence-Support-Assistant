from pathlib import Path
from typing import TypedDict, Optional, Dict, Any
from langgraph.graph import StateGraph, END
 
from part3_rag import retrieve
from part3_tools import check_return_risk, classify_product_image
from part3_guardrails import input_guardrail, grounded, GROUNDING_THRESHOLD
from part3_mock_llm import mock_policy_response, mock_return_response, mock_image_response, make_response
 
ROOT = Path(__file__).resolve().parent
 
SYSTEM_PROMPT = """
ROLE: You are Flipkart's support assistant for this project.
SPECIFIC: Answer only from retrieved policy context or approved tool output.
SHORT: Keep answers concise and factual.
SURROUND: Treat retrieved context/tool output as the only factual evidence for the answer.
SINGLE: Return exactly one structured object with answer, source, confidence.
 
Few-shot intent examples:
1) 'Can I return shoes after five days?' -> policy
2) 'How likely is this order to be returned?' -> return_risk
3) 'What category is this product image?' -> product_image
""".strip()
 
class AgentState(TypedDict, total=False):
    user_text: str
    intent: str
    blocked: bool
    block_reason: Optional[str]
    order_features: Dict[str, Any]
    image_path: str
    retrieved: list
    tool_output: dict
    response: dict
    conversation: dict
 
def intent_node(state: AgentState):
    text = state["user_text"].lower()
    g = input_guardrail(text)
    if g["blocked"]:
        return {"blocked": True, "block_reason": g["matched_pattern"], "intent": "blocked"}
    if any(k in text for k in ["image", "photo", "picture", "category of this product"]):
        intent = "product_image"
    elif any(k in text for k in ["return risk", "likely to be returned", "return probability", "risk for order"]):
        intent = "return_risk"
    else:
        intent = "policy"
    return {"intent": intent, "blocked": False}
 
def route_after_intent(state):
    if state.get("blocked"): return "response"
    if state["intent"] == "policy": return "rag"
    return "tool"
 
def rag_node(state: AgentState):
    return {"retrieved": retrieve(state["user_text"], k=3)}
 
def tool_node(state: AgentState):
    if state["intent"] == "return_risk":
        features = state.get("order_features") or state.get("conversation", {}).get("last_order_features")
        if not features:
            return {"tool_output": {"error":"Missing order_features"}}
        return {"tool_output": check_return_risk(features),
                "conversation": {**state.get("conversation", {}), "last_order_features": features}}
    if state["intent"] == "product_image":
        path = state.get("image_path") or state.get("conversation", {}).get("last_image_path")
        if not path:
            return {"tool_output": {"error":"Missing image_path"}}
        return {"tool_output": classify_product_image(path),
                "conversation": {**state.get("conversation", {}), "last_image_path": path}}
    return {}
 
def response_node(state: AgentState):
    if state.get("blocked"):
        return {"response": make_response(
            f"Request blocked by input guardrail ({state.get('block_reason')}).",
            "policy_kb", 1.0)}
    if state["intent"] == "policy":
        retrieved = state.get("retrieved", [])
        if not retrieved:
            return {"response": make_response("I do not have grounded policy evidence for that question.", "policy_kb", 0.0)}
        top = retrieved[0]
        if not grounded(top["score"]):
            msg = f"I cannot answer from the current policy knowledge base. top_similarity={top['score']:.4f}; threshold={GROUNDING_THRESHOLD:.4f}."
            return {"response": make_response(msg, "policy_kb", top["score"])}
        return {"response": mock_policy_response(retrieved)}
    out = state.get("tool_output", {})
    if "error" in out:
        source = "return_risk_tool" if state["intent"] == "return_risk" else "image_classifier_tool"
        return {"response": make_response(out["error"], source, 0.0)}
    if state["intent"] == "return_risk": return {"response": mock_return_response(out)}
    return {"response": mock_image_response(out)}
 
builder = StateGraph(AgentState)
builder.add_node("intent", intent_node)
builder.add_node("rag", rag_node)
builder.add_node("tool", tool_node)
builder.add_node("response", response_node)
builder.set_entry_point("intent")
builder.add_conditional_edges("intent", route_after_intent, {"rag":"rag", "tool":"tool", "response":"response"})
builder.add_edge("rag", "response")
builder.add_edge("tool", "response")
builder.add_edge("response", END)
app = builder.compile()
 
def run_agent(user_text, conversation=None, order_features=None, image_path=None):
    state = {
        "user_text": user_text,
        "conversation": conversation or {},
    }
    if order_features is not None: state["order_features"] = order_features
    if image_path is not None: state["image_path"] = image_path
    result = app.invoke(state)
    return result["response"], result.get("conversation", conversation or {}), result
 
if __name__ == "__main__":
    response, conv, raw = run_agent("What is the return window for footwear?")
    print(response)