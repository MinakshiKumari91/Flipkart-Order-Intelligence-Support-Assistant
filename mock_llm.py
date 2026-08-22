import json
 
def make_response(answer, source, confidence):
    return {
        "answer": str(answer),
        "source": source,
        "confidence": round(float(confidence), 4),
    }
 
def mock_policy_response(retrieved):
    best = retrieved[0]
    return make_response(best["text"], "policy_kb", best["score"])
 
def mock_return_response(tool_output):
    p = tool_output["return_probability"]
    bucket = tool_output["risk_bucket"]
    text = f"Predicted return probability is {p:.3f}; risk bucket is {bucket}."
    return make_response(text, "return_risk_tool", max(p, 1-p))
 
def mock_image_response(tool_output):
    text = f"Predicted product category is {tool_output['predicted_category']}."
    return make_response(text, "image_classifier_tool", tool_output["confidence"])