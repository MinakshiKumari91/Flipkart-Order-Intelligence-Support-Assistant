from pathlib import Path
import json
import joblib
import pandas as pd
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
 
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
 
_return_model = None
_image_model = None
_image_meta = None
 
def check_return_risk(order_features: dict) -> dict:
    global _return_model
    if _return_model is None:
        _return_model = joblib.load(MODELS / "return_risk_model.pkl")
    threshold_meta = json.loads((MODELS / "return_risk_threshold.json").read_text())
    t_rf = float(threshold_meta["t_rf"])
    p = float(_return_model.predict_proba(pd.DataFrame([order_features]))[0, 1])
    high_cut = min(1.0, t_rf + 0.15)
    bucket = "Low" if p < t_rf else ("High" if p >= high_cut else "Medium")
    return {
        "return_probability": round(p, 6),
        "risk_bucket": bucket,
        "t_rf": round(t_rf, 6),
        "high_cut": round(high_cut, 6),
    }
 
def _load_image_model():
    global _image_model, _image_meta
    if _image_model is not None: return
    _image_meta = json.loads((MODELS / "product_classifier_meta.json").read_text())
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(_image_meta["class_names"]))
    state = torch.load(MODELS / "product_classifier.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    _image_model = model
 
def classify_product_image(image_path: str) -> dict:
    _load_image_model()
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(tuple(_image_meta["input_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    img = Image.open(image_path).convert("L")
    x = tfm(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(_image_model(x), dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
    return {
        "predicted_category": _image_meta["class_names"][int(idx)],
        "confidence": round(float(conf), 6),
    }
 
if __name__ == "__main__":
    sample_order = {
        "product_category":"Apparel", "price_inr":1800, "discount_pct":35.0,
        "payment_method":"COD", "customer_tenure_days":120,
        "num_previous_orders":5, "num_previous_returns":2,
        "delivery_distance_km":150.0, "delivery_days":7,
        "is_weekend_order":1, "rating_given":None
    }
    print("Return-risk smoke test:", check_return_risk(sample_order))
    samples = sorted((ROOT / "data" / "sample_images").glob("*.png"))
    if samples:
        print("Image smoke test:", samples[0], classify_product_image(str(samples[0])))