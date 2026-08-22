from pathlib import Path
import json
 
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
 
policies = [
  {"doc_id":"P01","title":"Apparel returns","text":"Apparel items may be requested for return within 10 days of delivery when the item is unused and retains original tags. Size or fit issues are eligible for reverse pickup where service is available."},
  {"doc_id":"P02","title":"Footwear returns","text":"Footwear may be requested for return within 10 days of delivery. The product should be unworn outdoors and returned with the original box and accessories."},
  {"doc_id":"P03","title":"Electronics returns","text":"Eligible electronics may be reported within 7 days of delivery for damage, defect, or incorrect item. Some electronics may require troubleshooting or authorized-service verification before replacement or return."},
  {"doc_id":"P04","title":"Home category returns","text":"Eligible home products may be requested for return within 10 days of delivery. Large items may require an inspection before reverse pickup is approved."},
  {"doc_id":"P05","title":"Beauty hygiene restriction","text":"Opened personal-care or beauty products are generally not returnable for hygiene reasons unless the item is damaged, defective, or incorrect. Unopened eligible items follow the category return window shown at purchase."},
  {"doc_id":"P06","title":"COD refunds","text":"Approved refunds for cash-on-delivery orders are issued to the customer’s selected eligible refund destination after return verification. Processing normally starts after the returned item passes the required check."},
  {"doc_id":"P07","title":"Prepaid refunds","text":"Approved refunds for prepaid orders are initiated to the original payment method after return verification. Bank or payment-provider posting time may occur after the refund is initiated."},
  {"doc_id":"P08","title":"Standard delivery SLA","text":"Standard-delivery orders are delivered according to the promise date shown for the order. Delays caused by weather, transport disruption, or remote-location constraints may require a revised delivery estimate."},
  {"doc_id":"P09","title":"Remote delivery SLA","text":"Remote or hard-to-service locations may have longer delivery estimates than metro areas. The support agent should use the order’s latest promised date when discussing delivery timing."},
  {"doc_id":"P10","title":"Reverse pickup eligibility","text":"Reverse pickup is available only where the product and pickup location are serviceable. If pickup is unavailable, the customer may be offered an approved alternative return method."},
  {"doc_id":"P11","title":"Damaged item","text":"A customer receiving an item damaged in transit should report it within the applicable category window. The support workflow may request packaging or damage details before approving the resolution."},
  {"doc_id":"P12","title":"Wrong item","text":"If the delivered item does not match the ordered product, the customer can request a return or replacement within the applicable category window. The returned item should include the received accessories and packaging where available."},
  {"doc_id":"P13","title":"Cancellation","text":"An order may be cancelled before it reaches a stage where cancellation is no longer operationally possible. After dispatch, the customer may need to follow the applicable return process instead."},
  {"doc_id":"P14","title":"Refund verification","text":"Refund processing begins only after the return reaches the required verification stage. A refund may be held for manual review when product condition or transaction details require additional checks."},
  {"doc_id":"P15","title":"Replacement availability","text":"Replacement depends on inventory availability and category eligibility. If a replacement cannot be fulfilled, an eligible refund may be offered instead."}
]
 
answer_key = [
  {"query":"How many days do I have to return shoes?","relevant_docs":["P02"]},
  {"query":"Can you pick up my return from my home?","relevant_docs":["P10"]},
  {"query":"What happens to my refund for a COD order?","relevant_docs":["P06","P14"]},
  {"query":"My electronics item arrived damaged. What can I do?","relevant_docs":["P03","P11"]},
  {"query":"Why is delivery taking longer in a remote location?","relevant_docs":["P08","P09"]}
]
 
with open(DATA / "policies.json", "w") as f: json.dump(policies, f, indent=2)
with open(DATA / "retrieval_answer_key.json", "w") as f: json.dump(answer_key, f, indent=2)
print("Wrote", len(policies), "policy documents")
print("Wrote", len(answer_key), "retrieval-evaluation queries")