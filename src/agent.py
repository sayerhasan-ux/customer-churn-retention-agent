from typing import TypedDict
import requests
from langgraph.graph import StateGraph, START, END

API_URL = "http://127.0.0.1:8000/predict"

class State(TypedDict, total=False):
    customer: dict
    result: dict
    action: str
    message: str

ACTIONS = {
    "Contract_Two year": "Offer a discount for moving to a 1-year or 2-year contract",
    "Contract_One year": "Offer a discount for moving to a 1-year or 2-year contract",
    "tenure": "Schedule an onboarding check-in call (customer is still new)",
    "InternetService_Fiber optic": "Check fiber service quality and offer a support visit",
    "PaymentMethod_Electronic check": "Offer a small credit for switching to automatic payment",
    "TechSupport_Yes": "Offer a free tech support trial",
    "OnlineSecurity_Yes": "Offer a free online security trial",
}
NAMES = {
    "tenure": "short tenure",
    "Contract_Two year": "no 2-year contract",
    "Contract_One year": "no 1-year contract",
    "InternetService_Fiber optic": "fiber optic service",
    "PaymentMethod_Electronic check": "pays by electronic check",
    "PaperlessBilling_Yes": "paperless billing",
}

def predict_node(state: State):
    resp = requests.post(API_URL, json=state["customer"], timeout=10)
    resp.raise_for_status()
    return {"result": resp.json()}


def route(state: State):
    return "high" if state["result"]["risk"] == "high" else "low"


def no_action_node(state: State):
    p = state["result"]["churn_probability"]
    return {"action": "No action",
            "message": f"Churn risk is low ({p:.0%}). No retention action needed."}


def retain_node(state: State):
    r = state["result"]
    action = next((ACTIONS[x["feature"]] for x in r["reasons_to_leave"]
                   if x["feature"] in ACTIONS),
                  "Assign to a retention agent for a personal call")
    top = ", ".join(NAMES.get(x["feature"], x["feature"]) for x in r["reasons_to_leave"])
    message = (f"Churn risk is high ({r['churn_probability']:.0%}). "
               f"Main risk factors: {top}. Recommended action: {action}.")
    return {"action": action, "message": message}

graph = StateGraph(State)
graph.add_node("predict", predict_node)
graph.add_node("no_action", no_action_node)
graph.add_node("retain", retain_node)
graph.add_edge(START, "predict")
graph.add_conditional_edges("predict", route, {"high": "retain", "low": "no_action"})
graph.add_edge("retain", END)
graph.add_edge("no_action", END)
agent = graph.compile()


if __name__ == "__main__":
    customer = {"gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
                "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
                "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
                "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check", "MonthlyCharges": 75.0, "TotalCharges": 150.0}
    out = agent.invoke({"customer": customer})
    print(out["message"])

    