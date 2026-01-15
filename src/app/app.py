from re import S
from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from sklearn.metrics import PredictionErrorDisplay

# initialize FastAPI app
app = FastAPI(
    title = "Customer Churn Prediction API",
    description = "API for predicting customer churn using a pre-trained machine learning model.",
    version = "1.0.0"
)
# ---health check endpoint---
@app.get("/")

def root():
    
    return {"status": "ok"}

# request schema
class CustomerData(BaseModel):
    SupportCalls: int
    InternetService: str
    OnlineSecurity: str
    TechSupport: str
    Contract: str
    PaymentMethod: str
    tenure: int
    MonthlyCharges: float
    TotalCharges: float


# main preidction endpoint
@app.post("/predict")

def get_prediction(data:CustomerData):
    """
    Endpoint to get churn prediction for a customer.

    Args:
        data (CustomerData): Input customer data.

    Returns:
    """
    try:
        # Convert pydantic model tp doct and call inference pipeline
        result = PredictionErrorDisplay(data.dict())
        return {"churn_prediction": result}
    except Exception as e:
        return {"error": str(e)}
    
    #------------------------------------------------------------#

    # === GRADIO WEB INTERFACE ===
def gradio_interface( MultipleLines,SupportCalls,
    InternetService, OnlineSecurity,
    TechSupport, Contract, PaymentMethod, tenure, MonthlyCharges, TotalCharges
):
    """
    Gradio interface function that processes form inputs and returns prediction.
    
    This function:
    1. Takes individual form inputs from Gradio UI
    2. Constructs the data dictionary matching the API schema
    3. Calls the same inference pipeline used by the API
    4. Returns user-friendly prediction string
    
    """
    # Construct data dictionary matching CustomerData schema
    data = {
        "SupportCalls":SupportCalls,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "TechSupport": TechSupport,
        "Contract": Contract,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),              # Ensure integer type
        "MonthlyCharges": float(MonthlyCharges),  # Ensure float type
        "TotalCharges": float(TotalCharges),      # Ensure float type
    }
    
    # Call same inference pipeline as API endpoint
    result = predict(data)
    return str(result)  # Return as string for Gradio display

# === GRADIO UI CONFIGURATION ===
# Build comprehensive Gradio interface with all customer features
demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
    

        # Phone services section
        gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes"),
        gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines", value="No"),
        
        # Internet services section (key churn predictors)
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="Yes"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="Yes"),
        
        # Contract and billing section (major churn factors)
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes"),
        gr.Dropdown([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], label="Payment Method", value="Electronic check"),
        
        # Numeric features (important for churn prediction)
        gr.Number(label="Tenure (months)", value=1, minimum=0, maximum=100),
        gr.Number(label="Monthly Charges ($)", value=85.0, minimum=0, maximum=200),
        gr.Number(label="Total Charges ($)", value=85.0, minimum=0, maximum=10000),
    ],
    outputs=gr.Textbox(label="Churn Prediction", lines=2),
    title="🔮 Customer Churn Predictor",
    description="""
    **Predict customer churn probability using machine learning**
    
    Fill in the customer details below to get a churn prediction. The model uses XGBoost trained on 
    historical telecom customer data to identify customers at risk of churning.
    
    💡 **Tip**: Month-to-month contracts with fiber optic internet and electronic check payments 
    tend to have higher churn rates.
    """
    theme=gr.themes.Soft()  # Professional appearance
)

# === MOUNT GRADIO UI INTO FASTAPI ===
# This creates the /ui endpoint that serves the Gradio interface
# IMPORTANT: This must be the final line to properly integrate Gradio with FastAPI
app = gr.mount_gradio_app(
    app,           # FastAPI application instance
    demo,          # Gradio interface
    path="/ui"     # URL path where Gradio will be accessible
)



