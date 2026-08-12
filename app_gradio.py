import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr

from src.serving.inference import predict


def predict_churn(
    tenure,
    monthly_charges,
    total_charges,
    contract,
    payment_method,
    internet_service,
    tech_support,
    online_security,
    support_calls,
):
    inputs = {
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract": contract,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "tech_support": tech_support,
        "online_security": online_security,
        "support_calls": support_calls,
    }
    result = predict(inputs)
    return result["prediction"], result["probability"], result["confidence"]


def main():
    title = "Customer Churn Predictor"
    description = (
        "Use Gradio to predict customer churn risk with the trained model. "
        "Enter customer details and inspect the churn probability and confidence." 
    )

    iface = gr.Interface(
        fn=predict_churn,
        inputs=[
            gr.Number(label="Tenure (months)", value=12, precision=0),
            gr.Number(label="Monthly Charges ($)", value=70.0),
            gr.Number(label="Total Charges ($)", value=840.0),
            gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract"),
            gr.Dropdown(["Cash", "Credit", "Debit", "UPI"], label="Payment Method"),
            gr.Dropdown(["DSL", "Fiber", "No"], label="Internet Service"),
            gr.Dropdown(["Yes", "No"], label="Tech Support"),
            gr.Dropdown(["Yes", "No"], label="Online Security"),
            gr.Number(label="Customer Support Calls", value=1, precision=0),
        ],
        outputs=[
            gr.Textbox(label="Prediction"),
            gr.Number(label="Churn Probability"),
            gr.Number(label="Confidence"),
        ],
        title=title,
        description=description,
        examples=[
            [12, 70.0, 840.0, "Month-to-month", "Credit", "DSL", "Yes", "Yes", 1],
        ],
    )

    iface.launch()


if __name__ == "__main__":
    main()
