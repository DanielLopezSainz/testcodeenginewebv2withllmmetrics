import os
import json
import logging
import nltk
import pandas as pd
from flask import Flask, request, jsonify, render_template
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson_openscale import APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_metrics_plugin.metrics.llm.utils.constants import LLMTextMetricGroup, LLMCommonMetrics

# Download necessary NLTK assets
nltk.download("stopwords")
nltk.download("punkt_tab")

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

# Fetch credentials from environment variables
API_KEY = os.getenv("API_KEY", "default_api_key")
PROJECT_ID = os.getenv("PROJECT_ID", "default_project_id")
IAM_URL = os.getenv("IAM_URL", "https://iam.cloud.ibm.com")
endpoint_url = "https://us-south.ml.cloud.ibm.com"

# Flask app initialization
app = Flask(__name__)

# WatsonxAI setup
credentials = {
    "apikey": API_KEY,
    "url": endpoint_url
}

generate_params = {
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.MIN_NEW_TOKENS: 10
}

model = ModelInference(
    model_id=ModelTypes.FLAN_T5_XXL,
    params=generate_params,
    credentials=credentials,
    project_id=PROJECT_ID
)

authenticator = IAMAuthenticator(apikey=API_KEY, url=IAM_URL)
client = APIClient(authenticator=authenticator, service_url="https://aiopenscale.cloud.ibm.com")

def scoring_fn(input_prompts):
    batch_size = 2
    model_responses = []
    prompts = input_prompts["prompts"]
    for i in range(0, len(prompts), batch_size):
        upper_limit = min(i + batch_size, len(prompts))
        model_responses.extend(model.generate_text(prompt=prompts[i:upper_limit].tolist(), guardrails=True))
    return pd.DataFrame({"generated_text": model_responses})

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Render the web interface and handle text input.
    """
    response_text = ""
    if request.method == "POST":
        input_text = request.form.get("input_text", "")
        if input_text:
            try:
                # Call the WatsonxAI metric computation function
                config_json = {
                    "configuration": {
                        "scoring_fn": scoring_fn,
                        "prompt_template": input_text,
                        "feature_columns": ["input"],
                        LLMTextMetricGroup.QA.value: {
                            LLMCommonMetrics.ROBUSTNESS.value: {
                                "adversarial_robustness": {
                                    "show_recommendations": True,
                                    "explanations_count": 3
                                }
                            }
                        }
                    }
                }
                response_text = client.llm_metrics.compute_metrics(config_json)
            except Exception as e:
                response_text = f"Error: {str(e)}"
    return render_template("index.html", response_text=response_text)

@app.route("/generate", methods=["GET", "POST"])
def generate():
    """
    Endpoint to handle input prompts and generate a response.
    """
    try:
        if request.method == "POST":
            data = request.json
            prompt = data.get("prompt")
            if not prompt:
                return jsonify({"error": "Prompt is required"}), 400
        else:
            prompt = request.args.get("prompt")
            if not prompt:
                return jsonify({"error": "Prompt query parameter is required"}), 400

        config_json = {
            "configuration": {
                "scoring_fn": scoring_fn,
                "prompt_template": prompt,
                "feature_columns": ["input"],
                LLMTextMetricGroup.QA.value: {
                    LLMCommonMetrics.ROBUSTNESS.value: {
                        "adversarial_robustness": {
                            "show_recommendations": True,
                            "explanations_count": 3
                        }
                    }
                }
            }
        }
        response = client.llm_metrics.compute_metrics(config_json)
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET", "POST"])
def health_check():
    """
    Health check endpoint to verify the application is running.
    """
    return jsonify({
        "status": "healthy",
        "PROJECT_ID": PROJECT_ID
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
