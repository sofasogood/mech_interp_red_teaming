import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import requests
from typing import Dict, List, Tuple, Any
import pickle
import os
from dotenv import load_dotenv
import goodfire

from classifier import load_and_use_model

# Load environment variables from .env file
load_dotenv(override=True)

# Set page config
st.set_page_config(
    page_title="Red Team Prompt Testing Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Add CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF5555;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555555;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F0F2F6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .feature-importance {
        margin-top: 1rem;
        padding: 10px;
        background-color: #F8F9FA;
        border-radius: 5px;
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #888888;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions for API and classifier

def call_goodfire_api(prompt: str, variant: str = "meta-llama/Llama-3.3-70B-Instruct") -> Dict:
    """
    Call the Goodfire API with the given prompt and return the results.
    Replace this with your actual API call implementation.
    """
    try:
        api_key = os.environ.get("GOODFIRE_API_KEY", "")
        client = goodfire.AsyncClient(api_key=api_key)
        # Instantiate a model variant. 
        variant = goodfire.Variant(variant)

        response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model=variant
    )
        response_str = str(response.choices[0].message["content"])
    except Exception as e:
        response_str = f"Error calling API: {str(e)}"
    return  [
    [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response_str}
    ]
]

def call_goodfire_inspector(api_response: Dict) -> Dict:
    """
    Call the Goodfire Inspector API to extract features from the response.
    """
    data = {}
    try:
        api_key = os.environ.get("GOODFIRE_API_KEY", "")
        client = goodfire.AsyncClient(api_key=api_key)
        variant = goodfire.Variant(variant)

        inspector = client.features.inspect(
                api_response[0],
                model=variant
            )
    except Exception as e:
        print(f"Error processing index {i}: {e}")
        print(f"Finished getting inspector for index {i}")
    data["prompt"] = api_response[0][0]["content"]
    data["response"] = api_response[0][1]["content"]
    for activation in inspector.top(k=5):
        data[activation.feature.label] = activation.activation
    return data

# def load_classifier():
#     """
#     Load your pre-trained classifier model.
#     Replace this with your actual model loading code.
#     """
#     # This is a placeholder - replace with your actual model loading code
#     try:
#         with open('binary_cla', 'rb') as f:
#         #             classifier = pickle.load(f)
#         # For now, we'll simulate a classifier
#         class MockClassifier:
#             def predict_proba(self, features):
#                 # Simulate probability prediction based on input features
#                 # In reality, this would use your actual classifier
#                 return np.array([[0.3, 0.7]])
                
#         return MockClassifier()
#     except Exception as e:
#         st.error(f"Error loading classifier: {str(e)}")
#         return None

def format_features(api_response: Dict) -> pd.DataFrame:
    """
    Extract and format features from the API response.
    This should be customized based on your API's response structure.
    """
    df_features = pd.DataFrame()    
        
    # Example feature extraction - modify based on your actual API response
    for col in api_response:
        if col != "prompt" and col != "response":
            continue
        df_features[col] = api_response[col]
    
    # Sort features by value in descending order
    return df_features

def predict_risk(prob: float) -> str:
    """
    Use the classifier to predict the probability of a successful attack
    based on API features.
    """    
    try:    
        # Determine risk level
        if prob < 0.3:
            risk = "Low Risk"
        elif prob < 0.7:
            risk = "Medium Risk"
        else:
            risk = "High Risk"
            
        return prob, risk
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return "Error in prediction"

# Main app
def main():
    
    # App header
    st.markdown('<div class="main-header">Red Team Prompt Testing Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analyze prompts for potential attacks and evaluate risk factors</div>', unsafe_allow_html=True)
    
    # Create columns for layout
    col1, col2 = st.columns([3, 2])
    
    # Prompt input area
    with col1:
        prompt = st.text_area("Enter a prompt to test", height=200, 
                              placeholder="Enter the prompt you want to analyze for potential risks...")
        
        analyze_button = st.button("Analyze Prompt", type="primary")
        
        # Advanced options collapsible
        with st.expander("Advanced Options"):
            api_timeout = st.slider("API Timeout (seconds)", 5, 60, 30)
            show_raw_response = st.checkbox("Show Raw API Response")
    
    # Results area
    if analyze_button and prompt:
        # Call the API
        with st.spinner("Analyzing prompt..."):
            api_response = call_goodfire_api(prompt)
            raw_features = call_goodfire_inspector(api_response)

        
        # Format features
        features = format_features(raw_features)
        # Load the classifier
        labels, predictions = load_and_use_model("binary_classifier.pt", features)



        
        # Predict probability
        prob, risk_level = predictions[1], predict_risk(predictions[1])
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Attack Probability", f"{predictions[1]:.1%}")
        
        with col2:
            st.metric("Risk Level", risk_level)
            
        # with col3:
        #     if "error" in api_response:
        #         st.metric("API Status", "Error")
        #     else:
        #         st.metric("API Status", "Success")
        
        # # Display feature importance
        # if features:
        #     st.subheader("Feature Importance")
                        
        #     # Create horizontal bar chart for top features
        #     top_n = min(10, len(features))
        #     fig = px.bar(
        #         features.head(top_n),
        #         y="name",
        #         x="value",
        #         orientation='h',
        #         title=f"Top {top_n} Important Features",
        #         labels={"value": "Importance", "name": "Feature"},
        #         color="value",
        #         color_continuous_scale=["green", "yellow", "red"],
        #         text="value"
        #     )
        #     fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        #     st.plotly_chart(fig, use_container_width=True)
            
        #     # Display feature details in a table
        #     st.dataframe(
        #         features_df[["name", "value", "description"]].rename(
        #             columns={"name": "Feature", "value": "Importance", "description": "Description"}
        #         ),
        #         use_container_width=True
        #     )
            
        # Display raw API response if requested
        if show_raw_response and api_response:
            with st.expander("Raw API Response"):
                st.json(api_response)
    
    # Sidebar with additional info
    with st.sidebar:
        st.header("About")
        st.write("This dashboard uses the Goodfire API and a custom classifier to analyze prompts for potential red team attacks.")
        
        st.header("How to Use")
        st.write("1. Enter a prompt in the text area")
        st.write("2. Click 'Analyze Prompt'")
        st.write("3. Review the risk assessment and feature importance")
        
        st.header("Features")
        st.write("- Attack probability estimation")
        st.write("- Feature importance visualization")
        st.write("- Risk level assessment")
        
    # Footer
    st.markdown('<div class="footer">Powered by Goodfire API and Custom Classifier</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()