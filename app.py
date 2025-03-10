import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
from typing import Dict
import os
from dotenv import load_dotenv
import goodfire
import traceback
from classifier import load_and_use_model

# Load environment variables from .env file
load_dotenv(override=True)


try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Set page config
st.set_page_config(
    page_title="Red Team Prompt Testing Dashboard", page_icon="🛡️", layout="wide"
)

# Add CSS for better styling
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# Helper functions for API and classifier


def call_goodfire_api(
    prompt: str, variant_name: str = "meta-llama/Llama-3.3-70B-Instruct"
):
    """
    Call the Goodfire API with the given prompt and return the results.
    """
    try:
        api_key = os.environ.get("GOODFIRE_API_KEY", "")
        if not api_key:
            return [
                [
                    {"role": "user", "content": prompt},
                    {
                        "role": "assistant",
                        "content": "Error: API key not found in environment variables",
                    },
                ]
            ]

        # Create synchronous client
        client = goodfire.Client(api_key=api_key)

        # Create the variant
        variant = goodfire.Variant(variant_name)

        # Make the API call
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=variant
        )

        # Extract the response content
        response_str = response.choices[0].message["content"]

        return [
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response_str},
            ]
        ]
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        traceback.print_exc()
        return [
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"Error calling API: {str(e)}"},
            ]
        ]


def call_goodfire_inspector(
    api_response, variant_name: str = "meta-llama/Llama-3.3-70B-Instruct"
):
    """
    Call the Goodfire Inspector API to extract features from the response.
    """
    data = {"prompt": "", "response": ""}

    try:
        if not api_response or len(api_response) == 0 or len(api_response[0]) < 2:
            st.warning("Invalid API response format")
            return data

        # Extract basic content
        data["prompt"] = api_response[0][0]["content"]
        data["response"] = api_response[0][1]["content"]

        # Check for API errors
        if "Error calling API" in data["response"]:
            st.warning("Skipping inspector due to API error in the response")
            return data

        # Get API key
        api_key = os.environ.get("GOODFIRE_API_KEY", "")
        if not api_key:
            st.warning("API key not found for inspector")
            return data

        # Create client and variant
        client = goodfire.Client(api_key=api_key)
        variant = goodfire.Variant(variant_name)

        # Call inspector
        context = client.features.inspect(messages=api_response[0], model=variant)

        # Get top features (limited to 5)
        top_features = context.top(k=5)

        # Add these features to our data
        for feature_act in top_features:
            data[feature_act.feature.label] = feature_act.activation

    except Exception as e:
        st.error(f"Inspector Error: {str(e)}")
        traceback.print_exc()

    return data


def format_features(api_response: Dict) -> pd.DataFrame:
    """
    Extract and format features from the API response.
    This should be customized based on your API's response structure.
    """
    data_dict = {}

    # Add prompt and response as strings
    data_dict["prompt"] = [str(api_response.get("prompt", ""))]
    data_dict["response"] = [str(api_response.get("response", ""))]

    # Process all other fields as floats
    for k, v in api_response.items():
        if k not in ["prompt", "response"]:
            try:
                # Convert to float, with fallback to 0.0 if conversion fails
                data_dict[k] = [float(v)]
            except (ValueError, TypeError):
                # If conversion fails, use a default value of 0.0
                data_dict[k] = [0.0]

    return pd.DataFrame(data_dict)


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

        return risk
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return "Error in prediction"


# Main app
def main():
    # App header
    st.markdown(
        '<div class="main-header">Red Team Prompt Testing Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Analyze prompts for potential attacks and evaluate risk factors</div>',
        unsafe_allow_html=True,
    )

    # Create columns for layout
    col1, col2 = st.columns([3, 2])

    # Prompt input area
    with col1:
        prompt = st.text_area(
            "Enter a prompt to test",
            height=200,
            placeholder="Enter the prompt you want to analyze for potential risks...",
        )

        analyze_button = st.button("Analyze Prompt", type="primary")

        # Advanced options collapsible
        with st.expander("Advanced Options"):
            show_raw_response = st.checkbox("Show Raw API Response")

    # Results area
    if analyze_button and prompt:
        # Call the API
        with st.spinner("Analyzing prompt..."):
            api_response = call_goodfire_api(prompt)
            raw_features = call_goodfire_inspector(api_response)
            st.write("API Response received. Inspecting features...")

        # Format features
        aligned_features = format_features(raw_features)

        # Load the classifier
        labels, predictions = load_and_use_model(
            "binary_classifier.pt",
            aligned_features[
                [x for x in aligned_features.columns if x not in ["prompt", "response"]]
            ],
        )
        df_new_probs = pd.DataFrame(predictions, columns=["prob0", "prob1"])
        prob = df_new_probs["prob1"][0]

        # Predict risk level
        risk_level = predict_risk(prob)

        # Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Attack Probability", f"{prob:.1%}")

        with col2:
            st.metric("Risk Level", risk_level)


        # # Display feature importance
        if aligned_features is not None and not aligned_features.empty:
            st.subheader("Activated Features")

            # Extract feature columns (all columns except 'prompt' and 'response')
            feature_cols = [
                col
                for col in aligned_features.columns
                if col not in ["prompt", "response"]
            ]

            if feature_cols:
                # Create a DataFrame for visualization
                feature_data = pd.DataFrame(
                    {
                        "feature": feature_cols,
                        "activation": [
                            aligned_features[col].iloc[0] for col in feature_cols
                        ],
                    }
                )

                # Sort by activation value in descending order
                feature_data = feature_data.sort_values("activation", ascending=False)

                # Take top 10 features (or fewer if there are less than 10)
                top_n = min(10, len(feature_data))
                top_features = feature_data.head(top_n)

                # Create horizontal bar chart
                fig = px.bar(
                    top_features,
                    y="feature",
                    x="activation",
                    orientation="h",
                    title=f"Top {top_n} Activated Features",
                    labels={"activation": "Activation Strength", "feature": "Feature"},
                    color="activation",
                    color_continuous_scale=["green", "yellow", "red"],
                    text="activation",
                )

                # Improve layout
                fig.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                )

                # Format text
                fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")

                # Display chart
                st.plotly_chart(fig, use_container_width=True)

                # Also display as a table for detailed view
                st.subheader("All Feature Activations")
                st.dataframe(
                    feature_data,
                    column_config={
                        "feature": "Feature Name",
                        "activation": st.column_config.ProgressColumn(
                            "Activation Strength",
                            format="%.2f",
                            min_value=0,
                            max_value=feature_data["activation"].max() or 1,
                        ),
                    },
                    use_container_width=True,
                )
            else:
                st.info("No feature activation data available")
        else:
            st.info("No feature data available for visualization")

        # Display raw API response if requested
        if show_raw_response and api_response:
            with st.expander("Raw API Response"):
                st.json(api_response)

    # Sidebar with additional info
    with st.sidebar:
        st.header("About")
        st.write(
            "This dashboard uses the Goodfire API and a custom classifier to analyze prompts for potential red team attacks."
        )

        st.header("How to Use")
        st.write("1. Enter a prompt in the text area")
        st.write("2. Click 'Analyze Prompt'")
        st.write("3. Review the risk assessment and feature importance")

        st.header("Features")
        st.write("- Attack probability estimation")
        st.write("- Feature importance visualization")
        st.write("- Risk level assessment")

    # Footer
    st.markdown(
        '<div class="footer">Powered by Goodfire API and Custom Classifier</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
