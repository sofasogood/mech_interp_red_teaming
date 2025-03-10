# Mech Interp Red Teaming

A dashboard for analyzing and testing prompts for potential red team attacks using mechanistic interpretability and machine learning.
<img width="1014" alt="image" src="https://github.com/user-attachments/assets/effe9707-89ba-4736-8ddd-9fd0b62c6dc7" />
<img width="1128" alt="image" src="https://github.com/user-attachments/assets/a089f6d3-8a6f-40e7-bc44-beb26a321829" />
<img width="1128" alt="image" src="https://github.com/user-attachments/assets/208e1de2-fbf5-4f86-8494-fc0731bbbd9a" />


## Overview

This tool provides a streamlined interface for evaluating the risk level of potential red team prompts. It uses a combination of:

1. The Goodfire API to analyze prompt responses
2. A custom binary classifier trained on mechanistic interpretability data
3. A visual dashboard for real-time prompt testing and risk assessment

## Features

- **Prompt Testing**: Enter prompts and get immediate risk analysis
- **Attack Probability**: Quantitative assessment of attack success likelihood
- **Risk Level Categorization**: Low, Medium, or High risk classification
- **Machine Learning Backend**: Trained on mechanistic interpretability data to identify patterns in successful attacks

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- Streamlit
- Pandas, NumPy, Plotly
- Goodfire API access

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mech-interp-red-teaming.git
   cd mech-interp-red-teaming
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root and add:
   ```
   GOODFIRE_API_KEY=your_api_key_here
   ```

## Usage

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. Enter a prompt in the text area and click "Analyze Prompt"

4. Review the risk assessment results, including:
   - Attack probability
   - Risk level
   - Feature importance (if available)

## Technical Details

### Architecture

The application consists of three main components:

1. **Frontend**: Streamlit-based dashboard (`app.py`)
2. **API Integration**: Goodfire API client for LLM response generation
3. **Classifier**: PyTorch neural network for risk prediction (`classifier.py`)

```mermaid
flowchart LR
    subgraph Frontend
        UI[Streamlit Dashboard]
        Viz[Data Visualization]
    end
    
    subgraph Backend
        API[API Integration]
        Classifier[ML Classifier]
        Features[Feature Extraction]
    end
    
    subgraph External
        GoodfireAPI[Goodfire API]
        Model[PyTorch Model]
    end
    
    UI --> API
    API --> GoodfireAPI
    GoodfireAPI --> Features
    Features --> Classifier
    Classifier --> Model
    Classifier --> Viz
    Viz --> UI
    
    classDef frontend fill:#f9f,stroke:#333,stroke-width:2px
    classDef backend fill:#bbf,stroke:#333,stroke-width:2px
    classDef external fill:#fbb,stroke:#333,stroke-width:2px
    
    class UI,Viz frontend
    class API,Classifier,Features backend
    class GoodfireAPI,Model external
```

### The Classifier

The binary classifier is a 3-layer MLP (Multi-Layer Perceptron) trained on mechanistic interpretability data from successful and unsuccessful red team attacks. It processes features extracted from LLM responses to predict attack success probability.

Key aspects:
- Input features are standardized before prediction
- The model uses dropout layers for regularization
- Training metrics include accuracy and loss curves

### Data Pipeline

1. User submits a prompt
2. Prompt is sent to Goodfire API
3. API response is inspected for key features
4. Features are processed and passed to the classifier
5. Risk assessment is displayed to the user

#### System Flow Diagram

```mermaid
flowchart TD
    A[User] -->|Enters Prompt| B[Streamlit UI]
    B -->|Submit| C[API Handler]
    C -->|Request| D[Goodfire API]
    D -->|LLM Response| E[Feature Extraction]
    E -->|Processed Features| F[PyTorch Classifier]
    F -->|Prediction| G[Risk Assessment]
    G -->|Results| B
    
    subgraph Dashboard
    B
    G
    end
    
    subgraph Backend Processing
    C
    E
    F
    end
    
    subgraph External Service
    D
    end
    
    classDef dashboard fill:#d0f0c0,stroke:#333,stroke-width:2px
    classDef backend fill:#c0e0f0,stroke:#333,stroke-width:2px
    classDef external fill:#f0d0c0,stroke:#333,stroke-width:2px
    
    class B,G dashboard
    class C,E,F backend
    class D external
```

## Development

### Model Training

To retrain the classifier on new data:

```bash
python classifier.py
```

This will save a new model to `binary_classifier.pt`.

### Adding New Features

To extend the dashboard with new features:
1. Modify `app.py` to include your new UI elements
2. Update the classifier if needed to handle new feature types
3. Test thoroughly with various prompt types

## License

[Your License Here]

## Acknowledgments

- Goodfire API for LLM access
- [Any other acknowledgments]
