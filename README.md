# Healthcare AI: Breast Cancer Prediction Capstone

📋 Project Overview
This project builds a robust Machine Learning model to predict the likelihood of malignant breast tumors based on 30 numerical features. The goal is to serve as a Clinical Decision Support tool that maximizes **Recall** (sensitivity) to ensure no malignant tumors are missed, while maintaining high overall accuracy.

The project covers the complete ML lifecycle: Problem Framing, EDA, Feature Engineering, Model Training, Ethical Bias Auditing, and Deployment via a **FastAPI REST API** with an integrated **Generative AI assistant** and a **PySide6 Desktop GUI**.

✨ Key Features
*   **High-Performance ML Model:** Achieved an ROC-AUC of **0.9954** and a Recall of **98%** for malignant tumors using Logistic Regression.
*   **Bias & Fairness Audit:** Implemented a proxy bias audit (Large vs. Small tumors) to evaluate disparate impact and model reliability across subgroups (Disparate Impact Ratio: 1.09).
*   **REST API Deployment:** Deployed a local FastAPI server for real-time predictions.
*   **Desktop GUI (PySide6):** Built a user-friendly desktop application to interact with the API without needing a web browser.
*   **GenAI Integration (Bonus):** Built a RAG (Retrieval-Augmented Generation) pipeline using **Google Gemini** and FAISS to answer user questions about the model's metrics, ethics, and clinical guidelines.

----------------------------------------------------------------------------------------------------


🛠️ 1. Setup & Installation

# Prerequisites
*   Python 3.10 or higher
*   Git
*   A Google Gemini API Key (Get one for free at [aistudio.google.com](https://aistudio.google.com))

# Step-by-Step Setup
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/DreamerZ-2025/capstone_healthcare_ai.git
    cd capstone_healthcare_ai

2.  Create and Activate a Virtual Environment:
    # use powershell
    # bash
        python -m venv venv
        .\venv\Scripts\activate

3.  Install Dependencies:
    # use powershell
    # bash
        pip install -r requirements.txt

4.  Set Up Environment Variables:
    # create .env
    GOOGLE_API_KEY=your-google-api-key-here

🚀 2. How to Run the Project

Step 1: Train the Machine Learning Models
    # use powershell
    # bash
        python -m src.train

Step 2: Run the Ethical Bias Audit
    # use powershell
    # bash
        python -m src.bias_audit

Step 3: Launch the FastAPI Backend (Terminal 1)
    # use powershell
    # bash
        uvicorn app.main:app --reload

Step 4: Launch the PySide6 Desktop GUI (Terminal 2)
    # use powershell
    # bash
        python gui.py

🔌 3. API Endpoints & GUI Features

    # Once the FastAPI server is running, navigate to http://127.0.0.1:8000/docs to test the endpoints  directly in your browser.
    # /predict (POST)
    # Accepts 30 numerical features and returns the prediction.
    # Sample Request Body:
        # json
            {
                "features": [
                    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
                ]
            }

    # /ask (POST)
    # A Generative AI endpoint (RAG) that answers questions based on the project's documentation.
    # Sample Request Body:
        # json
            {
                "question": "How accurate is the model?"
            }

🧠 4. Generative AI Usage 

=> This project implements a Retrieval-Augmented Generation (RAG) architecture to provide a conversational interface for the ML model's documentation.

* How it works:

    -> Document Embedding: Project documentation (model metrics, bias audit results, ethical disclaimers) is embedded using GoogleGenerativeAIEmbeddings.

    -> Vector Storage: Embeddings are stored in a FAISS vector database for fast similarity search.

    -> Retrieval & Generation: When a user submits a question to the /ask endpoint, the system retrieves the most relevant context from the vector store and passes it to the gemini-2.5-flash LLM. The LLM then generates an accurate, grounded response based strictly on the provided context.

    -> Engineering Note: During testing, the free tier of the Google Gemini API occasionally experienced quota limits. To ensure a seamless demo experience, the /ask endpoint includes a mock fallback that returns pre-validated answers if the external API is unavailable. This demonstrates a mature, resilient engineering approach to handling third-party API dependencies.


📜 License

=> This project is for educational purposes as part of the Post Graduate Diploma in Artificial Intelligence and Machine Learning Capstone Project.

    
    🚀 Final Step: Push to GitHub
    Once you have saved this updated `README.md`, push it to your repository:

    # bash
        git add README.md requirements.txt
        git commit -m "Add comprehensive README with GUI and GenAI documentation"
        git push