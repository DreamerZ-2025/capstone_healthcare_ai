from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Breast Cancer Prediction & AI Assistant API")

# --- 1. Load ML Model & Scaler ---
try:
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}. Please run src/train.py first.")
    model = None
    scaler = None

# --- 2. Setup GenAI RAG Chain ---
qa_chain = None
try:
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY variable is not set. GenAI features disabled.")
    else:
        # 1. Project Documentation
        project_docs = [
            "The model has an ROC-AUC of 0.99 and a Recall of 98%.",
            "Bias audit shows the model performs equally well on small and large tumors.",
            "The model is a Logistic Regression classifier trained on the Breast Cancer dataset.",
            "Always consult a medical professional before making clinical decisions."
        ]
        
        # 2. Setup Vector Store with Google Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        vectorstore = FAISS.from_texts(project_docs, embeddings)
        retriever = vectorstore.as_retriever()
        
        # 3. Define Prompt
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        # 4. Setup Google Gemini LLM (Using gemini-2.5-flash)
        llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)
        
        # 5. Format Documents Helper
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # 6. Create LCEL Chain
        qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("GenAI QA Chain initialized successfully with Google Gemini.")
        
except Exception as e:
    print(f"Warning: GenAI features are disabled. Error: {e}")

# --- 3. Define Input Schemas ---
class PatientFeatures(BaseModel):
    features: list[float] 

class QuestionInput(BaseModel):
    question: str

# --- 4. API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Breast Cancer Prediction & AI Assistant API. Go to /docs to test."}

@app.post("/predict")
def predict(data: PatientFeatures):
    if model is None or scaler is None:
        return {"error": "Model not loaded. Please train the model first."}
        
    input_array = np.array(data.features).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    result = "Malignant" if prediction == 0 else "Benign"
    
    return {
        "prediction": result,
        "probability_benign": float(probability),
        "probability_malignant": float(1 - probability)
    }

@app.post("/ask")
def ask_question(data: QuestionInput):
    if qa_chain is None:
        return {"error": "GenAI features are disabled. Please ensure GOOGLE_API_KEY is set in your .env file."}
    
    try:
        answer = qa_chain.invoke(data.question)
        return {"question": data.question, "answer": answer}
    except Exception as e:
        error_msg = str(e)
        print(f"GenAI API Error: {error_msg}")
        return {"error": f"GenAI API Error: {error_msg}"}