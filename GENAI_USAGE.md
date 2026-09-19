# Generative AI Project

## 1. Automated Data Dictionary & EDA Summary
- **Tool Used:** ChatGPT (GPT-4) / OpenAI API
- **Prompt Strategy:** Provided the `.describe()` output of the dataset.
- **Outcome:** Generated a 15-row data dictionary and identified 3 key outliers (mean area, mean perimeter).
- **Time Saved:** Estimated 1.5 hours.

## 2. XAI Translation for Clinicians
- **Tool Used:** Google Gemini API
- **Implementation:** `src/genai_xai.py`
- **Outcome:** Converted SHAP values into plain-English clinical explanations. See `demo_video_xai.mp4`.

## 3. GenAI-Enhanced Application
- **Tool Used:** LangChain, FAISS, GPT-4
- **Implementation:** Added a `/ask` endpoint to the FastAPI app.
- **Outcome:** Doctors can ask questions about the model's performance and guidelines. See `app/main.py`.

## 4. Slide Deck Generation
- **Tool Used:** Gamma.app / ChatGPT
- **Prompt Strategy:** "Generate an outline for a 10-slide business presentation for hospital executives about an AI cancer prediction tool."
- **Outcome:** Used the generated outline as a baseline, then manually refined it with project-specific metrics.