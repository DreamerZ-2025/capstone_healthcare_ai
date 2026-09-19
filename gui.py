import sys
import json
import requests
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QTextEdit, QLineEdit, 
                               QPushButton, QTabWidget, QMessageBox)
from PySide6.QtCore import Qt

# The URL of your running FastAPI server
API_URL = "http://127.0.0.1:8000"

class CapstoneGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Breast Cancer Prediction & AI Assistant")
        self.setGeometry(100, 100, 700, 600)

        # Main Tab Widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Create Tabs
        self.predict_tab = QWidget()
        self.ask_tab = QWidget()
        self.tabs.addTab(self.predict_tab, "🩺 Predict Tumor")
        self.tabs.addTab(self.ask_tab, "🤖 Ask AI Assistant")

        self.init_predict_tab()
        self.init_ask_tab()

    # --- Tab 1: Predict ---
    def init_predict_tab(self):
        layout = QVBoxLayout()
        
        # Instructions
        layout.addWidget(QLabel("Paste the 30 patient features as a JSON list below:"))
        
        # Input Text Box
        self.features_input = QTextEdit()
        self.features_input.setPlaceholderText("[17.99, 10.38, 122.8, ...]")
        layout.addWidget(self.features_input)
        
        # Load Sample Data Button
        self.sample_btn = QPushButton("Load Sample Patient Data")
        self.sample_btn.clicked.connect(self.load_sample_data)
        layout.addWidget(self.sample_btn)
        
        # Predict Button
        self.predict_btn = QPushButton("Predict")
        self.predict_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.predict_btn.clicked.connect(self.run_prediction)
        layout.addWidget(self.predict_btn)
        
        # Output Label
        layout.addWidget(QLabel("Prediction Result:"))
        self.predict_output = QTextEdit()
        self.predict_output.setReadOnly(True)
        layout.addWidget(self.predict_output)
        
        self.predict_tab.setLayout(layout)

    # --- Tab 2: Ask AI ---
    def init_ask_tab(self):
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Ask a question about the model's accuracy, bias, or architecture:"))
        
        # Question Input
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("e.g., How accurate is the model?")
        layout.addWidget(self.question_input)
        
        # Ask Button
        self.ask_btn = QPushButton("Ask AI")
        self.ask_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        self.ask_btn.clicked.connect(self.run_ask)
        layout.addWidget(self.ask_btn)
        
        # Output Label
        layout.addWidget(QLabel("AI Answer:"))
        self.ask_output = QTextEdit()
        self.ask_output.setReadOnly(True)
        layout.addWidget(self.ask_output)
        
        self.ask_tab.setLayout(layout)

    # --- Logic ---
    def load_sample_data(self):
        sample = "[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]"
        self.features_input.setText(sample)

    def run_prediction(self):
        try:
            # Parse the input string into a list of floats
            raw_text = self.features_input.toPlainText().strip()
            features_list = json.loads(raw_text)
            
            if len(features_list) != 30:
                QMessageBox.warning(self, "Input Error", "Please provide exactly 30 features.")
                return

            payload = {"features": features_list}
            response = requests.post(f"{API_URL}/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                formatted_result = json.dumps(result, indent=4)
                self.predict_output.setText(formatted_result)
            else:
                self.predict_output.setText(f"Error: {response.text}")
                
        except json.JSONDecodeError:
            QMessageBox.critical(self, "JSON Error", "Invalid JSON format. Please ensure it's a list of numbers like [1.0, 2.0, ...]")
        except requests.exceptions.ConnectionError:
            QMessageBox.critical(self, "Connection Error", "Could not connect to the API. Is the FastAPI server running?")

    def run_ask(self):
        question = self.question_input.text().strip()
        if not question:
            QMessageBox.warning(self, "Input Error", "Please enter a question.")
            return

        try:
            payload = {"question": question}
            response = requests.post(f"{API_URL}/ask", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                self.ask_output.setText(result.get("answer", "No answer found."))
            else:
                self.ask_output.setText(f"Error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            QMessageBox.critical(self, "Connection Error", "Could not connect to the API. Is the FastAPI server running?")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CapstoneGUI()
    window.show()
    sys.exit(app.exec())