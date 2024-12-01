# ADS-project
Program to classify a student profile(CV) to 3 most suitable bachelor degress in IE
---

## Main Functions

The platform has two main functionalities:

1. **Upload and Analyze CVs**: Students can upload their CVs in PDF format, and the platform extracts key information to recommend the most suitable bachelor's programs.
2. **View Extracted Data**: Students can verify the extracted information from their CV and ensure accuracy before viewing recommendations.

---

## Installation

To set up and run the program, ensure the following:

1. **Python Programming Language**  
   - Supported versions: 3.7 and above
     
2. **Required Libraries**  
   Install the necessary libraries using:
   ```bash
   pip install -r requirements.txt

The requirements.txt file includes all required dependencies.


## Usage

Install the required dependencies:

pip install streamlit
pip install openai
pip install pytesseract
pip install pdfminer.six
pip install pandas
pip install numpy
pip install scikit-learn


Run the Streamlit app:

    streamlit run CV_classification.py

    Open the link provided by Streamlit in your browser.

How It Works
Workflow

    Upload Your CV
        Upload a PDF version of your CV.
        The application extracts and processes the data using either pdfminer or Tesseract OCR.

    View Extracted Data
        The application parses the CV and categorizes information into sections like education, experience, and skills.
        This data is displayed for user verification.

    View Program Recommendations
        The AI ranks the top 3 programs that match the user's profile, with brief explanations for each choice.


Powered by:

    OpenAI GPT-4
    Streamlit Framework

Disclaimer

This program uses an OpenAI API key, which incurs costs per API call. Please use the program responsibly and avoid sharing API keys.


---



   
# CREDITS
The authors for this project are:   
Angel Acorda
Ilia Artamonov  
Claudia Guerrero  
Carla Gonzalez
Sahitya Anand
  
