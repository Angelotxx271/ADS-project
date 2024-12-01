#Preliminary Step,load all the necessary libraries to proceed
from openai import OpenAI
from pdfminer.high_level import extract_text
import pytesseract
from pdf2image import convert_from_path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import re
import numpy as np
import os


#IMPORTANT:For the implemention of this program we will be using openai ChatGPT, in order to implement it we need to make use of an API key from openai, for the purpose of being
#able to fullfill this assignment, we are using our own API key, this step is neccesary since we will be implementing our classification mehtod using AI, since this implies
#a cost per each API call we make this script will try to make as few calls as possible, therefore we kindly ask making a responsible use of the code provided here and ask for
#it not to be shared and deleted after its use for the grading in the project.

client = OpenAI(api_key='') 

#Function to extract the pdf information, trying 2 methods if one fails
def extract_cv_content(pdf_path):
    """
    Extract structured content from the CV using pdfminer or Tesseract OCR.
    Args:
        pdf_path (str): Path to the CV PDF file.
    Returns:
        dict: A dictionary containing extracted CV sections.
    """
    try:
        pdf_text = extract_text(pdf_path)
        if pdf_text.strip():  
            return parse_cv_text(pdf_text)
    except Exception as e:
        print(f"Error using pdfminer: {e}")
    
    print("Falling back to OCR with Tesseract")
    images = convert_from_path(pdf_path)
    pdf_text = ''
    for image in images:
        pdf_text += pytesseract.image_to_string(image) + '\n'

    return parse_cv_text(pdf_text)

#Function to parse text into fields for better understanding
def parse_cv_text(text):
    """
    Parse the extracted text into structured fields.
    Args:
        text (str): Extracted text from the CV.
    Returns:
        dict: Structured fields like summary, education, experience, etc.
    """
    cv_data = {
        "summary": extract_section("SUMMARY", text),
        "education": extract_section("EDUCATION", text, include_bullet_points=True),
        "experience": extract_section("EXPERIENCE", text, include_bullet_points=True),
        "extracurricular_activities": extract_section("EXTRACURRICULAR ACTIVITIES", text, include_bullet_points=True),
        "languages": extract_section("LANGUAGES", text),
        "additional_information": extract_section("ADDITIONAL INFORMATION", text)
    }
    return cv_data

#Extract every section and its corresponding text
def extract_section(section_title, text, include_bullet_points=False):
    """
    Extract a specific section from the CV text using regex.
    Args:
        section_title (str): The title of the section to extract.
        text (str): The full text of the CV.
        include_bullet_points (bool): Whether to include bullet points in the extracted content.
    Returns:
        str: Extracted section content or "Not Found" if missing.
    """
    pattern = rf"{section_title}\s*(.*?)(?=\n[A-Z ]+\n|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return "Not Found"

    section_content = match.group(1).strip()

    if include_bullet_points:
        lines = [line.strip() for line in section_content.split("\n") if line.strip()]
        return "\n".join(lines)

    return re.sub(r"\n\s+", " ", section_content).strip()



def get_chat_response(prompt):
    """
    Calls the GPT-4 API with a prompt and returns the response.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500,  
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

def classify_cv(cv_content):
    """
    Uses GPT-4 to classify a CV into the most suitable bachelor programs.
    Args:
        cv_content (dict): Extracted content from the CV.
    Returns:
        str: GPT-4's recommendations for the top 3 programs.
    """
    # Hash table for bachelor programs
    bachelor_programs = {
        "BBA": "Bachelor in Business Administration. Focuses on leadership, management, and organizational skills.",
        "BBSS": "Behavior & Social Sciences. Focuses on understanding human behavior, psychology, and social impact.",
        "LLB": "Bachelor in Law. Focuses on legal studies, leadership, and international law.",
        "PPLE": "Philosophy, Politics, Laws & Economics. Focuses on ethics, strategic thinking, and interdisciplinary approaches.",
        "BIE": "Bachelor in Economics. Focuses on macroeconomics, microeconomics, and global financial systems.",
        "BIR": "Bachelor in International Relations. Covers global politics, multilingual skills, and volunteering.",
        "BIH": "Bachelor in Humanities. Focuses on history, culture, and human thought.",
        "BCDM": "Communication & Digital Media. Focuses on content creation, marketing, and modern media.",
        "BAS": "Architectural Studies. Focuses on architecture, urban planning, and design.",
        "BID": "Bachelor in Design. Focuses on creativity, graphic design, and visual arts.",
        "BIF": "Bachelor in Fashion. Focuses on design, textiles, and the fashion industry.",
        "BCSAI": "Computer Science & Artificial Intelligence. Focuses on STEM, data science, AI, and coding.",
        "BAM": "Applied Mathematics. Focuses on statistics, mathematical modeling, and problem-solving.",
        "BDBA": "Data & Business Analytics. Combines business intelligence with data analytics and decision-making.",
        "BESS": "Environmental Sciences & Sustainability. Focuses on climate change, sustainability, and environmental studies."
    }
    cv_summary = "\n".join([f"{key.capitalize()}: {value}" for key, value in cv_content.items()])

    
    prompt = f"""
    You are a career advisor. Based on the following CV content, recommend the top 3 most suitable bachelor programs.

    CV Content:
    {cv_summary}

    Available Programs:
    {', '.join([f"{program}: {description}" for program, description in bachelor_programs.items()])}

    Provide the top 3 programs in order of suitability and briefly explain why each program is a good fit.
    """

    response = get_chat_response(prompt)
    if len(response) < 3:  
        response += "\n(Warning: Response may be truncated. Check the max_tokens setting.)"
    return response


import streamlit as st
from tempfile import NamedTemporaryFile


# Set page configuration 
st.set_page_config(page_title="University Program Classifier", layout="wide")

# Centered header
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>University Program Classifier</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <p style='text-align: center; font-size: 18px;'>
        Upload your CV as a PDF file, and our AI will classify it into the top 3 most suitable bachelor programs for you.
    </p>
    """,
    unsafe_allow_html=True,
)

# File upload section
st.markdown(
    "<h3 style='text-align: center;'>Upload Your CV (PDF Format)</h3>",
    unsafe_allow_html=True,
)
uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

# Main functionality
if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_pdf_path = temp_file.name

    # Extract content from the CV
    st.info("Extracting content from your CV. Please wait...")
    cv_content = extract_cv_content(temp_pdf_path)

    if cv_content:
        st.success("CV content extracted successfully!")

        # Display extracted content in an expandable section
        with st.expander("View Extracted CV Content", expanded=False):
            st.markdown("<h4>Extracted CV Data</h4>", unsafe_allow_html=True)
            for section, content in cv_content.items():
                st.markdown(f"**{section.upper()}**:")
                st.write(content)

        # Perform classification using GPT-4
        st.info("Classifying CV. This may take a few seconds...")
        recommendations = classify_cv(cv_content)

        # Format recommendations properly
        if isinstance(recommendations, list):
            recommendations = [str(item).strip() for item in recommendations if item.strip()]
        else:
            recommendations = [str(recommendations).strip()]

        # Display GPT-4 recommendations
        st.markdown(
            "<h3 style='text-align: center; color: #4CAF50;'>Recommended Programs</h3>",
            unsafe_allow_html=True,
        )

        if recommendations:
            for idx, program in enumerate(recommendations, 1):
                st.markdown(
                    f"<p style='text-align: justify; font-size: 16px; margin-left: 20px;'>"
                    f"**{idx}.** {program}</p>",
                    unsafe_allow_html=True,
                )
        else:
            st.error("No valid recommendations found.")
    else:
        st.error(
            "Failed to extract content from the uploaded CV. Please try again with a valid CV format."
        )
else:
    st.warning("Please upload a CV to get started.")

# Footer
st.markdown(
    """
    <hr style='border: none; border-top: 1px solid #ccc;'>
    <p style='text-align: center; font-size: 12px; color: #aaa;'>
        Developed for Algorithms and Data Structures Project. Powered by GPT-4.
    </p>
    """,
    unsafe_allow_html=True,
)
