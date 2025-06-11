import streamlit as st
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import tempfile
import traceback
from dotenv import load_dotenv
import google.generativeai as genai

# Load Gemini API Key from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.warning("🔐 Please set your GEMINI_API_KEY in a `.env` file.")

st.set_page_config(page_title="Notebook Checker", layout="wide")

st.title("🧪 Notebook Executability Checker with Gemini AI")
st.markdown("Upload a `.ipynb` Jupyter notebook. This app will execute it and tell you if all cells run without errors. If not, Gemini will help explain the issues.")

uploaded_file = st.file_uploader("📤 Upload your .ipynb file", type=["ipynb"])

def check_notebook(notebook_bytes) -> dict:
    try:
        nb = nbformat.reads(notebook_bytes.decode("utf-8"), as_version=4)

        ep = ExecutePreprocessor(timeout=300, allow_errors=True)
        ep.preprocess(nb, {"metadata": {"path": "."}})

        errors = [
            output for cell in nb.cells if cell.cell_type == 'code'
            for output in cell.get('outputs', []) if output.output_type == 'error'
        ]

        if errors:
            return {
                'status': 'failure',
                'errors': errors,
                'executed_nb': nb
            }
        else:
            return {
                'status': 'success',
                'executed_nb': nb
            }

    except Exception as e:
        return {
            'status': 'crash',
            'error_message': str(e),
            'traceback': traceback.format_exc()
        }

def get_gemini_analysis(error_msgs: list) -> str:
    if not GEMINI_API_KEY:
        return "❌ Gemini API Key not set."

    try:
        model = genai.GenerativeModel("gemini-pro")
        error_text = "\n\n".join(error_msgs)
        prompt = f"""I ran a Jupyter notebook and got the following error(s):

{error_text}

Can you explain what caused these errors and how I might fix them?
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API error: {str(e)}"

if uploaded_file:
    notebook_bytes = uploaded_file.read()

    with st.spinner("🚀 Executing notebook..."):
        result = check_notebook(notebook_bytes)

    if result["status"] == "success":
        st.success("✅ All cells executed without errors!")
    elif result["status"] == "failure":
        st.error("❌ Errors were found during execution.")
        error_msgs = [f"{e.get('ename')}: {e.get('evalue')}" for e in result["errors"]]
        for i, e in enumerate(result["errors"]):
            st.code("".join(e.get("traceback", [])), language="python")

        # Gemini AI Analysis
        with st.spinner("🤖 Analyzing errors using Gemini..."):
            analysis = get_gemini_analysis(error_msgs)
            st.subheader("💡 Gemini Suggestions")
            st.markdown(analysis)
    else:
        st.error("⚠️ Failed to process notebook.")
        st.text(result.get("error_message", "Unknown error"))
        st.code(result.get("traceback", ""), language="python")
