import streamlit as st
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from tempfile import NamedTemporaryFile
from typing import Optional, Dict
import google.generativeai as genai
import os

# ---- PAGE SETUP ----
st.set_page_config(page_title="Notebook Executor + Gemini", layout="centered")
st.title("📘 Jupyter Notebook Error Checker + 🤖 Gemini Suggestions")

# ---- GEMINI CONFIG ----
GEMINI_API_KEY = st.text_input("🔐 Enter your Gemini API Key", type="password")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

    def get_gemini_fix(error_msg, traceback_text):
        try:
            prompt = f"""I have a Jupyter Notebook that failed with the following error:

Error Message:
{error_msg}

Traceback:
{traceback_text}

Please:
1. Explain what this error means.
2. Suggest a fix for the code that caused this.
3. Mention if this looks like AI-generated code or not.
"""
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Gemini API Error: {e}"

# ---- FUNCTION TO CHECK NOTEBOOK EXECUTION ----
def check_notebook_executability(notebook_path: str, timeout: int = 600, save_executed: Optional[str] = None) -> Dict[str, Optional[str]]:
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_content = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=timeout, allow_errors=True)
        ep.preprocess(notebook_content, {'metadata': {'path': '.'}})

        errors = [
            output for cell in notebook_content.cells if cell.cell_type == 'code'
            for output in cell.get('outputs', []) if output.output_type == 'error'
        ]

        if save_executed:
            with open(save_executed, 'w', encoding='utf-8') as f:
                nbformat.write(notebook_content, f)

        if errors:
            error_messages = [f"{e['ename']}: {e['evalue']}" for e in errors]
            tracebacks = ["\n".join(e.get('traceback', [])) for e in errors]

            return {
                'status': 'failure',
                'error_message': "; ".join(error_messages),
                'traceback': "\n\n".join(tracebacks)
            }

        return {
            'status': 'success',
            'error_message': None,
            'traceback': None
        }

    except Exception as e:
        return {
            'status': 'failure',
            'error_message': f"Unexpected error: {e}",
            'traceback': None
        }

# ---- FILE UPLOAD ----
uploaded_file = st.file_uploader("📤 Upload a `.ipynb` notebook", type="ipynb")

if uploaded_file is not None:
    with NamedTemporaryFile(delete=False, suffix=".ipynb") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.info("⏳ Executing notebook...")
    result = check_notebook_executability(tmp_path, save_executed="executed_output.ipynb")

    if result['status'] == 'success':
        st.success("✅ All cells executed successfully.")
        with open("executed_output.ipynb", "rb") as f:
            st.download_button("📥 Download Executed Notebook", f, file_name="executed_output.ipynb")
    else:
        st.error("❌ Errors during execution.")
        st.write("**Error Message:**")
        st.code(result['error_message'])

        if result['traceback']:
            st.write("**Traceback:**")
            st.code(result['traceback'])

            if GEMINI_API_KEY:
                st.subheader("💡 Gemini Suggestion")
                with st.spinner("Thinking with Gemini..."):
                    gemini_response = get_gemini_fix(result['error_message'], result['traceback'])
                    st.markdown(gemini_response)
            else:
                st.warning("🔐 Enter your Gemini API key above to get AI suggestions.")
