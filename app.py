import streamlit as st
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import tempfile
import os
import openai

st.set_page_config(page_title="Notebook Checker", layout="centered")

st.title("📒 Jupyter Notebook Validator")
st.write("Upload a `.ipynb` file. This app will run it and explain any errors using OpenAI.")

uploaded_file = st.file_uploader("Upload a Jupyter Notebook", type=["ipynb"])

def explain_error_with_openai(traceback: str) -> str:
    try:
        openai.api_key = st.secrets["openai"]["api_key"]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": f"Explain the following Jupyter notebook error:\n\n{traceback}"}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Failed to get explanation from OpenAI: {e}"

def validate_notebook(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ipynb") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=600, allow_errors=True, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": "."}})

        errors = []
        for cell in nb.cells:
            if cell.cell_type == "code":
                for output in cell.get("outputs", []):
                    if output.output_type == "error":
                        errors.append("\n".join(output.get("traceback", [])))

        if errors:
            return "failure", "\n\n".join(errors)
        return "success", None

    except Exception as e:
        return "error", str(e)
    finally:
        os.remove(tmp_path)


if uploaded_file:
    st.info("📦 Executing notebook...")
    status, result = validate_notebook(uploaded_file.read())

    if status == "success":
        st.success("✅ Notebook executed successfully without errors.")
    else:
        st.error("❌ Errors found while executing the notebook.")
        st.code(result, language="python")

        st.markdown("### 💡 AI Explanation")
        explanation = explain_error_with_openai(result)
        st.info(explanation)
