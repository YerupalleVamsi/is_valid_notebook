import streamlit as st
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import tempfile
import os
import openai

st.set_page_config(page_title="Notebook Validator", layout="centered")

st.title("✅ Jupyter Notebook Validator")
st.write("Upload a `.ipynb` file. This app will execute it and check for errors. If there are issues, OpenAI GPT will explain them.")

uploaded_file = st.file_uploader("Upload your `.ipynb` file", type=["ipynb"])

def get_openai_explanation(traceback: str) -> str:
    try:
        openai.api_key = st.secrets["openai"]["api_key"]
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=[{
                "role": "user",
                "content": f"I got this error when running a Jupyter notebook. Please explain:\n\n{traceback}"
            }],
            max_tokens=500,
            temperature=0.3
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Failed to get response from OpenAI API: {e}"

def check_notebook(notebook_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ipynb") as tmp:
        tmp.write(notebook_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=600, kernel_name="python3", allow_errors=True)
        ep.preprocess(nb, {"metadata": {"path": "."}})

        errors = []
        for cell in nb.cells:
            if cell.cell_type == "code":
                for output in cell.get("outputs", []):
                    if output.output_type == "error":
                        errors.append("\n".join(output.get("traceback", [])))

        if errors:
            return "failure", "\n\n".join(errors)
        else:
            return "success", None

    except Exception as e:
        return "crash", str(e)

    finally:
        os.remove(tmp_path)


if uploaded_file is not None:
    with st.spinner("Running notebook..."):
        status, output = check_notebook(uploaded_file.read())

    if status == "success":
        st.success("✅ Notebook ran successfully with no errors.")
    else:
        st.error("❌ Notebook execution failed.")
        st.markdown("### Raw Error Traceback")
        st.code(output or "Unknown error", language="bash")

        st.markdown("### 💡 AI Explanation (OpenAI GPT):")
        explanation = get_openai_explanation(output or "Unknown error")
        st.info(explanation)
