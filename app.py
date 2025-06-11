import streamlit as st
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import openai
import os
from io import BytesIO

st.set_page_config(page_title="Jupyter Notebook Validator", layout="centered")

st.title("📒 Jupyter Notebook Validator")
st.markdown("Upload a `.ipynb` file. This app will run it and explain any errors using OpenAI.")

uploaded_file = st.file_uploader("Upload a Jupyter Notebook", type=["ipynb"])

def check_notebook(file):
    try:
        nb = nbformat.read(file, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": "."}})
        return True, "✅ All cells executed successfully."
    except Exception as e:
        return False, str(e)

def explain_error(error_text):
    openai.api_key = st.secrets["openai"]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You're a Python error explainer."},
            {"role": "user", "content": f"Explain this error in detail:\n{error_text}"}
        ]
    )
    return response['choices'][0]['message']['content']

if uploaded_file:
    with st.spinner("🔍 Executing notebook..."):
        success, result = check_notebook(BytesIO(uploaded_file.read()))

    if success:
        st.success(result)
    else:
        st.error("❌ Errors found while executing the notebook.")
        st.code(result)
        st.subheader("💡 AI Explanation")
        try:
            explanation = explain_error(result)
            st.info(explanation)
        except Exception as e:
            st.error(f"❌ Failed to get explanation from OpenAI: {e}")
            st.markdown(
                "[📄 Instructions for setting secrets](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management)"
            )

