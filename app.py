import streamlit as st
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import tempfile
import traceback
import openai

# Load OpenAI API key
openai.api_key = st.secrets["openai"]

st.set_page_config(page_title="Jupyter Notebook Validator", layout="centered")

st.markdown("## 📒 Jupyter Notebook Validator")
st.markdown("Upload a `.ipynb` file. This app will run it and explain any errors using OpenAI.")

uploaded_file = st.file_uploader("Upload a Jupyter Notebook", type=["ipynb"])

if uploaded_file:
    with st.spinner("⚙️ Executing notebook..."):
        try:
            # Read the notebook
            nb = nbformat.read(uploaded_file, as_version=4)

            # Execute the notebook
            ep = ExecutePreprocessor(timeout=300, kernel_name="python3")

            with tempfile.TemporaryDirectory() as tmpdir:
                ep.preprocess(nb, {"metadata": {"path": tmpdir}})
            
            st.success("✅ Notebook executed successfully!")

        except Exception as e:
            st.error("❌ Errors found while executing the notebook.")
            tb = traceback.format_exc()
            st.code(tb)

            # Ask OpenAI to explain the error
            st.markdown("### 💡 AI Explanation")
            try:
                prompt = f"Explain the following Python traceback error and how to fix it:\n{tb}"
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                )
                explanation = response.choices[0].message.content
                st.info(explanation)
            except Exception as ai_error:
                st.error("❌ Failed to get explanation from OpenAI.")
                st.code(str(ai_error))

