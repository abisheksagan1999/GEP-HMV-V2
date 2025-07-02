import streamlit as st
import pandas as pd
import re
import difflib
from sentence_transformers import SentenceTransformer, util
import torch

st.set_page_config(page_title="🛠️ HMV Historical Validator", layout="wide")
st.title("🛠️ HMV Historical Maintenance Validator")
st.markdown("""
This tool checks your current maintenance task (description + corrective action) against historical data to suggest a fair man-hour benchmark. 
Upload your Excel file with historical data and input your current maintenance details below.
""")

# Load NLP model for semantic matching
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

def normalize(text):
    text = str(text).upper()
    text = re.sub(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", "", text)  # Remove dates
    text = re.sub(r'[^A-Z0-9 ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

uploaded_file = st.file_uploader("📤 Upload your historical Excel file", type=["xlsx"])

desc_input = st.text_area("🔎 Enter Current Description (Discrepancy):")
corr_input = st.text_area("🔧 Enter Current Corrective Action:")

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        df = df.dropna(subset=["Description", "Corrective Action", "Hours"])

        df['Combined'] = df['Description'].astype(str) + " | " + df['Corrective Action'].astype(str)
        df['Combined_Norm'] = df['Combined'].apply(normalize)

        if desc_input and corr_input:
            current_combined = normalize(desc_input + " | " + corr_input)
            
            # Get embeddings
            df_embeddings = model.encode(df['Combined_Norm'].tolist(), convert_to_tensor=True)
            input_embedding = model.encode(current_combined, convert_to_tensor=True)
            
            # Compute similarities
            cosine_scores = util.pytorch_cos_sim(input_embedding, df_embeddings)[0]
            df['Similarity'] = cosine_scores.cpu().numpy()
            df_matches = df[df['Similarity'] > 0.90].sort_values(by='Similarity', ascending=False)

            # Exact match
            exact = df[df['Combined_Norm'] == current_combined]

            if not exact.empty:
                st.subheader("✅ Exact Match Found")
                mode_hours = exact['Hours'].mode()[0]
                st.success(f"Historical Hours: **{mode_hours}**")
                st.info(f"Suggested Fair Quote (99% of historic): **{int(mode_hours * 0.99)} hours**")
                st.dataframe(exact[['W/O', 'Year', 'Description', 'Corrective Action', 'Hours']])
            else:
                st.warning("❌ No exact match found.")

            if not df_matches.empty:
                st.subheader("🔍 Approximate Matches (>90% Similarity)")
                for _, row in df_matches.head(5).iterrows():
                    st.markdown(f"**🔹 Description:** {row['Description']}")
                    st.markdown(f"**🔧 Corrective Action:** {row['Corrective Action']}")
                    st.markdown(f"**🕒 Hours:** {row['Hours']}, **📅 Year:** {row['Year']}, **🔢 WO #:** {row['W/O']}")
                    st.markdown("---")
            else:
                st.info("No sufficiently similar historical records found.")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

else:
    st.info("⬆️ Please upload your Excel file to begin.")

# Optional add-on: Export approximate matches
if 'df_matches' in locals() and not df_matches.empty:
    download = st.download_button("⬇️ Download Approximate Matches", df_matches.to_csv(index=False), file_name="approximate_matches.csv")
