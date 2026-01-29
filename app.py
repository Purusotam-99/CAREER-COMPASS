import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import PyPDF2 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Career Compass", page_icon="🧭", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("career_data.csv")
    return df

df = load_data()

# Extract all unique skills from the database
all_skills = set(",".join(df['Skills']).replace('"', '').split(','))
all_skills = sorted([s.strip() for s in all_skills if s.strip()])

# --- HEADER ---
st.markdown("<h1 style='text-align: center;'>🧭 AI Career Compass</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload your resume to automatically detect your skills, or manually configure your profile.</p>", unsafe_allow_html=True)
st.divider()

# --- 1. RESUME PARSER (MAIN AREA) ---
st.subheader("📄 Upload Resume")
uploaded_file = st.file_uploader("Upload your CV (PDF) to auto-fill skills:", type=["pdf"])

extracted_skills = []

if uploaded_file is not None:
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        resume_text = ""
        for page in reader.pages:
            resume_text += page.extract_text()
        
        # Keyword Matching against Database
        resume_text_lower = resume_text.lower()
        for skill in all_skills:
            if skill.lower() in resume_text_lower:
                extracted_skills.append(skill)
        
        if extracted_skills:
            st.success(f"✅ AI detected {len(extracted_skills)} skills from your resume!")
        else:
            st.warning("No specific technical skills found in resume. Please add them manually below.")
            
    except Exception as e:
        st.error(f"Error reading PDF: {e}")

# --- 2. TECHNICAL SKILLS (MAIN AREA) ---
st.subheader("🛠️ Technical Skills")
# Use extracted skills as the default selection
default_skills = extracted_skills if extracted_skills else []

selected_skills = st.multiselect(
    "Verify or Add Skills:", 
    all_skills, 
    default=default_skills
)

# --- 3. SIDEBAR: PERSONAL DETAILS ---
with st.sidebar:
    st.header("👤 Personal Profile")
    st.caption("Refine your recommendations with these filters.")
    
    experience = st.selectbox("Experience Level", ["Beginner (Student)", "Intermediate (1-3 yrs)", "Expert (3+ yrs)"])
    
    st.divider()
    st.subheader("📚 Academic Scores")
    math_score = st.slider("Mathematics", 0, 100, 75)
    code_score = st.slider("Programming/CS", 0, 100, 70)
    
    st.divider()
    personality = st.radio("Personality Type", ["Introvert", "Extrovert", "Ambivert/Any"])
    
    st.divider()
    interest_options = ["AI", "Web", "Data", "Design", "Security", "Business", "Gaming", "People", "Research"]
    selected_interests = st.multiselect("Core Interests", interest_options, default=["AI", "Data"])

# --- 4. MATCHING ENGINE & RESULTS ---
st.divider()

if st.button("🚀 Analyze Career Paths", type="primary", use_container_width=True):
    if not selected_skills:
        st.error("Please select at least one skill to proceed!")
    else:
        final_results = []
        
        for index, row in df.iterrows():
            # A. SKILL MATCH
            job_skills = [s.strip() for s in row['Skills'].split(',')]
            common_skills = set(selected_skills).intersection(set(job_skills))
            skill_match_pct = (len(common_skills) / len(job_skills)) * 100
            
            # B. INTEREST MATCH
            job_interests = [i.strip() for i in row['Interests'].split(',')]
            common_interests = set(selected_interests).intersection(set(job_interests))
            interest_bonus = len(common_interests) * 15 
            
            # C. TOTAL SCORE CALCULATION
            total_score = skill_match_pct + interest_bonus
            
            # D. PENALTIES
            if math_score < row['Min_Math']: total_score -= 10
            if code_score < row['Min_Code']: total_score -= 10
            
            # E. BONUS
            if personality != "Ambivert/Any" and personality == row['Personality']:
                total_score += 5
                
            final_score = max(0, min(100, total_score))
            final_results.append(final_score)
            
        df['Final_Score'] = final_results
        top_jobs = df.sort_values(by='Final_Score', ascending=False).head(3)
        
        # --- DISPLAY RESULTS ---
        st.success("🎉 Analysis Complete! Here are your Top 3 Career Matches:")
        cols = st.columns(3)
        
        for i, (index, row) in enumerate(top_jobs.iterrows()):
            with cols[i]:
                st.markdown(f"### {i+1}. {row['Job Title']}")
                
                match_val = row['Final_Score']
                if match_val > 80: color = "green"
                elif match_val > 50: color = "orange"
                else: color = "red"
                
                st.markdown(f"**Match:** :{color}[{match_val:.1f}%]")
                st.progress(min(100, int(match_val)))
                
                st.write(f"💰 **Salary:** {row['Salary_Range']}")
                st.write(f"📈 **Demand:** {row['Trend_Growth']}")
                
                job_skills = [s.strip() for s in row['Skills'].split(',')]
                missing = [s for s in job_skills if s not in set(selected_skills)]
                
                if missing:
                    st.warning(f"**Missing:** {', '.join(missing[:3])}...")
                else:
                    st.success("You have the core skills!")
                    
                with st.expander("📚 Learning Roadmap"):
                    st.write("1. Master the missing skills.")
                    st.write("2. Build 1-2 capstone projects.")
                    if "AI" in row['Job Title']: st.info("Focus on Python & Math.")
                    if "Web" in row['Job Title']: st.info("Focus on React & APIs.")

        # --- DEMAND CHART ---
        st.divider()
        st.subheader("📊 Industry Demand Trend (Next 5 Years)")
        years = [2024, 2025, 2026, 2027, 2028]
        fig = go.Figure()
        
        for index, row in top_jobs.iterrows():
            base_growth = 5 if row['Trend_Growth'] == "Stable" else 10 if row['Trend_Growth'] == "High" else 15
            growth = [100 + (base_growth * i) + np.random.randint(-2, 2) for i in range(5)]
            fig.add_trace(go.Scatter(x=years, y=growth, mode='lines+markers', name=row['Job Title']))
            
        fig.update_layout(title="Projected Job Market Growth Index", xaxis_title="Year", yaxis_title="Demand Index (Baseline=100)")
        st.plotly_chart(fig, use_container_width=True)