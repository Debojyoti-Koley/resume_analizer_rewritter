import streamlit as st
from io import StringIO
# Assuming screening_op.py contains all the logic from the last correct script
from screening_op import rewrite_resume, calculate_scores

# --- 1. Initialize Session State ---
# These variables persist across reruns and are crucial for managing state
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False
if 'resume_content' not in st.session_state:
    st.session_state.resume_content = ""
if 'job_desc_content' not in st.session_state:
    st.session_state.job_desc_content = ""

# --- 2. Callback Function for the Button ---
def set_button_clicked():
    st.session_state.button_clicked = True

# --- 3. UI and Input Widgets ---
st.title("Resume Scorer and Rewriter")
st.write("Upload your resume and a job description to see how well they match, and get a rewritten resume to improve your chances.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Resume")
    resume_file = st.file_uploader(
        "Upload your resume (Text or PDF)", 
        type=['txt', 'pdf'], 
        key="resume_upload",
        disabled=st.session_state.button_clicked
    )
    if resume_file:
        st.session_state.resume_content = StringIO(resume_file.getvalue().decode("utf-8")).read()
    st.text_area(
        "Resume Content", 
        value=st.session_state.resume_content, 
        height=200,
        disabled=st.session_state.button_clicked,
        key="resume_text_area"
    )

with col2:
    st.subheader("Job Description")
    st.session_state.job_desc_content = st.text_area(
        "Paste the job description here", 
        value=st.session_state.job_desc_content,
        height=200, 
        disabled=st.session_state.button_clicked,
        key="job_desc_text_area"
    )

# --- 4. The Analyze Button ---
button_disabled = (
    st.session_state.button_clicked or
    not st.session_state.resume_content.strip() or
    not st.session_state.job_desc_content.strip()
)
st.button(
    "Analyze & Rewrite", 
    on_click=set_button_clicked,
    disabled=button_disabled
)

# --- 5. Conditional Output based on Button State ---
# This entire block only runs if the button has been clicked
if st.session_state.button_clicked:
    if not st.session_state.resume_content or not st.session_state.job_desc_content:
        st.error("Please upload a resume and paste a job description.")
        # Re-enable button if inputs are missing
        st.session_state.button_clicked = False
    else:
        # Calculate and display results
        st.subheader("Analysis Results")
        with st.spinner("Calculating scores..."):
            score_base, score_ft = calculate_scores(st.session_state.resume_content, st.session_state.job_desc_content)
            match_percentage = round(score_base * 100, 2)
            st.metric(label="Match Percentage", value=f"{match_percentage}%")
        
        # Conditional rewriting based on score
        if match_percentage >= 70:
            st.success("✅ Result: The resume is a great match! No need for rewriting.")
        elif match_percentage >= 60:
            st.warning("⚠️ Result: The resume is a good match but could be improved. Generating an improved version...")
            with st.spinner("Rewriting resume... This might take a while."):
                improved_resume = rewrite_resume(st.session_state.resume_content, st.session_state.job_desc_content)
                st.markdown("### Improved Resume")
                print("\n--- Improved Resume ---\n")
                print(improved_resume)
                st.text_area("Rewritten Resume", improved_resume, height=300)
        else:
            st.error("❌ Result: The resume is not a strong match for this job description.")

    # --- 6. The Reset Button ---
    st.write("---") 
    if st.button("Start Over"):
        # Reset the session state to initial values and rerun the script
        st.session_state.button_clicked = False
        st.session_state.resume_content = ""
        st.session_state.job_desc_content = ""
        st.rerun()