from sentence_transformers import SentenceTransformer, util
from ctransformers import AutoModelForCausalLM as CTransformersAutoModel

# ------------------ 1. Load the Rewrite Model (tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf) ------------------
# We will use the Mistral 7B Instruct model as it is lightweight and performs well. Not using as it takes more memory.
model_id = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
model_file = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
print(f"Loading {model_id} with optimizations...")

try:
    # Use ctransformers to load the GGUF model
    llm = CTransformersAutoModel.from_pretrained(
        model_file,
        model_type="llama", # specify the model type
        local_files_only=True
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    # Exit or handle the error gracefully if the model cannot be loaded
    exit()

# ------------------ 2. Sample Resume & Job Description ------------------
# resume_text = """
# Experienced Python developer with 3 years of experience in Django, REST APIs, 
# and MySQL. Worked on AWS deployments and CI/CD pipelines.
# """

# job_text = """
# We are looking for a backend engineer skilled in Python, Django, REST APIs, AWS,
# and containerization using Docker. Experience in database optimization is a plus.
# """

# ------------------ 3. Function to Rewrite Resume with Post-Processing ------------------
def rewrite_resume(resume, job_desc):
    """
    Rewrites a resume to match a job description using a language model.
    It includes post-processing to clean the output.
    """
    prompt = f"""
    Rewrite the following resume so that it matches the style and keywords of this job description.
    Keep the meaning and experiences unchanged. Only adjust wording and phrasing.

    Job Description:
    {job_desc}

    Resume:
    {resume}

    Rewritten Resume:
    """
    try:
        # Use the llm.generate() method from ctransformers
        rewritten_resume = llm(prompt, max_new_tokens=500, temperature=0.7)
        
        # Post-processing to remove the prompt part
        start_marker = "### Rewritten Resume:"
        if start_marker in rewritten_resume:
            rewritten_resume = rewritten_resume.split(start_marker, 1)[-1].strip()
        
        return rewritten_resume.strip()
    except Exception as e:
        print(f"An error occurred during resume rewriting: {e}")
        return "Failed to rewrite the resume."

# ------------------ 4. Load Screening Models ------------------
print("Loading screening models...")
try:
    model_base = SentenceTransformer('all-MiniLM-L6-v2')
    # Note: Ensure that 'models/fine_tuned_screening' is a valid path to your fine-tuned model
    model_ft = SentenceTransformer('models/fine_tuned_screening') 
except Exception as e:
    print(f"An error occurred while loading SentenceTransformer models: {e}")
    # Handle the error gracefully if the models cannot be loaded
    exit()

# ------------------ 5. Calculate Scores ------------------
# emb1_base = model_base.encode(resume_text, convert_to_tensor=True)
# emb2_base = model_base.encode(job_text, convert_to_tensor=True)
# score_base = util.cos_sim(emb1_base, emb2_base).item()

# emb1_ft = model_ft.encode(resume_text, convert_to_tensor=True)
# emb2_ft = model_ft.encode(job_text, convert_to_tensor=True)
# score_ft = util.cos_sim(emb1_ft, emb2_ft).item()
def calculate_scores(resume, job_desc):
    emb1_base = model_base.encode(resume, convert_to_tensor=True)
    emb2_base = model_base.encode(job_desc, convert_to_tensor=True)
    score_base = util.cos_sim(emb1_base, emb2_base).item()

    emb1_ft = model_ft.encode(resume, convert_to_tensor=True)
    emb2_ft = model_ft.encode(job_desc, convert_to_tensor=True)
    score_ft = util.cos_sim(emb1_ft, emb2_ft).item()

    return score_base, score_ft

# ------------------ 6. Output Results and Conditional Rewriting ------------------
# score_base, score_ft = calculate_scores(resume_text, job_text)
# print(f"\nBase model score: {score_base:.4f}")
# print(f"Fine-tuned model score: {score_ft:.4f}")

# match_percentage = round(score_base * 100, 2)
# print(f"Match Percentage (Fine-tuned): {match_percentage}%")

# if match_percentage >= 70:
#     print("\nResult: ✅ Accept")
# elif match_percentage >= 60:
#     print("\nResult: ⚠️ Send for improvement")
#     improved_resume = rewrite_resume(resume_text, job_text)
#     print("\n--- Improved Resume ---\n")
#     print(improved_resume)
# else:
#     print("\nResult: ❌ Reject")
