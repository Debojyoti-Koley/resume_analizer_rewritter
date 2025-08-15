from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os

# ------------------ 1. Hugging Face Authentication ------------------
os.environ["HF_TOKEN"] = "YOUR_HUGGINGFACE_ACCESS_TOKEN"  # Replace with your HF token

# ------------------ 2. Load Mistral-7B Model ------------------
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
print("Loading Mistral-7B-Instruct-v0.2...")

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=os.environ["HF_TOKEN"])
model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=os.environ["HF_TOKEN"])
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ------------------ 3. Sample Resume & Job Description ------------------
resume_text = """
Experienced Python developer with 3 years of experience in Django, REST APIs, 
and MySQL. Worked on AWS deployments and CI/CD pipelines.
"""

job_text = """
We are looking for a backend engineer skilled in Python, Django, REST APIs, AWS,
and containerization using Docker. Experience in database optimization is a plus.
"""

# ------------------ 4. Function to Rewrite Resume ------------------
def rewrite_resume(resume, job_desc):
    prompt = f"""
    Rewrite the following resume so that it matches the style and keywords of this job description.
    Keep the meaning and experiences unchanged. Only adjust wording and phrasing.

    Job Description:
    {job_desc}

    Resume:
    {resume}

    Rewritten Resume:
    """
    result = pipe(prompt, max_length=500, do_sample=True, temperature=0.7)
    return result[0]['generated_text']

# ------------------ 5. Load Screening Models ------------------
print("Loading screening models...")
model_base = SentenceTransformer('all-MiniLM-L6-v2')
model_ft = SentenceTransformer('models/fine_tuned_screening')

# ------------------ 6. Calculate Scores ------------------
def calculate_scores(resume, job_desc):
    emb1_base = model_base.encode(resume, convert_to_tensor=True)
    emb2_base = model_base.encode(job_desc, convert_to_tensor=True)
    score_base = util.cos_sim(emb1_base, emb2_base).item()

    emb1_ft = model_ft.encode(resume, convert_to_tensor=True)
    emb2_ft = model_ft.encode(job_desc, convert_to_tensor=True)
    score_ft = util.cos_sim(emb1_ft, emb2_ft).item()

    return score_base, score_ft

# emb1_base = model_base.encode(resume_text, convert_to_tensor=True)
# emb2_base = model_base.encode(job_text, convert_to_tensor=True)
# score_base = util.cos_sim(emb1_base, emb2_base).item()

# emb1_ft = model_ft.encode(resume_text, convert_to_tensor=True)
# emb2_ft = model_ft.encode(job_text, convert_to_tensor=True)
# score_ft = util.cos_sim(emb1_ft, emb2_ft).item()

# ------------------ 7. Output Results ------------------
# print(f"\nBase model score: {score_base:.4f}")
# print(f"Fine-tuned model score: {score_ft:.4f}")

# score_base, score_ft = calculate_scores(resume_text, job_text)
# match_percentage = round(score_base * 100, 2)
# print(f"Match Percentage (Fine-tuned): {match_percentage}%")

# if match_percentage < 60:
#     print("\nResult: ❌ Reject")
# elif 60 <= match_percentage <= 70:
#     print("\nResult: ⚠️ Send for improvement")
#     improved_resume = rewrite_resume(resume_text, job_text)
#     print("\n--- Improved Resume ---\n")
#     print(improved_resume)
# else:
#     print("\nResult: ✅ Accept")
