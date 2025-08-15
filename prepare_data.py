import pandas as pd

df = pd.read_csv("data/resume_job_matching_dataset.csv")  # your combined dataset
df = df[['resume', 'job_description', 'match_score']]  # keep only needed columns

# Optional: Normalize score to 0–1
df['score'] = df['match_score'].apply(lambda x: float(x) / 100 if x > 1 else float(x))

# Save for training
df.to_csv("data/cleaned_pairs.csv", index=False)
print("Data preparation complete. Cleaned pairs saved to 'data/cleaned_pairs.csv'.")