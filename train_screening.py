# from sentence_transformers import SentenceTransformer, InputExample, losses
# from sentence_transformers import SentencesDataset, SentencesDataset, LoggingHandler
# from sentence_transformers import datasets
# import pandas as pd
# from torch.utils.data import DataLoader

# # 1. Load base model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # 2. Load dataset
# df = pd.read_csv("data/cleaned_pairs.csv")
# train_examples = [
#     InputExample(texts=[row['resume'], row['job_description']], label=row['score'])
#     for _, row in df.iterrows()
# ]

# # 3. Create DataLoader
# train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# # 4. Define loss function
# train_loss = losses.CosineSimilarityLoss(model)

# # 5. Fine-tune
# model.fit(
#     train_objectives=[(train_dataloader, train_loss)],
#     epochs=1,  # start small for testing
#     warmup_steps=100
# )

# # 6. Save fine-tuned model
# model.save('models/fine_tuned_screening')
# print("Fine-tuning complete. Model saved to 'models/fine_tuned_screening'.")

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd

# 1. Load base model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Load your dataset
df = pd.read_csv("data/resume_job_matching_dataset.csv")

# Make sure column names match
examples = []
for _, row in df.iterrows():
    examples.append(
        InputExample(
            texts=[row['resume'], row['job_description']],
            label=float(row['match_score'])  # should be between 0 and 1
        )
    )

# 3. Create dataloader
train_dataloader = DataLoader(examples, shuffle=True, batch_size=16)

# 4. Loss function (correct one for similarity scoring)
train_loss = losses.CosineSimilarityLoss(model=model)

# 5. Train
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=int(0.1 * len(train_dataloader)),
    show_progress_bar=True
)

# 6. Save model
model.save("models/fine_tuned_screening")

