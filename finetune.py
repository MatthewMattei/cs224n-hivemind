from together import Together
from keys import TOGETHER_API_KEY

client = Together(api_key=TOGETHER_API_KEY)

resp = client.files.upload(file="classify3.jsonl")
filesUploaded = client.files.list()
print(filesUploaded)

resp = client.fine_tuning.create(
  training_file = 'file-ee660f4a-afab-4b78-a0e8-903fa0600593',
  model = 'meta-llama/Meta-Llama-3-8B',
  n_epochs = 4,
  n_checkpoints = 1,
  batch_size = 8,
  learning_rate = 1e-5,
)