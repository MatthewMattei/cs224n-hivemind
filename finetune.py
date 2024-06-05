import os
from together import Together

client = Together(api_key="af55a5d60e08e7064287b3099b7c22c18366a4bee70bcc4e25beb839a40ce8c2")

resp = client.files.upload(file="stem.jsonl") # uploads a file
filesUploaded = client.files.list()
print(filesUploaded)

# resp = client.fine_tuning.create(
#   training_file = 'file-daa17011-01c6-4b7d-844e-9d49b675993f',
#   model = 'meta-llama/Meta-Llama-3-8B',
#   n_epochs = 4,
#   n_checkpoints = 1,
#   batch_size = 8,
#   learning_rate = 1e-5,
# )

# 1) other - file-daa17011-01c6-4b7d-844e-9d49b675993f - ft-86f8d1cf-1ebd-4316-a113-0c19f2cb8074 - mdmattei@stanford.edu/Meta-Llama-3-8B-2024-06-04-23-20-24
# 2) humanities - file-8b3e99a8-d4b1-4268-8181-e245be02bb67 - ft-a14e2609-a8f6-4fe2-b6bc-f76235c10920 - mdmattei@stanford.edu/Meta-Llama-3-8B-2024-06-04-23-25-37
# 3) socialscience - file-452311a4-2114-41f1-a2d1-ae0097bcdec4 - ft-f065990c-f64c-4dc4-bce1-eb585f97c379 - mdmattei@stanford.edu/Meta-Llama-3-8B-2024-06-04-23-26-33