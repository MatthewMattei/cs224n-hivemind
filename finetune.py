import os
from together import Together

client = Together(api_key="af55a5d60e08e7064287b3099b7c22c18366a4bee70bcc4e25beb839a40ce8c2")

# resp = client.files.upload(file="classify3.jsonl") # uploads a file
# filesUploaded = client.files.list()
# print(filesUploaded)

resp = client.fine_tuning.create(
  training_file = 'file-ee660f4a-afab-4b78-a0e8-903fa0600593',
  model = 'meta-llama/Meta-Llama-3-8B',
  n_epochs = 4,
  n_checkpoints = 1,
  batch_size = 8,
  learning_rate = 1e-5,
)

# 1) other - file-daa17011-01c6-4b7d-844e-9d49b675993f - ft-86f8d1cf-1ebd-4316-a113-0c19f2cb8074 - mdmattei@stanford.edu/Meta-Llama-3-8B-2024-06-04-23-20-24-4c9b70b7
# 2) humanities - file-8b3e99a8-d4b1-4268-8181-e245be02bb67 - ft-a14e2609-a8f6-4fe2-b6bc-f76235c10920 - mdmattei@stanford.edu/Meta-Llama-3-8B-2024-06-04-23-25-37-12c540c5
# 3) socialscience - file-452311a4-2114-41f1-a2d1-ae0097bcdec4 - ft-f065990c-f64c-4dc4-bce1-eb585f97c379 - mdmattei@stanford.edu/Meta-Llama-3-8B-2024-06-04-23-26-33-e15174ab
# 4) stem - file-f33b5646-db26-4e97-8eda-f830a4a52442 - ft-644c95e4-378f-4eb6-92b8-15263c2d099e - mdmattei@stanford.edu/Meta-Llama-3-8B-2024-06-05-00-23-38-c6149bf9
# 5) classifier - file-26cf65b5-fad7-4bc7-91de-adceb1339c9c - ft-93b19d19-6521-47dc-aa48-1ffe7172ca1b - mdmattei@stanford.edu/Meta-Llama-3-8B-2024-06-05-03-45-44
# 6) CHAD - file-7462b77c-15d7-4dde-88d5-7c131994301e - ft-f47a4f1b-fa7f-42db-8ce0-467de233f517 - mdmattei@stanford.edu/Meta-Llama-3-8B-2024-06-05-04-36-18
# 7) classifier2 - file-57c7bc94-5919-40b9-82b1-076432888dd5 - ft-3e371956-9e25-43ee-b351-c14b84e037da - mdmattei@stanford.edu/Meta-Llama-3-8B-2024-06-06-21-14-24-9a675604
# 8) classifier3 - file-ee660f4a-afab-4b78-a0e8-903fa0600593 - ft-0351cd09-383b-45d6-8bd9-762afb2026b8 - mdmattei@stanford.edu/Meta-Llama-3-8B-2024-06-07-03-50-36-affd94b3