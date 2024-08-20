from datasets import load_dataset
import pandas as pd

subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["economics"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering", "health"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology", "business"],
}

# Convert the list of dictionaries to a DataFrame
classify_df = pd.DataFrame(columns=['text'])

def combine_columns(row):
    res = cat + ", " + broader_cat
    return '<human>: question: ' + row['question'] + ' answer choices: ' + str({i: v for i,v in enumerate(row['choices'])}) + f"\n<bot>: {res}"

for cat in subcategories:
    dataset = load_dataset("cais/mmlu", cat)
    df = dataset['test'].to_pandas()
    broader_cat = subcategories[cat][0]
    if broader_cat in categories["STEM"]:
        broader_cat = "STEM"
    if broader_cat in categories["social sciences"]:
        broader_cat = "social_sciences"
    if broader_cat in categories["humanities"]:
        broader_cat = "humanities"
    df['text'] = df.apply(combine_columns, axis=1)
    df = df[['text']]
    classify_df = pd.concat([classify_df, df], ignore_index=True)

classify_df.to_json(f'classify3.jsonl', orient='records', lines=True)