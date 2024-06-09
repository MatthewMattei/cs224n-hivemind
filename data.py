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
    "global_facts": ["other"],
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
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
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
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other": ["other", "business", "health"],
}

stem_df = pd.DataFrame(columns=['text'])
socialscience_df = pd.DataFrame(columns=['text'])
humanities_df = pd.DataFrame(columns=['text'])
other_df = pd.DataFrame(columns=['text'])

def combine_columns(row):
    return '<human>: question: ' + row['question'] + ' answer choices: ' + str({i: v for i,v in enumerate(row['choices'])})

for cat in subcategories:
    dataset = load_dataset("cais/mmlu", cat)
    df = dataset['test'].to_pandas()
    df['text'] = df.apply(combine_columns, axis=1)
    df = df[['text']]
    broader_cat = subcategories[cat][0]
    if broader_cat in categories["STEM"]:
        stem_df = pd.concat([stem_df, df], ignore_index=True)
    if broader_cat in categories["social sciences"]:
        socialscience_df = pd.concat([socialscience_df, df], ignore_index=True)
    if broader_cat in categories["humanities"]:
        humanities_df = pd.concat([humanities_df, df], ignore_index=True)
    if broader_cat in categories["other"]:
        other_df = pd.concat([other_df, df], ignore_index=True)

stem_random_sample = stem_df.sample(n=50, random_state=42)
stem_df.drop(stem_random_sample.index)
stem_random_sample.to_json("stem_validation.jsonl", orient='records', lines=True)
stem_df.to_json(f'stem.jsonl', orient='records', lines=True)

other_random_sample = other_df.sample(n=50, random_state=42)
other_df.drop(other_random_sample.index)
other_random_sample.to_json("other_validation.jsonl", orient='records', lines=True)
other_df.to_json(f'other.jsonl', orient='records', lines=True)

socialscience_random_sample = socialscience_df.sample(n=50, random_state=42)
socialscience_df.drop(socialscience_random_sample.index)
socialscience_random_sample.to_json("socialscience_validation.jsonl", orient='records', lines=True)
socialscience_df.to_json(f'socialscience.jsonl', orient='records', lines=True)

humanities_random_sample = humanities_df.sample(n=50, random_state=42)
humanities_df.drop(humanities_random_sample.index)
humanities_random_sample.to_json("humanities_validation.jsonl", orient='records', lines=True)
humanities_df.to_json(f'humanities.jsonl', orient='records', lines=True)
