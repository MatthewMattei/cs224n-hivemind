import pandas as pd

PARQUET_DATA_FILE = "auxiliary_data.parquet"
CSV_DATA_FILE = "auxiliary_data.csv"

# Convert parquet file to csv file
def parquet_to_csv():
    data = pd.read_parquet(PARQUET_DATA_FILE)
    data.to_csv(CSV_DATA_FILE, index=False)

def tasks_to_subject():
    data = pd.read_csv(CSV_DATA_FILE)

    # Clear the subject of the data point
    data['subject'] = ""

    # Collected sub-categories into four main categories
    SOCIAL_SCIENCE = [
        "management", "marketing", "human_aging", "professional_accounting", "econometrics", 
        "public_relations", "high_school_macroeconomics", "us_foreign_policy", "professional_psychology", 
        "high_school_microeconomics", "sociology", "security_studies", "high_school_psychology", 
        "human_sexuality", "high_school_government_and_politics", "high_school_geography"
    ]
    HUMANITIES = [
        "high_school_european_history", "formal_logic", "prehistory", "philosophy", "high_school_world_history", 
        "high_school_us_history", "world_religions", "moral_scenarios", "logical_fallacies", "jurisprudence", 
        "professional_law", "business_ethics", "international_law", "moral_disputes"
    ]
    STEM = [
        "anatomy", "college_medicine", "professional_medicine", "medical_genetics", "nutrition", "clinical_knowledge",
        "conceptual_physics", "machine_learning", "high_school_mathematics", "high_school_computer_science", 
        "high_school_biology", "college_computer_science", "high_school_chemistry", "elementary_mathematics", 
        "computer_security", "college_mathematics","college_biology", "high_school_statistics", "electrical_engineering",
        "high_school_physics", "college_physics", "college_chemistry", "astronomy"
    ]

    # Change the subject into the category
    data.loc[data['task'].isin(STEM), 'subject'] = "STEM"
    data.loc[data['task'].isin(HUMANITIES), 'subject'] = "humanities"
    data.loc[data['task'].isin(SOCIAL_SCIENCE), 'subject'] = "social sciences"

    # Delete the task column
    data = data.drop(columns = ['task'])
    
    # Update the auxiliary_data.csv
    data.to_csv(CSV_DATA_FILE, index=False)

def separate_tasks():
    STEM_FILE = "stem.csv"
    HUMANITIES_FILE = "humanities.csv"
    SOCIAL_SCIENCE_FILE = "social_science.csv"

    data = pd.read_csv(CSV_DATA_FILE)

    data[data['subject'] == "STEM"].to_csv(STEM_FILE, index=False)
    data[data['subject'] == "humanities"].to_csv(HUMANITIES_FILE, index=False)
    data[data['subject'] == "social sciences"].to_csv(SOCIAL_SCIENCE_FILE, index=False)

separate_tasks()