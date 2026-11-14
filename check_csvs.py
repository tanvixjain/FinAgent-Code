import pandas as pd
import os

files = ["Budgeting_Personal_Finance_QA.csv", "Government_Schemes_Banking.csv"]

for f in files:
    if os.path.exists(f):
        print(f"--- {f} ---")
        df = pd.read_csv(f)
        print("Columns:", df.columns.tolist())
        print("Sample Data (1 row):")
        print(df.head(1).to_dict(orient='records'))
        print("\n")
    else:
        print(f"File {f} not found")
