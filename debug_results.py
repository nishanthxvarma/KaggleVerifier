import os, re, ast

path = "simulation_results_final.txt"
if not os.path.exists(path):
    print("Log file not found.")
    exit(1)

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Improved regex to capture the whole Context dict
matches = re.finditer(r"Context: (\{.*?\})", content, re.DOTALL)

for i, match in enumerate(matches):
    m = match.group(1)
    # The string representation of CI might have line breaks or weird stuff, let's clean it up basically
    # Or just replace newlines in m
    m = m.replace("\n", " ").replace("\r", " ")
    try:
        # Use ast.literal_eval
        d = ast.literal_eval(m)
        print(f"\n--- TEST {i+1} ---")
        print(f"Type: {d.get('dataset_type')}")
        print(f"Prob: {d.get('raw_score')} -> Final (check simulation log)")
        print(f"Reasons: {d.get('calibration_reasons')}")
        print(f"CI: {d.get('ci_lower')} - {d.get('ci_upper')}")
    except Exception as e:
        print(f"Error parsing test {i+1}: {e}")
        # print("Snippet:", m[:100])
