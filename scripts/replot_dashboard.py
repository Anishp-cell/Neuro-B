import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from scripts.run_digital_sphinx_audit import generate_audit_dashboard

def main():
    data_path = Path("outputs/digital_sphinx_audit.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bio_results = data["biological_results"]
    null_results = data["scrambled_null_results"]
    falsification_report = data["falsification_report"]

    out_file = Path("outputs/digital_sphinx_audit.png")
    generate_audit_dashboard(bio_results, null_results, falsification_report, out_file)
    print("Re-rendered", out_file)

if __name__ == "__main__":
    main()
