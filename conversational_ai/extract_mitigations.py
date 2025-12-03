import json
import pandas as pd

# Load MITRE ATT&CK JSON file
with open("enterprise-attack.json", "r", encoding="utf-8") as f:
    data = json.load(f)

mitigations = []
for obj in data["objects"]:
    if obj["type"] == "course-of-action":  # Mitigation entry
        mitigation_id = obj.get("external_references", [{}])[0].get("external_id", "")
        name = obj.get("name", "")
        description = obj.get("description", "")
        mitigations.append([mitigation_id, name, description])

# Save to CSV
df = pd.DataFrame(mitigations, columns=["ID", "Name", "Description"])
df.to_csv("mitre_mitigations.csv", index=False, encoding="utf-8")

print("✅ Saved mitigations to mitre_mitigations.csv")
print(f"Total mitigations extracted: {len(mitigations)}")
print("\nFirst few entries:")
print(df.head())
