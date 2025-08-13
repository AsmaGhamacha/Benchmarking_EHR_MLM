import os
import json
import sys
from tqdm import tqdm

# Allow absolute import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.utils.config import load_config
from src.preprocessing.parse_ehr_json import parse_ehr_bundle

cfg = load_config()

RAW_DIR = cfg["paths"]["raw_data"]
PROCESSED_DIR = cfg["paths"]["processed_data"]

os.makedirs(PROCESSED_DIR, exist_ok=True)



def extract_patient_text(ehr_data, cfg):
    grouped = parse_ehr_bundle(ehr_data, cfg)
    patient_info = grouped.get("Patient", [{}])[0]

    text = [
        f"Patient ID: {patient_info.get('id', 'N/A')}",
        f"Gender: {patient_info.get('gender', 'N/A')}",
        f"Birth Date: {patient_info.get('birthDate', 'N/A')}",
        ""
    ]

    # --- Conditions ---
    try:
        if cfg["extraction"].get("include_conditions", True):
            conditions = grouped.get("Condition", [])
            lines = []
            for c in conditions:
                code = c.get("code", {})
                name = code.get("text") or code.get("coding", [{}])[0].get("display", "Unknown Condition")
                status = c.get("clinicalStatus", {}).get("text", "")
                verified = c.get("verificationStatus", {}).get("text", "")
                onset = c.get("onsetDateTime", "")
                line = f"{name} | Status: {status} | Verification: {verified} | Onset: {onset}"
                lines.append(line)
            text.extend(["Diagnoses:"] + (["- " + l for l in lines] if lines else ["- None"]) + [""])
    except Exception as e:
        print(f"[ERROR] Failed to extract Condition: {e}")
        text.extend(["Diagnoses:", "- Extraction failed", ""])

    # --- Medications ---
    try:
        if cfg["extraction"].get("include_medications", True):
            medications = grouped.get("MedicationRequest", [])
            lines = []
            for m in medications:
                med_concept = m.get("medicationCodeableConcept", {})
                name = med_concept.get("text") or med_concept.get("coding", [{}])[0].get("display")
                dose = m.get("dosageInstruction", [{}])[0].get("text", "")
                status = m.get("status", "")
                intent = m.get("intent", "")
                authored = m.get("authoredOn", "")
                line = f"{name} | Dose: {dose} | Status: {status} | Intent: {intent} | Date: {authored}"
                if name:
                    lines.append(line)
            text.extend(["Medications:"] + (["- " + l for l in lines] if lines else ["- None"]) + [""])
    except Exception as e:
        print(f"[ERROR] Failed to extract MedicationRequest: {e}")
        text.extend(["Medications:", "- Extraction failed", ""])

    # --- Observations ---
    try:
        if cfg["extraction"].get("include_observations", True):
            observations = grouped.get("Observation", [])
            lines = []
            for o in observations:
                name = o.get("code", {}).get("text") or o.get("code", {}).get("coding", [{}])[0].get("display")
                value = o.get("valueQuantity", {}).get("value")
                unit = o.get("valueQuantity", {}).get("unit")
                date = o.get("effectiveDateTime", "")
                interp = o.get("interpretation", {}).get("text", "")
                if name and value and unit:
                    lines.append(f"{name}: {value} {unit} | Interpretation: {interp} | Date: {date}")
            text.extend(["Lab Results:"] + (["- " + l for l in lines] if lines else ["- None"]) + [""])
    except Exception as e:
        print(f"[ERROR] Failed to extract Observation: {e}")
        text.extend(["Lab Results:", "- Extraction failed", ""])

    # --- Procedures ---
    try:
        if cfg["extraction"].get("include_procedures", True):
            procedures = grouped.get("Procedure", [])
            lines = []
            for p in procedures:
                name = p.get("code", {}).get("text") or p.get("code", {}).get("coding", [{}])[0].get("display")
                date = p.get("performedDateTime", "")
                status = p.get("status", "")
                line = f"{name} | Date: {date} | Status: {status}"
                if name:
                    lines.append(line)
            text.extend(["Procedures:"] + (["- " + l for l in lines] if lines else ["- None"]) + [""])
    except Exception as e:
        print(f"[ERROR] Failed to extract Procedure: {e}")
        text.extend(["Procedures:", "- Extraction failed", ""])

    # --- Diagnostic Reports ---
    try:
        if cfg["extraction"].get("include_diagnostic_reports", True):
            reports = grouped.get("DiagnosticReport", [])
            lines = []
            for r in reports:
                name = r.get("code", {}).get("text") or r.get("code", {}).get("coding", [{}])[0].get("display")
                date = r.get("effectiveDateTime", "")
                conclusion = r.get("conclusion", "")
                status = r.get("status", "")
                line = f"{name} | Status: {status} | Date: {date} | Conclusion: {conclusion}"
                lines.append(line)
            text.extend(["Diagnostic Reports:"] + (["- " + l for l in lines] if lines else ["- None"]) + [""])
    except Exception as e:
        print(f"[ERROR] Failed to extract DiagnosticReport: {e}")
        text.extend(["Diagnostic Reports:", "- Extraction failed", ""])

    # --- Immunizations ---
    try:
        if cfg["extraction"].get("include_immunizations", True):
            immunizations = grouped.get("Immunization", [])
            lines = []
            for i in immunizations:
                vaccine = i.get("vaccineCode", {}).get("text") or i.get("vaccineCode", {}).get("coding", [{}])[0].get("display")
                date = i.get("occurrenceDateTime", "")
                source = i.get("primarySource", "")
                if vaccine:
                    lines.append(f"{vaccine} | Date: {date} | Source: {source}")
            text.extend(["Immunizations:"] + (["- " + l for l in lines] if lines else ["- None"]) + [""])
    except Exception as e:
        print(f"[ERROR] Failed to extract Immunization: {e}")
        text.extend(["Immunizations:", "- Extraction failed", ""])

    # --- Encounters ---
    try:
        if cfg["extraction"].get("include_encounters", True):
            encounters = grouped.get("Encounter", [])
            lines = []
            for e in encounters:
                status = e.get("status", "")
                visit_type = e.get("type", [{}])[0].get("text") or e.get("class", {}).get("code")
                period = e.get("period", {})
                start = period.get("start", "")
                end = period.get("end", "")
                lines.append(f"{visit_type} | Status: {status} | Period: {start} to {end}")
            text.extend(["Encounters:"] + (["- " + l for l in lines] if lines else ["- None"]) + [""])
    except Exception as e:
        print(f"[ERROR] Failed to extract Encounter: {e}")
        text.extend(["Encounters:", "- Extraction failed", ""])

    return "\n".join(text)




def process_all_files(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR):
    print(f"Scanning `{raw_dir}` for JSON files...")
    for root, _, files in os.walk(raw_dir):
        for file in tqdm(files):
            if file.endswith(".json"):
                raw_path = os.path.join(root, file)
                try:
                    with open(raw_path, "r") as f:
                        ehr_data = json.load(f)
                        patient_text = extract_patient_text(ehr_data, cfg)

                        # Save to file
                        patient_id = ehr_data.get("entry", [{}])[0].get("resource", {}).get("id", file)
                        save_path = os.path.join(processed_dir, f"{patient_id}.txt")
                        with open(save_path, "w") as out_f:
                            out_f.write(patient_text)
                except Exception as e:
                    print(f"[ERROR] Failed to process {file}: {e}")


if __name__ == "__main__":
    process_all_files()
    print(f"Done! Extracted text saved in `{PROCESSED_DIR}`")
