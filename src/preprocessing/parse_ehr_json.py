# src/preprocessing/parse_ehr_json.py

def parse_ehr_bundle(ehr_data, cfg):
    grouped = {
        "Patient": [],
        "Condition": [],
        "MedicationRequest": [],
        "Observation": [],
        "Procedure": [],
        "CarePlan": [],
        "DiagnosticReport": [],
        "Encounter": [],
        "Immunization": []
    }

    for entry in ehr_data.get("entry", []):
        resource = entry.get("resource", {})
        r_type = resource.get("resourceType")
        if r_type in grouped:
            grouped[r_type].append(resource)

    return grouped