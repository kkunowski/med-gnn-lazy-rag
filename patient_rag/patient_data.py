# patient_rag/patient_gnn_data.py
from med_gnn.med_gnn_data import DISEASES

PATIENTS = [
    {
        "patient_id": "pat001",
        "history": [
            {"date": "2025-09-01", "symptoms": {"fatigue": 1}, "labs": {"hemoglobin": 0.5}},
            {"date": "2025-09-10", "symptoms": {"fatigue": 1, "chest_pain": 0}, "labs": {"hemoglobin": 0.4}},
        ],
        "current": {
            "symptoms": {"fatigue": 1, "cough": 0, "chest_pain": 0},
            "labs": {"hemoglobin": 0.4, "CRP": 0.2, "troponin": 0.1},
        },
        "label": DISEASES.index("anemia"),
    },
    {
        "patient_id": "pat002",
        "history": [
            {"date": "2025-09-05", "symptoms": {"cough": 1}, "labs": {"CRP": 0.4}},
        ],
        "current": {
            "symptoms": {"cough": 1, "chest_pain": 0},
            "labs": {"CRP": 0.8, "troponin": 0.2},
        },
        "label": DISEASES.index("pneumonia"),
    },
]
