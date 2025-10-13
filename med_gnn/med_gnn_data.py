from .med_gnn_graph import MedGraph

graph = MedGraph()
IDX = graph.idx
DISEASES = [n for n in graph.node_table["diseases"]]  # names in order
DISEASE_TO_CLASS = {name: i for i, name in enumerate(DISEASES)}

TRAIN = [
    {
        "symptoms": {"fatigue": 1, "cough": 0, "chest_pain": 0},
        "labs": {"hemoglobin": 0.4, "CRP": 0.2, "troponin": 0.1},
        "label": DISEASE_TO_CLASS["anemia"],
    },
    {
        "symptoms": {"fatigue": 0, "cough": 1, "chest_pain": 0},
        "labs": {"hemoglobin": 0.9, "CRP": 0.8, "troponin": 0.2},
        "label": DISEASE_TO_CLASS["pneumonia"],
    },
    {
        "symptoms": {"fatigue": 0, "cough": 0, "chest_pain": 1},
        "labs": {"hemoglobin": 0.9, "CRP": 0.2, "troponin": 0.9},
        "label": DISEASE_TO_CLASS["myocardial_infarction"],
    },
]

TEST = [
    {
        "name": "anemia-like",
        "symptoms": {"fatigue": 1, "cough": 0, "chest_pain": 0},
        "labs": {"hemoglobin": 0.35, "CRP": 0.15, "troponin": 0.1},
    },
]
