# main.py
from med_gnn.med_gnn import MedGNN
from med_gnn.med_gnn_data import TRAIN, TEST, DISEASES
from med_gnn.med_gnn_utils import train_med_gnn, predict_med_gnn
from patient_rag.patient_data import PATIENTS
from patient_rag.patient_store import PatientStore
from utils import print_case_inputs, print_prediction

if __name__ == "__main__":
    # Train global model
    model = MedGNN()
    train_med_gnn(model, TRAIN, lr=0.03, epochs=100)

    # Prepare patient store
    store = PatientStore(max_events=30)
    for p in PATIENTS:
        pid = p["patient_id"]
        for event in p.get("history", []):
            store.append(pid, event)

    # Use one shared input for both tests
    case_input = PATIENTS[0]["current"]
    print_case_inputs(case_input, "INPUT DATA (Symptoms / Labs)")

    # Generic model prediction
    generic_probs = predict_med_gnn(model, case_input)
    print_prediction(generic_probs, "GENERIC MODEL RESULT")

    # Patient-personalized prediction
    msg = {"patient_id": "pat001", "current": case_input, "append_to_history": True}
    patient_probs = predict_med_gnn(model, msg, patient_store=store)
    print_prediction(patient_probs, "PATIENT-SPECIFIC RESULT")
