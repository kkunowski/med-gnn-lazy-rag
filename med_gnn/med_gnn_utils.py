# med_gnn/med_gnn_utils.py
import torch
import torch.nn.functional as F
from torch import nn
from .med_gnn_graph import MedGraph

def train_med_gnn(model, train_cases, lr=0.03, epochs=150):
    g = model.graph if hasattr(model, "graph") else MedGraph()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(epochs):
        total = 0.0
        for case in train_cases:
            X = g.case_to_X(case)
            y = torch.tensor([case["label"]], dtype=torch.long)
            logits = model(X)  # uses base adj internally
            loss = loss_fn(logits.unsqueeze(0), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        if (ep + 1) % max(1, epochs // 10) == 0:
            pass

def predict_med_gnn(model, message: dict, patient_store=None, hist_decay=0.85):
    if "patient_id" in message:
        pid = message["patient_id"]
        current = message.get("current", {})
        history = message.get("history")
        if history is None and patient_store is not None:
            history = patient_store.get_history(pid)
        logits = model.forward_patient(pid, current=current, history=history, hist_decay=hist_decay)
        if message.get("append_to_history") and patient_store is not None:
            patient_store.append(pid, current)
        return torch.softmax(logits, dim=-1)
    else:
        logits = model.forward_case(message)
        return torch.softmax(logits, dim=-1)
