# utils.py
from med_gnn.med_gnn_data import DISEASES
import torch
import numpy as np

def print_case_inputs(case: dict, title: str):
    """Show current symptoms/labs/treatments."""
    print(f"\n{title}")
    for group in ["symptoms", "labs", "treatments"]:
        if group in case and case[group]:
            print(f"  {group.capitalize()}:")
            for name, val in case[group].items():
                print(f"    • {name:<20} = {val}")
    print("-" * 55)

def print_prediction(probs: torch.Tensor, title: str):
    """Readable prediction results."""
    arr = probs.detach().cpu().numpy()
    order = np.argsort(arr)[::-1]
    print(f"\n{title}")
    for i in order:
        print(f"  {DISEASES[i]:<22} → {arr[i] * 100:6.2f}%")
    print(f"predicted: {DISEASES[int(order[0])]}")
    print("=" * 55)