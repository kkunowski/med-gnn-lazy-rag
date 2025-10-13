# patient_rag/patient_store.py
from collections import defaultdict, deque

class PatientStore:
    
    def __init__(self, max_events: int = 20):
        self.max_events = max_events
        self._hist = defaultdict(lambda: deque(maxlen=max_events))

    def append(self, patient_id: str, event: dict):
        self._hist[patient_id].append(event)

    def get_history(self, patient_id: str):
        return list(self._hist.get(patient_id, []))
