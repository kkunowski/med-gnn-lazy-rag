# LazyRAG-DAG: Adaptive Medical Graph Neural Network Framework

A lightweight experimental framework that explores **adaptive message passing in Graph Neural Networks (GNNs)** for medical reasoning tasks.

This project implements a **2-layer GNN** that predicts medical conditions from **symptom**, **lab**, and **treatment** nodes.  
It supports *personalized inference* by adding a patient-specific node whose history influences predictions.

---

## Overview

Traditional GNNs recompute embeddings for all nodes on every pass.  
This framework demonstrates a simplified **“lazy propagation”** idea, where only the relevant portions of the graph are updated when new information (e.g., patient data) arrives.

> *Rather than performing full graph recomputation, the model updates context only for connected or recently changed nodes.*

This approach reduces redundant computation and provides a stepping stone toward more dynamic or retrieval-augmented graph systems.

---

## Architecture

### 1. Two-Layer GNN
- **Layer 1 – Neighborhood Encoding:**  
  Aggregates feature information from connected medical entities (symptoms, labs, treatments).
- **Layer 2 – Context Refinement:**  
  Updates each node’s embedding using the aggregated neighborhood context and applies a nonlinear transformation.

### 2. Graph Topology
- The base graph connects:
  - **Symptoms → Diseases**
  - **Labs → Diseases**
  - **Treatments → Diseases**
- Each node stores:
  - **Feature vector** (numeric embedding)
  - **Group type** (symptom, lab, treatment, disease)
  - **Optional patient connections** (added dynamically)

### 3. Lazy / Adaptive Update Mechanism
- Patient-specific graphs are built **on demand**:
  - A temporary **patient node** connects to current observations.
- The rest of the graph remains unchanged, allowing **partial context reuse** between updates.

This mimics “lazy” propagation — only recomputing the affected subgraph when patient information changes.

---

## Key Components

| Module | Description |
|---------|--------------|
| `med_gnn/med_gnn_graph.py` | Builds the static medical knowledge graph (symptoms, labs, treatments, diseases). |
| `med_gnn/med_gnn.py` | Implements the 2-layer GNN using PyTorch. |
| `patient_rag/patient_store.py` | Maintains rolling patient history with decayed weights. |
| `patient_rag/patient_data.py` | Provides toy example patient records for demonstration. |
| `main.py` | Entry point for training and evaluation (generic + patient-personalized runs). |

---

## Future Extensions
While this prototype uses numeric features only, it can be extended to a **true RAG-DAG** system by integrating:
- Text/document embeddings as node content.
- Cosine-similarity–based change detection (`Δ > ε`).
- On-demand message forwarding controlled by lazy flags.
- Dynamic node addition for new knowledge entries.
