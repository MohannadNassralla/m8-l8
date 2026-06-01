# Setup & Engineering Notes: Multi-Modal Search Pipeline

This document details the configuration, schema design choices, and deployment runbook for the Stack Exchange Technical Q&A retrieval engine using Weaviate (v3 Client architecture) and Sentence-Transformers.

---

## Architecture Overview

The system implements a dual-engine architecture capable of running isolated inverted-index queries, dense semantic vector matches, and linearly fused hybrid matches.


+-------------------------------------------------------+
   |                  User Search Query                    |
   +-------------------------------------------------------+
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
    +-------------------+             +-------------------+
    |  Inverted Index   |             |   Dense Model     |
    |  (BM25 Search)    |             | (all-MiniLM-L6-v2)|
    +-------------------+             +-------------------+
              |                                 |
              | Keyword Scores                  | Vector Space Maps
              v                                 v
   +-------------------------------------------------------+
   |         Linear Fusion Layer ($\alpha = 0.52$)          |
   +-------------------------------------------------------+
                               |
                               v
   +-------------------------------------------------------+
   |             Top-K De-duplicated Results               |
   +-------------------------------------------------------+


---

## Environment Setup

### 1. Prerequisite Packages
Ensure your virtual environment (`.venv`) is fully initialized and contains the necessary dependencies:
```bash
pip install weaviate-client==3.26.2 sentence-transformers numpy pytest

# Verify cluster status
curl http://localhost:8080/v1/.well-known/ready