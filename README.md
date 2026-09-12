# uk-immigration-guidance-rag
Evaluating chunking, embedding, retrieval and RAG architecture choices for question answering over the UK Immigration rules and guidance.

# Immigration Rules RAG evaluation

Evaluation of Structure-Aware Retrieval Strategies for RAG-Based Question Answering over the UK Immigration Rules.

## What this repository is

Code, manifests and prompts for a controlled comparison of retrieval-augmented generation (RAG)
design choices over the UK Immigration Rules: three chunking strategies, two embedding models,
four retrieval strategies (keyword, dense, hybrid, hybrid with reranking) and three ways of
assembling retrieved context (flat, parent-child, cross-reference expansion). Every configuration
is scored on the same evaluation set of about 150 questions with paragraph-level gold labels.

Research question: how do chunking, embedding, retrieval strategy and RAG architecture choices
affect answer accuracy and faithfulness when a RAG system is used over the UK Immigration Rules?

## Data

- UK Immigration Rules, Home Office, fetched through the GOV.UK Content API and frozen as a dated snapshot.
- Home Office caseworker guidance for five visa routes (Skilled Worker, Student, Graduate, Visitor, Family)
  from the GOV.UK visas and immigration operational guidance collection.
- Contains public sector information licensed under the Open Government Licence v3.0.

Data files are not stored in this repository. Raw snapshots and processed datasets live on
Warwick OneDrive; `manifests/` holds the file lists and checksums that identify each snapshot.

## Ethics

WMG Student Ethics Form v3, approved 11 September 2026. No human participants, no personal data.
Data is shared only within the research team. See `.kiro/steering/data-compliance.md` for the
rules the code follows.
