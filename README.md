# Building a Simple RAG System from Scratch


This repository contains a lightweight, local implementation of a Retrieval-Augmented Generation (RAG) pipeline. The system vectorizes text documents, indexes them using Facebook AI Similarity Search (FAISS) for efficient semantic retrieval, and leverages Google's Flan-T5 model to generate precise context-aware answers.

## Features

* **Semantic Text Embeddings:** Utilizes `sentence-transformers/all-MiniLM-L6-v2` to convert raw text into dense 384-dimensional vectors.
* **Efficient Vector Storage:** Indexes document embeddings with FAISS (`IndexFlatIP`) for high-performance similarity search via normalized inner product (Cosine Similarity).
* **Context-Driven Generation:** Employs the `google/flan-t5-base` instruction-tuned model to synthesize accurate answers based strictly on retrieved context, minimizing hallucinations.
* **Deterministic Configuration:** Uses a low temperature setting (0.3) during generation to prioritize factual alignment with the source text.

## Architecture Overview

1. **Ingestion & Encoding:** A collection of unstructured documents is passed through the embedding model to generate normalized vector representations.
2. **Indexing:** Vectors are stored in a FAISS flat index.
3. **Retrieval:** A user query is encoded into the same vector space, and the index searches for the top $k$ nearest document matches.
4. **Augmentation & Generation:** The query and retrieved documents are formatted into a structured prompt and passed to the text-to-text generation pipeline to produce the final answer.

## Prerequisites

Ensure you have Python 3.8+ installed along with the required libraries.

pip install sentence-transformers faiss-cpu transformers torch numpy

## Implementation Details

The pipeline is built across the following core steps:

### 1. Document EmbeddingDocuments are encoded and normalized to ensure that the inner product calculation corresponds directly to cosine similarity.Pythonfrom sentence_transformers import SentenceTransformer

documents = [
    "The capital of Canada is Ottawa.",
    "Python is a popular programming language for AI and machine learning.",
    "The Great Wall of China is visible from space is a myth.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "Transformers models are widely used for natural language processing tasks."
]

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
doc_embeddings = embedding_model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)

### 2. FAISS IndexingA flat inner-product index is initialized using the dimensionality of the generated embeddings.Pythonimport faiss

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(doc_embeddings)

### 3. Retrieval MechanismA helper function extracts the top $k$ relevant documents matching the semantic meaning of the query.Pythondef retrieve(query, top_k=2):
    query_emb = embedding_model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_emb, top_k)
    retrieved_docs = [documents[i] for i in indices[0]]
    return retrieved_docs

### 4. Generation Pipeline (RAG)The generation step combines the retrieval module with the LLM pipeline.Pythonfrom transformers import pipeline

generator = pipeline("text2text-generation", model="google/flan-t5-base")

def rag_pipeline(query):
    retrieved_docs = retrieve(query)
    context = " ".join(retrieved_docs)
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    
    response = generator(prompt, max_length=100, temperature=0.3)[0]['generated_text']
    
    print("Retrieved Documents:\n", retrieved_docs)
    print("\nModel Answer:\n", response)

## Usage Example
Run the pipeline by passing natural language queries to the main interface:
Pythonrag_pipeline("What is FAISS used for?")
#### Expected Output:
##### Retrieved Documents: ['FAISS is a library for efficient similarity search...', 'Transformers models...']
##### Model Answer: efficient similarity search and clustering of dense vectors

rag_pipeline("Where is the capital of Canada?")
#### Expected Output:
##### Retrieved Documents: ['The capital of Canada is Ottawa.', 'The Great Wall of China...']
##### Model Answer: Ottawa

## License
This project is open-source and available under the MIT License.
