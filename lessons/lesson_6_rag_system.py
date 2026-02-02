"""
Lesson 6: Build a Simple RAG System
====================================

RAG = Retrieval Augmented Generation

1. RETRIEVAL: Find relevant documents from your knowledge base
2. AUGMENTED: Add those documents to the prompt
3. GENERATION: LLM generates answer using the context

This lesson builds a RAG system you can extend with your personal data.

Structure:
- documents.json: Your knowledge base (add your personal data here!)
- Embeddings: We'll use sentence-transformers (free, runs locally)
- Retrieval: Cosine similarity search
- Generation: We'll show the prompt (you can connect to Claude/GPT later)
"""

import numpy as np
import json
import os

# =============================================================================
# PART 1: Understanding RAG
# =============================================================================

print("=" * 60)
print("LESSON 6: BUILDING A RAG SYSTEM")
print("=" * 60)
print()

print("-" * 40)
print("WHAT IS RAG?")
print("-" * 40)
print("""
Problem: LLMs don't know YOUR personal information.
         They have general knowledge, not your notes/docs/data.

Solution: RAG - give the LLM relevant context before asking.

Without RAG:
  User: "What did I decide about the project timeline?"
  LLM: "I don't have access to your personal information."

With RAG:
  1. Search YOUR documents for "project timeline"
  2. Find: "Meeting notes: Decided to launch in Q2..."
  3. Give this context to LLM
  4. LLM: "Based on your notes, you decided to launch in Q2..."
""")

# =============================================================================
# PART 2: Create Your Knowledge Base
# =============================================================================

print("-" * 40)
print("STEP 1: CREATE KNOWLEDGE BASE")
print("-" * 40)
print()

# Sample documents - REPLACE THESE WITH YOUR OWN DATA!
sample_documents = [
    {
        "id": "1",
        "title": "About Me",
        "content": "I am a software engineer interested in machine learning. I started learning ML in January 2025. My goal is to build AI-powered applications.",
        "metadata": {"type": "personal", "date": "2025-01-01"}
    },
    {
        "id": "2",
        "title": "Project Ideas",
        "content": "Project idea 1: Build a personal AI assistant that knows my schedule and preferences. Project idea 2: Create a code review tool using LLMs. Project idea 3: Develop a RAG system for my notes.",
        "metadata": {"type": "ideas", "date": "2025-01-15"}
    },
    {
        "id": "3",
        "title": "Learning Progress",
        "content": "Completed lessons on linear classifiers, neural networks, and embeddings. Currently learning about RAG systems. Next: want to learn about transformers and attention mechanisms.",
        "metadata": {"type": "progress", "date": "2025-01-20"}
    },
    {
        "id": "4",
        "title": "Technical Notes - Embeddings",
        "content": "Embeddings convert text to vectors. Similar text has similar vectors. This enables semantic search. Key formula: similarity = cosine(embedding1, embedding2).",
        "metadata": {"type": "notes", "date": "2025-01-19"}
    },
    {
        "id": "5",
        "title": "Meeting Notes",
        "content": "Discussed project timeline with team. Decided to launch MVP in Q2 2025. Key features: document upload, semantic search, chat interface. Tech stack: Python, FastAPI, React.",
        "metadata": {"type": "meeting", "date": "2025-01-18"}
    },
    {
        "id": "6",
        "title": "Books to Read",
        "content": "1. Deep Learning by Goodfellow - foundational ML concepts. 2. Designing Data-Intensive Applications - system design. 3. The Pragmatic Programmer - software engineering best practices.",
        "metadata": {"type": "reading", "date": "2025-01-10"}
    },
]

# Save to JSON file
docs_path = '/Users/madhavayyagari/ml-from-scratch/documents.json'
with open(docs_path, 'w') as f:
    json.dump(sample_documents, f, indent=2)

print(f"Created knowledge base: {docs_path}")
print(f"Documents: {len(sample_documents)}")
print()
print("Sample documents:")
for doc in sample_documents[:3]:
    print(f"  - {doc['title']}")
print("  ...")
print()
print(">>> ADD YOUR OWN DOCUMENTS TO documents.json! <<<")
print()

# =============================================================================
# PART 3: Create Embeddings
# =============================================================================

print("-" * 40)
print("STEP 2: CREATE EMBEDDINGS")
print("-" * 40)
print()

# Check if sentence-transformers is installed
try:
    from sentence_transformers import SentenceTransformer
    USE_REAL_EMBEDDINGS = True
    print("Using sentence-transformers for real embeddings")
except ImportError:
    USE_REAL_EMBEDDINGS = False
    print("sentence-transformers not installed.")
    print("Install with: pip install sentence-transformers")
    print("Using simple TF-IDF-like embeddings for demo...")
print()

if USE_REAL_EMBEDDINGS:
    # Load a small, fast embedding model
    print("Loading embedding model (first time may download ~90MB)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast, good quality

    def get_embedding(text):
        """Get embedding for a piece of text."""
        return model.encode(text)

    embedding_dim = 384  # all-MiniLM-L6-v2 dimension
else:
    # Simple fallback: TF-IDF-like embeddings
    def simple_tokenize(text):
        return text.lower().replace('.', ' ').replace(',', ' ').split()

    # Build vocabulary from all documents
    all_words = set()
    for doc in sample_documents:
        all_words.update(simple_tokenize(doc['content']))
        all_words.update(simple_tokenize(doc['title']))
    vocab = sorted(list(all_words))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    embedding_dim = len(vocab)

    def get_embedding(text):
        """Simple bag-of-words embedding."""
        tokens = simple_tokenize(text)
        vec = np.zeros(embedding_dim)
        for token in tokens:
            if token in word_to_idx:
                vec[word_to_idx[token]] += 1
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

# Embed all documents
print("Embedding documents...")
document_embeddings = []
for doc in sample_documents:
    # Combine title and content for embedding
    text = f"{doc['title']}. {doc['content']}"
    embedding = get_embedding(text)
    document_embeddings.append(embedding)

document_embeddings = np.array(document_embeddings)
print(f"Document embeddings shape: {document_embeddings.shape}")
print(f"  → {len(sample_documents)} documents, {embedding_dim} dimensions each")
print()

# =============================================================================
# PART 4: Build the Retriever
# =============================================================================

print("-" * 40)
print("STEP 3: BUILD THE RETRIEVER")
print("-" * 40)
print()

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def retrieve(query, top_k=3):
    """
    Retrieve the most relevant documents for a query.

    This is the "R" in RAG!
    """
    # 1. Embed the query
    query_embedding = get_embedding(query)

    # 2. Compute similarity with all documents
    similarities = []
    for i, doc_embedding in enumerate(document_embeddings):
        sim = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((i, sim))

    # 3. Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 4. Return top_k documents
    results = []
    for idx, sim in similarities[:top_k]:
        results.append({
            'document': sample_documents[idx],
            'similarity': sim
        })

    return results

# Test retrieval
print("Testing retrieval...")
print()

test_queries = [
    "What are my project ideas?",
    "What am I currently learning?",
    "When is the project launching?",
]

for query in test_queries:
    print(f"Query: \"{query}\"")
    results = retrieve(query, top_k=2)
    for i, result in enumerate(results):
        doc = result['document']
        sim = result['similarity']
        print(f"  {i+1}. [{sim:.3f}] {doc['title']}")
    print()

# =============================================================================
# PART 5: Build the RAG Pipeline
# =============================================================================

print("-" * 40)
print("STEP 4: BUILD THE RAG PIPELINE")
print("-" * 40)
print()

def build_rag_prompt(query, retrieved_docs):
    """
    Build a prompt that includes retrieved context.

    This is the "A" (Augmented) in RAG!
    """
    # Format the context
    context_parts = []
    for i, result in enumerate(retrieved_docs):
        doc = result['document']
        context_parts.append(f"[Document {i+1}: {doc['title']}]\n{doc['content']}")

    context = "\n\n".join(context_parts)

    # Build the full prompt
    prompt = f"""You are a helpful assistant with access to the user's personal documents.

Use the following documents to answer the user's question. If the answer is not in the documents, say so.

DOCUMENTS:
{context}

USER QUESTION: {query}

ANSWER:"""

    return prompt

def rag_query(query, top_k=3):
    """
    Full RAG pipeline:
    1. Retrieve relevant documents
    2. Build augmented prompt
    3. (Would send to LLM here)
    """
    # Retrieve
    retrieved_docs = retrieve(query, top_k=top_k)

    # Build prompt
    prompt = build_rag_prompt(query, retrieved_docs)

    return {
        'query': query,
        'retrieved_docs': retrieved_docs,
        'prompt': prompt
    }

# Demo the full pipeline
print("Full RAG Pipeline Demo:")
print()

demo_query = "What are my goals and what have I learned so far?"
result = rag_query(demo_query, top_k=3)

print(f"Query: \"{result['query']}\"")
print()
print("Retrieved Documents:")
for i, doc_result in enumerate(result['retrieved_docs']):
    doc = doc_result['document']
    print(f"  {i+1}. {doc['title']} (similarity: {doc_result['similarity']:.3f})")
print()
print("Generated Prompt (would send to LLM):")
print("-" * 40)
print(result['prompt'][:1000] + "..." if len(result['prompt']) > 1000 else result['prompt'])
print("-" * 40)
print()

# =============================================================================
# PART 6: Interactive Demo
# =============================================================================

print("-" * 40)
print("STEP 5: TRY IT YOURSELF")
print("-" * 40)
print()

def interactive_rag():
    """Interactive RAG demo."""
    print("RAG System Ready!")
    print("Type a question to search your documents.")
    print("Type 'quit' to exit.")
    print()

    while True:
        query = input("Your question: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not query:
            continue

        result = rag_query(query, top_k=2)

        print()
        print("Retrieved context:")
        for i, doc_result in enumerate(result['retrieved_docs']):
            doc = doc_result['document']
            print(f"  [{doc_result['similarity']:.2f}] {doc['title']}: {doc['content'][:100]}...")
        print()
        print("(In production, this context would be sent to an LLM for a natural answer)")
        print()

# =============================================================================
# PART 7: Save Everything for Later
# =============================================================================

# Save embeddings
np.save('/Users/madhavayyagari/ml-from-scratch/document_embeddings.npy', document_embeddings)
print("Saved document embeddings to document_embeddings.npy")
print()

# =============================================================================
# PART 8: Summary and Next Steps
# =============================================================================

print("=" * 60)
print("RAG SYSTEM COMPLETE!")
print("=" * 60)
print("""
WHAT WE BUILT:
1. Knowledge base (documents.json)
2. Embedding function (converts text → vectors)
3. Retriever (finds similar documents)
4. RAG prompt builder (augments query with context)

THE RAG FLOW:
  User Query
      ↓
  Embed Query
      ↓
  Find Similar Documents (cosine similarity)
      ↓
  Build Prompt with Context
      ↓
  Send to LLM → Get Answer

TO ADD YOUR PERSONAL DATA:
1. Edit documents.json
2. Add your notes, ideas, meeting notes, etc.
3. Re-run this script to re-embed
4. Query your personal knowledge!

NEXT STEPS TO MAKE IT PRODUCTION-READY:
1. Add more documents (your notes, journals, etc.)
2. Connect to Claude/GPT API for actual generation
3. Build a simple API with FastAPI
4. Deploy to Railway/Render
5. Add a web interface

FILES CREATED:
- documents.json      (your knowledge base)
- document_embeddings.npy (cached embeddings)
""")

# Uncomment to run interactive mode:
# interactive_rag()
