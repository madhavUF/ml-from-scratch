# Personal AI Assistant - Project Plan

## Vision
A personal AI that knows everything about you. Ask questions in natural language, get answers from your own data.

**Long-term goal:** Productize this for others to use - a simple, private, local-first personal AI anyone can set up.

## Why This Matters
- Big tech has your data, but YOU can't easily search it
- LLMs are powerful but don't know YOUR information
- Privacy-first: your data stays on your machine
- Open source alternative to commercial "second brain" apps

```
You: "What's my driver's license number?"
AI: "Your DL number is X12345678, expires March 2027."

You: "Show me photos from my trip to Japan"
AI: [Shows relevant photos]

You: "What did I decide in last week's meeting?"
AI: "You decided to launch MVP in Q2, with John on frontend..."
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     YOUR MAC (Local)                        │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    my_data/                          │   │
│  │  documents/    images/    audio/    exports/         │   │
│  │  ├── work/     ├── photos/         ├── emails/       │   │
│  │  ├── personal/ ├── screenshots/    ├── messages/     │   │
│  │  └── notes/    └── scans/          └── calendar/     │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                 │
│                      [INDEXER]                              │
│                           │                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              LOCAL VECTOR DATABASE                   │   │
│  │  • Text embeddings (sentence-transformers)           │   │
│  │  • Image embeddings (CLIP)                           │   │
│  │  • Metadata (dates, types, sources)                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                 │
│                    [RETRIEVAL]                              │
│                           │                                 │
└───────────────────────────┼─────────────────────────────────┘
                            │ Only relevant chunks
                            ▼
                  ┌───────────────────┐
                  │   Claude API      │
                  │   (Generation)    │
                  └───────────────────┘
                            │
                            ▼
                       YOUR ANSWER
```

## Project Phases

### Phase 1: Foundation (Current)
**Status: ✅ Mostly Complete**

What we have:
- [x] Basic document loading (txt, md, pdf, docx)
- [x] Text embeddings (sentence-transformers)
- [x] Cosine similarity search
- [x] Chunking for large documents
- [x] Simple CLI interface

What to add:
- [ ] Better project structure
- [ ] Configuration file
- [ ] Persistent database (SQLite or ChromaDB)

### Phase 2: Enhanced Processing
**Status: 🔲 Not Started**

Goals:
- [ ] OCR for images with text (pytesseract)
- [ ] Screenshot text extraction
- [ ] Better metadata extraction (dates, authors)
- [ ] Smarter chunking (by sections, paragraphs)
- [ ] File watching (auto-index new files)

### Phase 3: Image Support
**Status: 🔲 Not Started**

Goals:
- [ ] CLIP embeddings for images
- [ ] Visual similarity search
- [ ] Image descriptions (BLIP or Claude Vision)
- [ ] Photo organization by content

### Phase 4: Claude Integration
**Status: 🔲 Not Started**

Goals:
- [ ] Claude API connection
- [ ] Conversational interface
- [ ] Multi-turn memory
- [ ] Send images to Claude for analysis

### Phase 5: Always-On Assistant
**Status: 🔲 Not Started**

Goals:
- [ ] Background service (auto-start on Mac)
- [ ] API endpoint (FastAPI)
- [ ] Voice input option (Whisper)
- [ ] Menu bar app or CLI daemon
- [ ] Mobile access (via API)

### Phase 6: Polish & Deploy
**Status: 🔲 Not Started**

Goals:
- [ ] Clean GitHub repo with README
- [ ] Docker container option
- [ ] Privacy controls (exclude sensitive files)
- [ ] Backup/restore functionality
- [ ] Usage dashboard

### Phase 7: Productize for Others
**Status: 🔲 Not Started**

Goals:
- [ ] Easy setup script (one command install)
- [ ] Configuration wizard (guided setup)
- [ ] Multiple platform support (Mac, Windows, Linux)
- [ ] Documentation and tutorials
- [ ] Example use cases and templates
- [ ] Community feedback integration
- [ ] Optional cloud sync (encrypted)
- [ ] Pricing model exploration (open core?)

## File Structure (Target)

```
ml-from-scratch/
├── README.md                 # Project documentation
├── PROJECT_PLAN.md           # This file
├── requirements.txt          # Dependencies
├── config.yaml               # Configuration
│
├── my_data/                  # YOUR PERSONAL DATA
│   ├── documents/            # PDFs, Word, text
│   ├── images/               # Photos, screenshots
│   ├── audio/                # Voice memos (future)
│   └── exports/              # Email, calendar exports
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── indexer.py            # Index files → embeddings
│   ├── retriever.py          # Search embeddings
│   ├── generator.py          # Claude API integration
│   ├── ocr.py                # Image text extraction
│   ├── embeddings.py         # Embedding models
│   └── utils.py              # Helpers
│
├── data/                     # Generated data (git-ignored)
│   ├── index.db              # SQLite database
│   ├── embeddings/           # Cached embeddings
│   └── cache/                # Temporary files
│
├── lessons/                  # Your learning files (keep!)
│   ├── lesson_1_*.py
│   ├── lesson_2_*.py
│   └── ...
│
└── scripts/
    ├── index.py              # Run indexer
    ├── search.py             # Search CLI
    ├── chat.py               # Chat with Claude
    └── serve.py              # API server
```

## Getting Started

### Current (Phase 1):
```bash
# Add files to my_data/
cp ~/Documents/*.pdf my_data/

# Index them
python load_documents.py

# Search
python rag.py
```

### Future (Phase 5+):
```bash
# Start the assistant (runs in background)
python -m personal_ai start

# Query from anywhere
personal-ai "What's my DL number?"

# Or via API
curl http://localhost:8000/ask?q="What's my DL number?"
```

## Tech Stack

| Component | Current | Future Option |
|-----------|---------|---------------|
| Embeddings | sentence-transformers | Same |
| Image embeddings | - | CLIP |
| OCR | - | pytesseract |
| Vector DB | numpy files | ChromaDB or SQLite |
| LLM | - | Claude API |
| API | - | FastAPI |
| Voice | - | Whisper |

## Privacy Considerations

- All indexing happens locally
- Only relevant chunks sent to Claude (not entire files)
- Can exclude sensitive folders (config.yaml)
- API key stored securely (environment variable)

## Next Steps

1. Reorganize project structure
2. Add configuration file
3. Improve document loader
4. Add OCR for images
5. Connect Claude API
