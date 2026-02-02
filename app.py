"""
Personal AI Agent Dashboard - FastAPI Backend
===============================================

Agent uses Claude to route queries between:
- Document search (ChromaDB RAG)
- Google Calendar
- General knowledge

Run: python app.py → http://localhost:8000
"""

import os
import json
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import anthropic
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from rag import get_engine
from load_documents import load_txt, load_md, load_pdf, load_docx, load_image, chunk_text, PDF_SUPPORT, DOCX_SUPPORT
import calendar_integration

app = FastAPI(title="Personal AI Agent")

PROJECT_DIR = '/Users/madhavayyagari/ml-from-scratch'
DOCS_PATH = os.path.join(PROJECT_DIR, 'data/documents.json')

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(PROJECT_DIR, "static")), name="static")


# =============================================================================
# Agent: Intent Classification + Routing
# =============================================================================

def classify_intent(query):
    """Classify the user's intent using keyword pre-check + Claude Haiku fallback."""
    q = query.lower()

    # Keyword pre-check for calendar
    calendar_keywords = ['schedule', 'meeting', 'meetings', 'calendar', 'appointment',
                         'event', 'events', 'today', 'tomorrow', 'this week', 'busy',
                         'free time', 'available']
    if any(kw in q for kw in calendar_keywords):
        return 'calendar'

    # Keyword pre-check for documents/personal info
    doc_keywords = ['my ', 'license', 'passport', 'ssn', 'social security',
                    'insurance', 'address', 'phone number', 'account',
                    'document', 'id number', 'birth', 'expir', 'policy',
                    'vin', 'registration', 'certificate', 'note', 'notes',
                    'wrote', 'saved', 'recorded', 'bill', 'invoice', 'payment',
                    'receipt', 'statement', 'att', 'at&t', 'verizon', 'tmobile']
    if any(kw in q for kw in doc_keywords):
        return 'documents'

    # Fall back to Claude for ambiguous queries
    client = anthropic.Anthropic()

    prompt = f"""Classify this query into one category. Respond with ONLY the category name.

Categories:
- "calendar" : about schedule, meetings, events, appointments
- "documents" : about user's personal stored information, records, IDs, files
- "general" : general knowledge, greetings, how-to questions

Query: {query}

Category:"""

    try:
        message = client.messages.create(
            model="claude-haiku-4-20250414",
            max_tokens=20,
            messages=[{"role": "user", "content": prompt}]
        )
        intent = message.content[0].text.strip().lower().strip('"')
        if intent in ('calendar', 'documents', 'general'):
            return intent
        return 'general'
    except Exception:
        return 'general'


def handle_calendar_query(query):
    """Handle calendar-related queries."""
    if not calendar_integration.is_authenticated():
        return {
            "answer": "Google Calendar is not connected. Please click the 'Connect' button in the sidebar to authorize access.",
            "intent": "calendar",
            "sources": []
        }

    events = calendar_integration.get_upcoming_events(days=7)
    if events is None:
        return {
            "answer": "Failed to fetch calendar events. Please try reconnecting Google Calendar.",
            "intent": "calendar",
            "sources": []
        }

    # Format events as context for Claude
    if not events:
        events_text = "No upcoming events found in the next 7 days."
    else:
        event_lines = []
        for e in events:
            start = e['start']
            summary = e['summary']
            location = f" at {e['location']}" if e['location'] else ""
            event_lines.append(f"- {start}: {summary}{location}")
        events_text = "\n".join(event_lines)

    # Use Claude to answer based on calendar context
    client = anthropic.Anthropic()
    prompt = f"""Based on the following calendar events, answer the user's question.
Today is {datetime.now().strftime('%A, %B %d, %Y')}.

UPCOMING EVENTS:
{events_text}

QUESTION: {query}"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        return {
            "answer": message.content[0].text,
            "intent": "calendar",
            "sources": ["Google Calendar"]
        }
    except Exception as e:
        return {
            "answer": f"Error generating calendar answer: {e}",
            "intent": "calendar",
            "sources": []
        }


def handle_documents_query(query):
    """Handle document search queries via RAG."""
    engine = get_engine()
    # Normalize phone numbers in query (spaces/dashes to dots) for better matching
    import re
    normalized_query = re.sub(r'(\d{3})[\s\-](\d{3})[\s\-](\d{4})', r'\1.\2.\3', query)
    results = engine.search(normalized_query, top_k=8)  # More chunks for complex queries
    answer = engine.generate_answer(query, results)
    sources = list(set(r['source'] for r in results)) if results else []

    return {
        "answer": answer,
        "intent": "documents",
        "sources": sources
    }


def handle_general_query(query):
    """Handle general knowledge queries."""
    client = anthropic.Anthropic()

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": query}]
        )
        return {
            "answer": message.content[0].text,
            "intent": "general",
            "sources": []
        }
    except Exception as e:
        return {
            "answer": f"Error: {e}",
            "intent": "general",
            "sources": []
        }


# =============================================================================
# API Endpoints
# =============================================================================

@app.post("/api/chat")
async def chat(request: Request):
    """Main agent endpoint: classify intent and route to handler."""
    body = await request.json()
    query = body.get("query", "").strip()

    if not query:
        return JSONResponse({"error": "Empty query"}, status_code=400)

    # Classify intent
    intent = classify_intent(query)

    # Route to appropriate handler
    if intent == "calendar":
        result = handle_calendar_query(query)
    elif intent == "documents":
        result = handle_documents_query(query)
    else:
        result = handle_general_query(query)

    return JSONResponse(result)


@app.get("/api/status")
async def status():
    """Return which integrations are connected."""
    # Check if notes are indexed
    notes_connected = False
    if os.path.exists(DOCS_PATH):
        with open(DOCS_PATH, 'r') as f:
            docs = json.load(f)
        notes_connected = any(d.get('metadata', {}).get('type') == 'apple_note' for d in docs)

    return JSONResponse({
        "calendar": calendar_integration.is_authenticated(),
        "documents": True,
        "notes": notes_connected
    })


@app.get("/api/calendar/today")
async def calendar_today():
    """Get today's events for the sidebar widget."""
    if not calendar_integration.is_authenticated():
        return JSONResponse({"authenticated": False, "events": []})

    events = calendar_integration.get_todays_events()
    if events is None:
        return JSONResponse({"authenticated": True, "events": [], "error": "Failed to fetch"})

    return JSONResponse({"authenticated": True, "events": events})


@app.get("/api/documents")
async def documents_list():
    """List indexed documents."""
    engine = get_engine()
    sources = engine.list_documents()
    return JSONResponse({"documents": sources, "count": len(sources)})


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file and add it to the knowledge base."""
    # Supported extensions
    ext = os.path.splitext(file.filename)[1].lower()
    supported = {'.txt', '.md', '.pdf', '.docx', '.jpg', '.jpeg', '.png'}

    if ext not in supported:
        return JSONResponse(
            {"error": f"Unsupported file type: {ext}. Supported: {', '.join(supported)}"},
            status_code=400
        )

    if ext == '.pdf' and not PDF_SUPPORT:
        return JSONResponse({"error": "PDF support not installed (pip install pypdf)"}, status_code=400)
    if ext == '.docx' and not DOCX_SUPPORT:
        return JSONResponse({"error": "DOCX support not installed (pip install python-docx)"}, status_code=400)

    # Save file to uploads folder
    uploads_dir = os.path.join(PROJECT_DIR, 'my_data', 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)

    filepath = os.path.join(uploads_dir, file.filename)

    # Handle duplicate filenames
    base, extension = os.path.splitext(file.filename)
    counter = 1
    while os.path.exists(filepath):
        filepath = os.path.join(uploads_dir, f"{base}_{counter}{extension}")
        counter += 1

    # Save the file
    content = await file.read()
    with open(filepath, 'wb') as f:
        f.write(content)

    # Process the file based on type
    try:
        loaders = {
            '.txt': load_txt,
            '.md': load_md,
            '.pdf': load_pdf,
            '.docx': load_docx,
            '.jpg': load_image,
            '.jpeg': load_image,
            '.png': load_image,
        }

        doc_data = loaders[ext](filepath)
        rel_path = os.path.relpath(filepath, os.path.join(PROJECT_DIR, 'my_data'))

        # Load existing documents
        if os.path.exists(DOCS_PATH):
            with open(DOCS_PATH, 'r') as f:
                documents = json.load(f)
        else:
            documents = []

        # Get next ID
        existing_ids = [int(d['id'].split('_')[0]) for d in documents if d['id'].split('_')[0].isdigit()]
        next_id = max(existing_ids, default=0) + 1

        # Chunk if needed
        chunks = chunk_text(doc_data['content'], chunk_size=500, overlap=50)
        added_count = 0

        for i, chunk in enumerate(chunks):
            doc_id = f"{next_id}_{i+1}" if len(chunks) > 1 else str(next_id)
            title = f"{doc_data['title']} (Part {i+1}/{len(chunks)})" if len(chunks) > 1 else doc_data['title']

            document = {
                'id': doc_id,
                'title': title,
                'content': chunk,
                'metadata': {
                    'source': rel_path,
                    'type': ext[1:],
                    'loaded': datetime.now().isoformat()
                }
            }
            documents.append(document)
            added_count += 1

        # Save updated documents
        with open(DOCS_PATH, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)

        # Force RAG engine to re-sync on next query
        import rag
        if rag._engine is not None:
            rag._engine._initialized = False

        return JSONResponse({
            "success": True,
            "filename": os.path.basename(filepath),
            "title": doc_data['title'],
            "chunks": added_count,
            "message": f"Added '{doc_data['title']}' ({added_count} chunk{'s' if added_count > 1 else ''})"
        })

    except Exception as e:
        # Clean up file on error
        if os.path.exists(filepath):
            os.remove(filepath)
        return JSONResponse({"error": f"Failed to process file: {str(e)}"}, status_code=500)


# =============================================================================
# OAuth Endpoints
# =============================================================================

@app.get("/auth/google")
async def auth_google(request: Request):
    """Start Google OAuth flow."""
    redirect_uri = "http://localhost:8000/auth/google/callback"
    try:
        flow = calendar_integration.get_oauth_flow(redirect_uri)
        auth_url, _ = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
        )
        return RedirectResponse(auth_url)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/auth/google/callback")
async def auth_google_callback(request: Request):
    """Handle OAuth callback."""
    redirect_uri = "http://localhost:8000/auth/google/callback"
    code = request.query_params.get("code")

    if not code:
        return JSONResponse({"error": "No authorization code received"}, status_code=400)

    try:
        flow = calendar_integration.get_oauth_flow(redirect_uri)
        flow.fetch_token(code=code)
        creds = flow.credentials
        calendar_integration.save_credentials(creds)
        # Redirect back to dashboard
        return RedirectResponse("/")
    except Exception as e:
        return JSONResponse({"error": f"OAuth error: {e}"}, status_code=500)


# =============================================================================
# Dashboard
# =============================================================================

@app.get("/")
async def dashboard():
    """Serve the dashboard HTML."""
    html_path = os.path.join(PROJECT_DIR, "static", "index.html")
    with open(html_path, 'r') as f:
        content = f.read()
    return HTMLResponse(content)


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("Starting Personal AI Agent Dashboard...")
    print("Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)
