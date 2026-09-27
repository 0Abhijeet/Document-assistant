import os
import re
import shutil
import uuid
import asyncio

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from src.ingest import ingest_documents
from src.generate import stream_answer

app = FastAPI()

ALLOWED_EXTENSIONS = {".pdf"}


def secure_filename(filename: str) -> str:
    """Strip any path components and non-safe characters. Prevents path traversal via upload filename."""
    filename = os.path.basename(filename)
    filename = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    return filename or "upload.pdf"


def _save_upload_sync(file_obj, file_path: str) -> None:
    """Blocking disk write, isolated into its own function so it can be
    offloaded with asyncio.to_thread from the async route below."""
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file_obj, buffer)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>RAG Document Assistant</title>
            <style>
                :root {
                    --bg: #f4f5f8;
                    --text: #1a1d24;
                    --text-muted: #6b7280;
                    --box: #ffffff;
                    --border: #e5e7eb;
                    --accent: #4f46e5;
                    --accent-hover: #4338ca;
                    --user: #eef0ff;
                    --user-text: #1a1d24;
                    --bot: #f4f5f7;
                    --shadow: 0 1px 3px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04);
                }

                body.dark {
                    --bg: #0f1115;
                    --text: #e7e9ee;
                    --text-muted: #9096a3;
                    --box: #171a21;
                    --border: #2a2e38;
                    --accent: #6366f1;
                    --accent-hover: #7577f3;
                    --user: #2a2a52;
                    --user-text: #e7e9ee;
                    --bot: #1e222b;
                    --shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
                }

                * { box-sizing: border-box; }

                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    max-width: 720px;
                    margin: 0 auto;
                    padding: 24px 16px 40px;
                    background-color: var(--bg);
                    color: var(--text);
                    transition: background-color 0.2s, color 0.2s;
                }

                .top-bar {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 4px;
                }

                .title-group h1 {
                    margin: 0;
                    font-size: 1.4rem;
                    font-weight: 700;
                }

                .title-group p {
                    margin: 4px 0 0;
                    font-size: 0.85rem;
                    color: var(--text-muted);
                }

                #themeToggle {
                    background: var(--box);
                    border: 1px solid var(--border);
                    border-radius: 8px;
                    width: 38px;
                    height: 38px;
                    font-size: 1.1rem;
                    flex-shrink: 0;
                }

                .box {
                    border: 1px solid var(--border);
                    padding: 16px;
                    margin-top: 20px;
                    border-radius: 12px;
                    background: var(--box);
                    box-shadow: var(--shadow);
                }

                .box h3 {
                    margin: 0 0 12px;
                    font-size: 0.95rem;
                    font-weight: 600;
                    color: var(--text-muted);
                    text-transform: uppercase;
                    letter-spacing: 0.03em;
                }

                .upload-row {
                    display: flex;
                    gap: 10px;
                    align-items: center;
                    flex-wrap: wrap;
                }

                #fileName {
                    font-size: 0.85rem;
                    color: var(--text-muted);
                    flex: 1;
                    min-width: 120px;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                }

                .chat-container {
                    padding: 4px 4px 4px 0;
                    height: 420px;
                    overflow-y: auto;
                    display: flex;
                    flex-direction: column;
                }

                .empty-state {
                    margin: auto;
                    text-align: center;
                    color: var(--text-muted);
                    font-size: 0.9rem;
                    max-width: 320px;
                    line-height: 1.5;
                }

                .message {
                    margin: 6px 4px;
                    padding: 10px 14px;
                    border-radius: 14px;
                    max-width: 82%;
                    white-space: pre-wrap;
                    line-height: 1.45;
                    font-size: 0.95rem;
                }

                .user {
                    background-color: var(--user);
                    color: var(--user-text);
                    margin-left: auto;
                    border-bottom-right-radius: 4px;
                }

                .bot {
                    background-color: var(--bot);
                    margin-right: auto;
                    border-bottom-left-radius: 4px;
                }

                .bot.error {
                    color: #dc2626;
                }

                .typing-dots span {
                    display: inline-block;
                    width: 6px;
                    height: 6px;
                    margin-right: 3px;
                    border-radius: 50%;
                    background: var(--text-muted);
                    animation: blink 1.2s infinite;
                }

                .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
                .typing-dots span:nth-child(3) { animation-delay: 0.4s; }

                @keyframes blink {
                    0%, 80%, 100% { opacity: 0.3; }
                    40% { opacity: 1; }
                }

                .input-row {
                    display: flex;
                    gap: 10px;
                    margin-top: 12px;
                }

                input[type="text"] {
                    flex: 1;
                    padding: 10px 12px;
                    border: 1px solid var(--border);
                    border-radius: 8px;
                    background: var(--bg);
                    color: var(--text);
                    font-size: 0.95rem;
                }

                input[type="text"]:focus, input[type="file"]:focus {
                    outline: 2px solid var(--accent);
                    outline-offset: 1px;
                }

                button {
                    padding: 10px 16px;
                    cursor: pointer;
                    border: none;
                    border-radius: 8px;
                    background: var(--accent);
                    color: #fff;
                    font-size: 0.9rem;
                    font-weight: 500;
                    transition: background 0.15s;
                }

                button:hover:not(:disabled) {
                    background: var(--accent-hover);
                }

                button:disabled {
                    opacity: 0.55;
                    cursor: not-allowed;
                }

                #status {
                    margin: 10px 0 0;
                    font-size: 0.85rem;
                    color: var(--text-muted);
                }

                #status.ok { color: #16a34a; }
                #status.err { color: #dc2626; }

                footer {
                    margin-top: 24px;
                    text-align: center;
                    font-size: 0.8rem;
                    color: var(--text-muted);
                }

                footer a {
                    color: var(--text-muted);
                    text-decoration: underline;
                }

                @media (max-width: 480px) {
                    .chat-container { height: 340px; }
                    .message { max-width: 90%; }
                }
            </style>
        </head>
        <body>
            <div class="top-bar">
                <div class="title-group">
                    <h1>📄 RAG Document Assistant</h1>
                    <p>Upload a PDF, then ask questions grounded in it.</p>
                </div>
                <button id="themeToggle" onclick="toggleTheme()" title="Toggle dark mode">🌙</button>
            </div>

            <div class="box">
                <h3>1. Upload a document</h3>
                <div class="upload-row">
                    <input type="file" id="fileInput" accept=".pdf" onchange="onFileChosen()" />
                    <span id="fileName">No file selected</span>
                    <button id="uploadBtn" onclick="uploadFile()">Upload</button>
                </div>
                <p id="status"></p>
            </div>

            <div class="box">
                <h3>2. Ask a question</h3>
                <div id="chat" class="chat-container">
                    <div class="empty-state" id="emptyState">
                        Upload a PDF above, then ask a question about it — the answer streams in, grounded only in what's in the document.
                    </div>
                </div>
                <div class="input-row">
                    <input type="text" id="questionInput" placeholder="Ask something about the document..." onkeydown="if(event.key === 'Enter') askQuestion()" />
                    <button id="askBtn" onclick="askQuestion()">Send</button>
                </div>
            </div>

            <footer>
                <a href="https://github.com/0Abhijeet/Document-assistant" target="_blank" rel="noopener">Source on GitHub</a>
            </footer>

            <script>
                const chatBox = document.getElementById("chat");
                const emptyState = document.getElementById("emptyState");
                const statusEl = document.getElementById("status");
                const uploadBtn = document.getElementById("uploadBtn");
                const askBtn = document.getElementById("askBtn");
                const questionInput = document.getElementById("questionInput");

                // Persist theme choice; fall back to the visitor's OS preference
                // on first load so a shared demo link doesn't force light mode
                // on someone who reads everything in dark.
                (function initTheme() {
                    const saved = localStorage.getItem("theme");
                    const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
                    if (saved === "dark" || (!saved && prefersDark)) {
                        document.body.classList.add("dark");
                    }
                })();

                function toggleTheme() {
                    document.body.classList.toggle("dark");
                    localStorage.setItem("theme", document.body.classList.contains("dark") ? "dark" : "light");
                }

                function addMessage(text, className) {
                    if (emptyState) emptyState.remove();
                    const div = document.createElement("div");
                    div.className = "message " + className;
                    div.innerText = text;
                    chatBox.appendChild(div);
                    chatBox.scrollTop = chatBox.scrollHeight;
                    return div;
                }

                function onFileChosen() {
                    const fileInput = document.getElementById("fileInput");
                    document.getElementById("fileName").innerText =
                        fileInput.files.length ? fileInput.files[0].name : "No file selected";
                }

                async function uploadFile() {
                    const fileInput = document.getElementById("fileInput");

                    if (!fileInput.files.length) {
                        statusEl.className = "err";
                        statusEl.innerText = "Please choose a PDF first.";
                        return;
                    }

                    const formData = new FormData();
                    formData.append("file", fileInput.files[0]);

                    uploadBtn.disabled = true;
                    statusEl.className = "";
                    statusEl.innerText = "Uploading and processing (embedding + storing)...";

                    try {
                        const response = await fetch("/upload", { method: "POST", body: formData });
                        const data = await response.json();
                        statusEl.className = response.ok ? "ok" : "err";
                        statusEl.innerText = data.message || data.detail || "Something went wrong.";
                    } catch (err) {
                        statusEl.className = "err";
                        statusEl.innerText = "Upload failed: could not reach the server.";
                    } finally {
                        uploadBtn.disabled = false;
                    }
                }

                async function askQuestion() {
                    const question = questionInput.value.trim();
                    if (!question) return;

                    addMessage(question, "user");
                    questionInput.value = "";
                    questionInput.disabled = true;
                    askBtn.disabled = true;

                    const botBubble = addMessage("", "bot");
                    botBubble.innerHTML = '<span class="typing-dots"><span></span><span></span><span></span></span>';

                    try {
                        const response = await fetch("/stream", {
                            method: "POST",
                            headers: { "Content-Type": "application/x-www-form-urlencoded" },
                            body: new URLSearchParams({ question })
                        });

                        if (!response.ok) {
                            botBubble.className = "message bot error";
                            botBubble.innerText = "Error: could not get an answer.";
                            return;
                        }

                        const reader = response.body.getReader();
                        const decoder = new TextDecoder();
                        let first = true;

                        while (true) {
                            const { done, value } = await reader.read();
                            if (done) break;
                            const chunk = decoder.decode(value);
                            if (first) { botBubble.innerText = ""; first = false; }
                            botBubble.innerText += chunk;
                            chatBox.scrollTop = chatBox.scrollHeight;
                        }
                    } catch (err) {
                        botBubble.className = "message bot error";
                        botBubble.innerText = "Error: could not reach the server.";
                    } finally {
                        questionInput.disabled = false;
                        askBtn.disabled = false;
                        questionInput.focus();
                    }
                }
            </script>

        </body>
    </html>
    """


# These are now `async def`. Every blocking step inside (disk I/O, PDF
# parsing, embedding inference, DB calls, the Groq call) has been converted
# to either a genuinely async call (asyncpg, AsyncGroq) or explicitly
# offloaded to a thread (asyncio.to_thread) further down the call chain in
# ingest.py / retrieve.py / generate.py. Nothing here blocks the event loop.

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs("data/docs", exist_ok=True)

    safe_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    file_path = f"data/docs/{unique_name}"

    await asyncio.to_thread(_save_upload_sync, file.file, file_path)

    try:
        doc_id = await ingest_documents(file_path, safe_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

    return {"message": "File uploaded and processed successfully!", "document_id": doc_id}


@app.post("/stream")
async def stream_endpoint(question: str = Form(...), provider: str = Form("groq")):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if provider not in {"groq", "bedrock_invoke", "bedrock_converse", "openai", "azure"}:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider!r}")
    generator = stream_answer(question, provider=provider)
    return StreamingResponse(generator, media_type="text/plain")