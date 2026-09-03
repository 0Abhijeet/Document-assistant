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
    <html>
        <head>
            <title>RAG Chat Assistant</title>
            <style>
                :root {
                    --bg: #f5f5f5;
                    --text: #000;
                    --box: #ffffff;
                    --user: #DCF8C6;
                    --bot: #eeeeee;
                }

                body.dark {
                    --bg: #121212;
                    --text: #ffffff;
                    --box: #1e1e1e;
                    --user: #2e7d32;
                    --bot: #2a2a2a;
                }

                body {
                    font-family: Arial, sans-serif;
                    max-width: 700px;
                    margin: auto;
                    padding: 20px;
                    background-color: var(--bg);
                    color: var(--text);
                    transition: 0.3s;
                }

                h1 {
                    text-align: center;
                }

                .box {
                    border: 1px solid #ddd;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 8px;
                    background: var(--box);
                }

                .chat-container {
                    border: 1px solid #ddd;
                    padding: 10px;
                    height: 400px;
                    overflow-y: auto;
                    border-radius: 8px;
                    background: var(--box);
                }

                .message {
                    margin: 10px 0;
                    padding: 10px;
                    border-radius: 8px;
                    max-width: 80%;
                    white-space: pre-wrap;
                }

                .user {
                    background-color: var(--user);
                    margin-left: auto;
                }

                .bot {
                    background-color: var(--bot);
                    margin-right: auto;
                }

                .input-row {
                    display: flex;
                    gap: 10px;
                }

                input[type="text"] {
                    flex: 1;
                    padding: 8px;
                }

                button {
                    padding: 8px 12px;
                    cursor: pointer;
                }

                #status {
                    margin-top: 10px;
                    color: green;
                }

                .top-bar {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
            </style>
        </head>
        <body>
            <div class="top-bar">
                <h1>📄 RAG Chat Assistant</h1>
                <button onclick="toggleTheme()">🌙</button>
            </div>

            <div class="box">
                <h3>Upload PDF</h3>
                <input type="file" id="fileInput" />
                <br><br>
                <button onclick="uploadFile()">Upload</button>
                <p id="status"></p>
            </div>

            <div class="box">
                <h3>Chat</h3>
                <div id="chat" class="chat-container"></div>
                <br>
                <div class="input-row">
                    <input type="text" id="questionInput" placeholder="Type your question..." />
                    <button onclick="askQuestion()">Send</button>
                </div>
            </div>

            <script>
                const chatBox = document.getElementById("chat");

                function toggleTheme() {
                    document.body.classList.toggle("dark");
                }

                function addMessage(text, className) {
                    const div = document.createElement("div");
                    div.className = "message " + className;
                    div.innerText = text;
                    chatBox.appendChild(div);
                    chatBox.scrollTop = chatBox.scrollHeight;
                    return div;
                }

                async function uploadFile() {
                    const fileInput = document.getElementById("fileInput");
                    const status = document.getElementById("status");

                    if (!fileInput.files.length) {
                        alert("Please select a file.");
                        return;
                    }

                    const formData = new FormData();
                    formData.append("file", fileInput.files[0]);

                    status.innerText = "Uploading and processing...";

                    const response = await fetch("/upload", {
                        method: "POST",
                        body: formData
                    });

                    const data = await response.json();
                    status.innerText = data.message || data.detail || "Something went wrong.";
                }

                async function askQuestion() {
                    const input = document.getElementById("questionInput");
                    const question = input.value;

                    if (!question) return;

                    addMessage(question, "user");
                    input.value = "";

                    const botBubble = addMessage("", "bot");

                    const response = await fetch("/stream", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/x-www-form-urlencoded"
                        },
                        body: new URLSearchParams({ question })
                    });

                    if (!response.ok) {
                        botBubble.innerText = "Error: could not get an answer.";
                        return;
                    }

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        const chunk = decoder.decode(value);
                        botBubble.innerText += chunk;
                        chatBox.scrollTop = chatBox.scrollHeight;
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
async def stream_endpoint(question: str = Form(...)):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    generator = stream_answer(question)
    return StreamingResponse(generator, media_type="text/plain")