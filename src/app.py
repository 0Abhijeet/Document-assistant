import os
import re
import uuid

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
import shutil

from src.ingest import ingest_documents
from src.generate import stream_answer

app = FastAPI()

ALLOWED_EXTENSIONS = {".pdf"}


def secure_filename(filename: str) -> str:
    """Strip any path components and non-safe characters. Prevents path traversal via upload filename."""
    filename = os.path.basename(filename)
    filename = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    return filename or "upload.pdf"


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


# NOTE: these are `def`, not `async def`, on purpose. Every step inside them
# (embedding, disk I/O, DB calls) is blocking, synchronous code. An `async def`
# route running blocking code freezes the entire event loop for every other
# request in flight. FastAPI automatically runs plain `def` routes in a worker
# thread pool, so blocking code here no longer blocks the server.
# Tradeoff: the thread pool has a fixed size, so this doesn't scale to heavy
# concurrent load — fine for a demo/portfolio app, not for production traffic.

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs("data/docs", exist_ok=True)

    safe_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    file_path = f"data/docs/{unique_name}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        doc_id = ingest_documents(file_path, safe_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

    return {"message": "File uploaded and processed successfully!", "document_id": doc_id}


@app.post("/stream")
def stream_endpoint(question: str = Form(...)):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    generator = stream_answer(question)
    return StreamingResponse(generator, media_type="text/plain")
