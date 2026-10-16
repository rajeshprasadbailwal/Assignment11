  </style>
</head>
<body>
  <main class="container">
    <h1>RAG Knowledge Transfer Assistant</h1>
    <p class="subtitle">Upload technical documentation and get answers to your development questions.</p>

    <section class="card">
      <h2>1. Upload Technical Document (PDF)</h2>
      <div class="form-group">
        <input type="file" id="pdfFile" accept=".pdf,application/pdf">
        <button id="uploadBtn" type="button">Upload &amp; Index</button>
      </div>
      <div id="uploadStatus" class="status" role="status"></div>
    </section>

    <section class="card">
      <h2>2. Ask a Question</h2>
      <div class="form-group">
        <input type="text" id="questionInput" placeholder="e.g., Which technologies are used?">
        <button id="askBtn" type="button">Submit</button>
      </div>
      <div id="answerResult" class="answer-box" role="status" style="display:none"></div>
    </section>
  </main>

  <script>
    const uploadBtn = document.getElementById("uploadBtn");
    const askBtn = document.getElementById("askBtn");
    const fileInput = document.getElementById("pdfFile");
    const uploadStatus = document.getElementById("uploadStatus");
    const questionInput = document.getElementById("questionInput");
    const answerResult = document.getElementById("answerResult");

    function showStatus(element, message, type) {
      element.className = "status " + (type || "");
      element.textContent = message;
    }

    async function readResponse(response) {
      const text = await response.text();
      let data;
      try { data = text ? JSON.parse(text) : {}; }
      catch { data = { detail: text || "The server returned an empty response." }; }
      if (!response.ok) {
        throw new Error(data.detail || data.message || ("Request failed with HTTP " + response.status));
      }
      return data;
    }

    async function uploadPDF() {
      // Read the current file selection at click time.
      const selectedFile = fileInput.files && fileInput.files.length ? fileInput.files[0] : null;
      if (!selectedFile) {
        showStatus(uploadStatus, "Choose a PDF using the file picker before clicking Upload & Index.", "error");
        return;
      }
      if (!selectedFile.name.toLowerCase().endsWith(".pdf")) {
        showStatus(uploadStatus, "Please select a file with a .pdf extension.", "error");
        return;
      }

      const formData = new FormData();
      formData.append("file", selectedFile, selectedFile.name);
      uploadBtn.disabled = true;
      showStatus(uploadStatus, "Uploading and indexing PDF...", "");

      try {
        const response = await fetch("/upload-kt-pdf/", { method: "POST", body: formData });
        const data = await readResponse(response);
        showStatus(uploadStatus, data.message + " Pages: " + data.total_pages + ". Chunks: " + data.total_chunks + ".", "success");
      } catch (error) {
        showStatus(uploadStatus, "Upload failed: " + error.message, "error");
      } finally {
        uploadBtn.disabled = false;
      }
    }

    async function askQuestion() {
      const question = questionInput.value.trim();
      if (!question) {
        answerResult.style.display = "block";
        answerResult.textContent = "Please enter a question.";
        return;
      }

      askBtn.disabled = true;
      answerResult.style.display = "block";
      answerResult.textContent = "Searching the PDF and generating an answer...";

      try {
        const response = await fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: question })
        });
        const data = await readResponse(response);
        answerResult.textContent = data.answer || "The assistant returned no answer.";
      } catch (error) {
        answerResult.textContent = "Assistant error: " + error.message;
      } finally {
        askBtn.disabled = false;
      }
    }

    uploadBtn.addEventListener("click", uploadPDF);
    askBtn.addEventListener("click", askQuestion);
    questionInput.addEventListener("keydown", event => {
      if (event.key === "Enter") askQuestion();
    });
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
