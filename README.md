# 📄 AI Text & Document Summarizer

> A modular, schema-enforced AI document summarization pipeline built in Python. Converts raw documents, meeting notes, PDFs, and articles into deterministic, type-safe executive briefings using Google Gemini or OpenAI.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=flat&logo=python&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20GenAI-Gemini%202.0%20Flash-4285F4.svg?style=flat&logo=google)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991.svg?style=flat&logo=openai)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063.svg?style=flat)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 Highlights

- **Dual-Provider Architecture:** Seamlessly switch between Google Gemini (`google-genai`) and OpenAI (`openai`) via `.env` configuration.
- **Strict Schema Enforcement:** Leverages Pydantic v2 and LLM native JSON schema constraints to guarantee uniform, predictable fields (`title`, `executive_summary`, `key_takeaways`, `action_items`, `sentiment_or_tone`).
- **Multi-Format Document Parsing:** Ingests `.txt`, `.md`, `.pdf`, and `.docx` with UTF-8 fallback guards.
- **Sliding-Window Map-Reduce:** Intelligently splits large documents along natural linguistic boundaries (paragraphs and sentences) with context overlap, synthesizing multi-part digests into unified executive briefings.
- **Interactive Terminal UI & Exporters:** Clean CLI powered by `rich`, supporting interactive wizard prompts and automatic export to `.md`, `.json`, and `.txt`.

---

## 🏗️ Architecture

```text
ai-text-summarizer/
├── output/                   # Auto-generated export artifacts (.md, .json, .txt)
├── samples/                  # Sample test fixtures (.md, .docx, .pdf)
└── summarizer/
    ├── chunker.py            # Sliding-window chunker & Map-Reduce synthesis
    ├── config.py             # Validated settings parser & LLM client factory
    ├── exporters.py          # Multi-format serializers (.md, .json, .txt)
    ├── ingestion.py          # Unified document parsers (.txt, .md, .pdf, .docx)
    ├── main.py               # Production CLI application & interactive wizard
    ├── schema.py             # Pydantic data contracts (SummaryResult)
    ├── summarizer.py         # Structured inference engine (Gemini & OpenAI)
    ├── requirements.txt      # Pinned production dependencies
    └── verify_client.py      # Connectivity diagnostics
```

---

## 🚀 Quick Start

### 1. Activate Environment
The virtual environment is pre-configured:
```powershell
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 2. Configure Credentials
Copy the environment template and set your API key:
```powershell
# Windows
copy summarizer\.env.example summarizer\.env

# Linux / macOS
cp summarizer/.env.example summarizer/.env
```
Open `summarizer/.env` and insert your Gemini or OpenAI API key:
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_actual_api_key_here
LLM_MODEL=gemini-2.0-flash
TEMPERATURE=0.2
```

---

## 💻 CLI Usage

### Direct Document Summarization
```powershell
python summarizer/main.py --file samples/sample_sprint_notes.md --format action_oriented
```

### Direct Inline Text Summarization
```powershell
python summarizer/main.py --text "Project Atlas is scheduled for production cutover next month." --format executive
```

### Interactive Terminal Wizard
```powershell
python summarizer/main.py
```

### Offline / Demonstration Mode (No API key needed)
```powershell
python summarizer/main.py --file samples/sample_sprint_notes.md --format action_oriented --mock
```

---

## ⚙️ CLI Options

| Option | Flag | Description | Default |
|---|---|---|---|
| `--file` | `-f` | Path to document (`.txt`, `.md`, `.pdf`, `.docx`) | `None` |
| `--text` | `-t` | Direct text string to summarize | `None` |
| `--format` | `-s` | Style preset: `bullet_points`, `executive`, `detailed`, `action_oriented` | `bullet_points` |
| `--instructions`| `-i` | Custom prompt focus instructions | `None` |
| `--export` | `-e` | Comma-separated export targets: `md`, `json`, `txt`, or `all` | `md,json` |
| `--no-export` | | Display in terminal only without writing files | `False` |
| `--interactive`| | Force interactive wizard mode | `False` |
| `--mock` | | Offline demo mode using synthetic structured output | `False` |

---

## 🧪 Testing & Verification

Run the automated test harnesses:
```powershell
# 1. Multi-format ingestion & sliding-window chunking benchmark
python summarizer/verify_ingestion.py

# 2. Structured schema assertions & Pydantic contract validation
python summarizer/verify_engine.py --mock

# 3. Unit test discovery
python -m unittest discover -s summarizer -p "test_*.py" -t .
```