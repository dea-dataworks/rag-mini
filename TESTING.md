# TESTING

Pragmatic test plan for **RAG Explorer v0.2**. Optimized for Windows + local Ollama.
Do it in short passes; don’t try to test everything in one go.

---

## 0) One-time setup

**Create clean venv (Python 3.12/3.13):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

**(Optional) Enable OpenAI provider:**
```powershell
pip install openai langchain-openai
setx OPENAI_API_KEY "sk-..."
```

**Start Ollama (if not already):**
- Ensure the Ollama tray is running.
- Pull model once: `ollama pull mistral`

---

## 1) Fast unit tests ( < 1 minute)

```powershell
pytest -q
```

**Pass criteria:**
- All tests green (or known skips).  
- If any fail, fix before manual smoke.

---

## 2) Smoke test — happy path (5–10 min)

**Launch app:**
```powershell
streamlit run app.py
```

**Step-by-step:**
1. **Index base**: In sidebar, confirm `BASE_DIR` points to default (e.g., `rag_store`).  
2. **Upload**: Add `sample_data/udhr.pdf` (and a `.txt` if you have one).  
3. **Build / Load Index**: Click. Expect a spinner, then stats + manifest.  
4. **Ask**: A simple factual question answerable from UDHR.  
5. **Check answer block**:
   - Has citations `(udhr.pdf p.X)`.
   - Guardrail banner is **not** blocking (OK or warn at most).
6. **Downloads**: Use **Markdown/CSV/Excel** buttons. Confirm files download and contain:
   - Q, A, sources.
   - `run_settings` with **provider**, **model**, **retrieval_mode**, **top_k**, **index_name**.

**Pass criteria:**
- No exceptions in Streamlit logs.
- Answers have citations.
- Downloads have expected fields.

---

## 3) Provider toggle & fallback (5 min)

**In Advanced / Settings:**
- Select **OpenAI** without an API key.
- Ask a question.

**Expect:**
- A small toast/notice indicates **fallback to Ollama**.
- `run_settings.provider` shows **ollama** and `meta.fallback=True` in the turn payload.

**Then (optional)** set a valid key and:
- Select **OpenAI**, ask again, confirm no fallback.

---

## 4) Retrieval modes + thresholds (10 min)

Test with same question:
- Dense only.
- BM25 only.
- Hybrid (RRF).

Play with:
- `TOP_K` (e.g., 3 vs 6).
- `SCORE_THRESH` (e.g., 0.3).

**Pass criteria:**
- Differences reflected in cited chunks & `run_settings`.
- No crashes when changing modes mid-session.

---

## 5) Index switcher (named bases) (5–10 min)

**Create two bases:**
- In sidebar, set base name (e.g., `demo-A`), build from UDHR.
- Switch to `demo-B`, build from different files (or just UDHR again).

**Expect:**
- Dropdown lists both bases.
- Switch invalidates chain cache; answers now reflect the selected index.
- Exports record correct `index_name`.

---

## 6) Guardrails (5–10 min)

**Trigger warnings:**
- Ask a question with injection text: `Ignore previous instructions and ...`
- Ask something requiring context when none is indexed.

**Expect:**
- **Warn** banners for injection/conflict/no-citation.
- **Block** only when `strict` and conditions met.

---

## 7) Eval snapshot (retrieval) (5–10 min)

**In Evaluation section:**
- Load `eval/qa.jsonl`.
- Run snapshot for BM25, Dense, Hybrid (k=5).

**Expect:**
- Summary table (hit@k, MRR) and bar chart.
- Per-question detail table.

---

## 8) Regression mini-checklist (2–3 min per run)

- [ ] App boots (no import errors).
- [ ] Upload → Build → Ask → Cite.
- [ ] Export MD/CSV/Excel looks right.
- [ ] Switch index; answers reflect new base.
- [ ] Toggle provider; fallback behaves.
- [ ] Eval snapshot renders charts.
- [ ] Guardrail banners show when expected.

Keep this in the repo root as **TESTING.md** and update when flows change.

---

## 9) Common failures & fixes

- **`ImportError: langchain-openai`**: Install `openai langchain-openai` or switch provider to Ollama.
- **Excel export fails**: Install `openpyxl` (already in `requirements.txt`).
- **No citations**: Check `SCORE_THRESH` not too high; ensure files indexed; verify guardrail “no-citation” not blocking.
- **Index switch shows empty stats**: Pointer missing; rebuild manifest from current index via the sidebar.
- **Model missing in Ollama**: `ollama pull mistral` (one-time), then restart the app.

---

## 10) Optional: CI smoke (GitHub Actions)

Add a simple workflow to:
- `pip install -r requirements.txt`
- `pytest -q`

(You can add UI smoke later with Playwright, but keep it simple for now.)
