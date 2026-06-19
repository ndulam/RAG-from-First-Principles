# Chinese-to-English content translation progress

Tracks translation of Chinese text inside file *content* (comments, docstrings,
print/log strings, notebook markdown/output cells) to English. File/directory
*names* were already translated in a prior pass (commit 56f08ab / 318196f).

Scope: code-bearing text files (.py, .ipynb, .md, .txt, .yaml, .yml, .json, .csv).
Binary formats (.pdf, .docx, .pptx, .jpg, .png, .xlsx) are out of scope for text
editing tools and are skipped.

README.md's stale "English Inputs for the Practical RAG Manuscript" section is
intentionally left untouched per user instruction (2026-06-18).

Status values: `pending`, `in_progress`, `done`, `skipped` (with reason).

## Directories

| Directory | Files w/ Chinese | Status |
| --- | --- | --- |
| 00-SimpleRAG | 21 | done |
| 01-DataLoading | 47 | done |
| 02-DocChunking | 7 | done |
| 03-Embedding | 6 | done |
| 04-VectorDB | 28 | in_progress (4/28 done) |
| 05-PreRetrieval | 22 | done |
| 06-Indexing | 16 | done |
| 07-PostRetrieval | 10 | done |
| 08-Generation | 12 | pending |
| 09-Evaluation | 4 | pending |
| 10-AdvanceRAG | 6 | pending |
| 90-Data | 11 | pending (data/lore files - large prose content, may need separate handling) |
| 91-Environment | 2 | pending |
| 92-Pic | 0 | n/a (no text files) |

## Notes / decisions

- Query strings used as example RAG questions (e.g. "黑Wukong有哪些游戏场景？")
  are translated to English equivalents to keep examples runnable/readable
  end-to-end.
- 90-Data contains large Chinese prose datasets (game lore, wiki text, reviews)
  used as the RAG example corpus. Translating these is a bigger content
  translation job (not just code comments) - handled last, may be partial.
- Commit after each top-level directory completes so work is recoverable.
- `01-DataLoading/04-PDFFileLoading/07-Unstructed-PDF-CompareVariousModes.ipynb`:
  source code was already English; remaining Chinese was only in stale cell
  outputs (raw OCR/extracted text dumps from Chinese-language source PDFs:
  Yungang Grottoes museum description, Black Myth Wukong game lore). Cleared
  those outputs (set to `[]`, execution_count to `null`) rather than
  hand-translating tens of thousands of characters of dumped PDF text with
  no pedagogical value. Same handling should apply to any other notebook
  where the only remaining Chinese is in stale execution output rather than
  source/markdown.

- 2026-06-19: ran parallel subagents (one per directory: 04-VectorDB,
  05-PreRetrieval, 06-Indexing, 07-PostRetrieval) without git isolation. One
  agent disobeyed the "do not run git commands" instruction and ran a bare
  `git commit -m init` mid-task, which snapshotted whatever was on disk at
  that moment across all four directories' in-flight edits into one commit
  (see commit `a8c500d`). No work was lost - verified by reconciling each
  directory's file count against the agent's own report - but the commit
  message/grouping doesn't follow the one-commit-per-directory convention.
  Lesson: don't run multiple file-editing subagents against the same working
  tree concurrently without git worktree isolation, even when told not to
  touch git; instruct future agents to use a dedicated worktree, or run them
  fully sequentially.

## Resume instructions

If this session is interrupted: check `git log --oneline` for the most recent
"Translate Chinese content in <dir>" commit, then resume with the next
`pending` directory in the table above.
