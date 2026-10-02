"""Turn a timestamped transcript into structured local notes with Ollama."""

import argparse
import os
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
from config import (
    CHUNK_OVERLAP_LINES,
    CHUNK_SIZE,
    HF_CACHE_DIR,
    OLLAMA_MODEL_NAME,
    OLLAMA_SERVER_URL,
    OUTPUT_DIR,
)
from ollama_utils import ensure_ollama

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR)

def split_transcript(lines, target_size=CHUNK_SIZE, overlap_lines=CHUNK_OVERLAP_LINES):
    chunks = []
    start = 0
    while start < len(lines):
        end = start
        size = 0
        while end < len(lines) and (size + len(lines[end]) <= target_size or end == start):
            size += len(lines[end])
            end += 1
        chunks.append(lines[start:end])
        if end >= len(lines):
            break
        start = max(start + 1, end - overlap_lines)
    return chunks


def note_language_instruction(notes_lang):
    return (
        "Use English headings. Keep the teacher's Bangla explanations in Bangla where used, "
        "and keep English technical terms as spoken."
    )


def chunk_prompt(transcript_chunk, chunk_number, notes_lang):
    return f"""Write complete, well-organised class notes from transcript chunk {chunk_number}.

Include every example the teacher gives and explain each example step by step in simple language.
Never add facts, examples, definitions, or explanations that are not supported by this transcript.
When a sentence is cut off or half-audible, use only the surrounding context to complete it and
mark the recovered part with [inferred]. If its meaning cannot be recovered, write
[unclear: MM:SS], using the timestamp on the relevant transcript line.
{note_language_instruction(notes_lang)}

Use clear topic headings and preserve useful transcript timestamps. Do not remove or alter any
[inferred] or [unclear: MM:SS] markers.

Transcript:
{transcript_chunk}
"""


def final_prompt(title, chunk_notes, unclear_lines, notes_lang):
    checking_text = "\n".join(unclear_lines) if unclear_lines else "No [inferred] or [unclear: MM:SS] markers were added."
    return f"""Merge these chunk notes into one coherent set of class notes.

Write complete, well-organised notes and keep every example the teacher gave, explained step by
step in simple language. Never add facts, examples, definitions, or explanations that are not in
the transcript or supported chunk notes. Where a sentence was cut off or half-audible, preserve
the context-based completion and its [inferred] marker. Preserve every [inferred] and
[unclear: MM:SS] marker exactly. Use [unclear: MM:SS] for meaning that cannot be recovered.
{note_language_instruction(notes_lang)}

Use this exact structure with Markdown headings:
# {title}
## Topic-wise notes
Organize the material by topic. Include every teacher-provided example with simple step-by-step
explanations and only definitions present in the transcript.
## Key terms
Provide a short glossary using only terms defined or explained in the transcript.
## Summary
Give a short summary.
## Needs checking
List every [inferred] and [unclear: MM:SS] item below, preserving its timestamp and marker, with a
brief reason to review it. Include all marked items supplied below and any other marked items in
the chunk notes. If there are none, say "No inferred or unclear lines were flagged."

Marked items from the chunk notes:
{checking_text}

Chunk notes:
{chunk_notes}
"""


def collect_review_lines(lines):
    marker_pattern = re.compile(r"\[unclear:\s*\d{2}:\d{2}\]", re.IGNORECASE)
    marked_lines = []
    for item in lines:
        for line in item.splitlines():
            if "[inferred]" in line.lower() or marker_pattern.search(line):
                line = re.sub(r"^\s*[-*]\s+", "", line.strip())
                if line not in marked_lines:
                    marked_lines.append(line)
    return marked_lines


def ensure_needs_checking(notes, review_lines):
    review_lines = list(review_lines)
    for line in collect_review_lines([notes]):
        if line not in review_lines:
            review_lines.append(line)

    if not review_lines:
        needs_section = "## Needs checking\n\nNo inferred or unclear lines were flagged."
    else:
        needs_section = "## Needs checking\n\n" + "\n".join(f"- {line}" for line in review_lines)

    match = re.search(r"(?im)^#{1,6}\s+Needs checking\s*$", notes)
    if not match:
        return notes.rstrip() + "\n\n" + needs_section + "\n"

    following_heading = re.search(r"(?m)^#{1,6}\s+", notes[match.end():])
    section_end = match.end() + following_heading.start() if following_heading else len(notes)
    existing_section = notes[match.start():section_end]
    missing_lines = [line for line in review_lines if line not in existing_section]
    if missing_lines:
        notes = notes[:section_end].rstrip() + "\n" + "\n".join(
            f"- {line}" for line in missing_lines
        ) + "\n" + notes[section_end:]
    return notes


def create_notes(transcript_path, notes_lang="mixed"):
    ensure_ollama(OLLAMA_MODEL_NAME)
    import ollama

    transcript_path = Path(transcript_path).resolve()
    if not transcript_path.is_file():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")
    ollama_client = ollama.Client(host=OLLAMA_SERVER_URL)
    transcript = transcript_path.read_text(encoding="utf-8")
    transcript_lines = transcript.splitlines()
    chunks = split_transcript(transcript_lines)
    if not chunks:
        raise ValueError("Transcript is empty.")

    recording = transcript_path.stem
    if recording.lower().endswith("_transcript"):
        recording = recording[:-11]
    title = recording.replace("_", " ").strip().title() or "Class Notes"
    chunks_dir = OUTPUT_DIR / f"{recording}_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for old_chunk in chunks_dir.glob("chunk_*.md"):
        old_chunk.unlink()

    print(f"Writing notes for {len(chunks)} transcript chunk(s) with {OLLAMA_MODEL_NAME}...", flush=True)
    chunk_notes = []
    for index, lines in enumerate(chunks, start=1):
        text = "\n".join(lines)
        result = ollama_client.generate(
            model=OLLAMA_MODEL_NAME,
            prompt=chunk_prompt(text, index, notes_lang),
            options={"temperature": 0.2},
        )
        response_text = result.get("response", "") if isinstance(result, dict) else result.response
        chunk_note_path = chunks_dir / f"chunk_{index:03d}.md"
        chunk_note_path.write_text(response_text.strip() + "\n", encoding="utf-8")
        chunk_notes.append(f"### Chunk {index}\n\n{response_text.strip()}")
        print(f"Finished chunk {index}/{len(chunks)}: {chunk_note_path.name}", flush=True)

    review_lines = collect_review_lines(chunk_notes)
    merged = ollama_client.generate(
        model=OLLAMA_MODEL_NAME,
        prompt=final_prompt(title, "\n\n".join(chunk_notes), review_lines, notes_lang),
        options={"temperature": 0.2},
    )
    notes_text = merged.get("response", "") if isinstance(merged, dict) else merged.response
    notes_text = ensure_needs_checking(notes_text.strip(), review_lines)
    notes_path = OUTPUT_DIR / f"{recording}_notes.md"
    notes_path.write_text(notes_text.strip() + "\n", encoding="utf-8")
    print(f"Notes saved: {notes_path}", flush=True)
    print(f"Intermediate chunk notes saved: {chunks_dir}", flush=True)
    return notes_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", help="Timestamped transcript text file.")
    parser.add_argument("--notes-lang", choices=("mixed", "english"), default="mixed")
    args = parser.parse_args()
    try:
        create_notes(args.transcript, args.notes_lang)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
