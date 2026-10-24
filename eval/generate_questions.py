"""Draft extra evaluation questions with an LLM from random transcript windows.

Output goes to eval/datasets/questions.generated.jsonl for manual review — only reviewed
questions should be copied into questions.jsonl.

    python eval/generate_questions.py --per-video 5
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from controllers.ChunkController import ChunkController  # noqa: E402
from helpers.config import Settings  # noqa: E402
from stores.llm.LLMProviderFactory import LLMProviderFactory  # noqa: E402
from stores.transcription import TranscriptSegment  # noqa: E402

PROMPT = """Write one question that a viewer could ask and that is answered ONLY by the passage
below (not by general knowledge). Use the passage's language ({language}). Do not quote the
passage word for word. Reply with the question only.

Passage:
{passage}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-video", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    settings = Settings(_env_file=str(ROOT / "src" / ".env"))
    llm = LLMProviderFactory(settings).create_generation_client()
    out_path = ROOT / "eval" / "datasets" / "questions.generated.jsonl"

    with out_path.open("w", encoding="utf-8") as out:
        for path in sorted((ROOT / "eval" / "corpus").glob("*.json")):
            doc = json.loads(path.read_text(encoding="utf-8"))
            segments = [TranscriptSegment(**s) for s in doc["segments"]]
            windows = ChunkController().build_chunks(segments, max_chars=400, overlap_segments=0)
            for i, window in enumerate(random.sample(windows, min(args.per_video, len(windows)))):
                question = llm.generate([{"role": "user", "content": PROMPT.format(
                    language=doc.get("language") or "en", passage=window.text)}]).strip()
                out.write(json.dumps({
                    "id": f"gen-{doc['youtube_video_id']}-{i}",
                    "video_id": doc["youtube_video_id"],
                    "language": doc.get("language") or "en",
                    "question": question,
                    "start": round(window.start, 1),
                    "end": round(window.end, 1),
                    "passage": window.text,
                }, ensure_ascii=False) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
