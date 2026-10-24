"""Download and transcribe the evaluation videos into eval/corpus/<video_id>.json.

Uses the same VideoController and transcription backend as the app (TRANSCRIPTION_BACKEND),
so the corpus reflects what production indexing sees.

    python eval/build_corpus.py                 # every video in eval/datasets/videos.json
    python eval/build_corpus.py --force         # re-transcribe existing files
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from controllers.VideoController import VideoController  # noqa: E402
from helpers.config import Settings  # noqa: E402
from stores.transcription.TranscriptionProviderFactory import (  # noqa: E402
    TranscriptionProviderFactory,
)

CORPUS_DIR = ROOT / "eval" / "corpus"
VIDEOS_FILE = ROOT / "eval" / "datasets" / "videos.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    settings = Settings(_env_file=str(ROOT / "src" / ".env"),
                        ASSETS_DIR=str(ROOT / "eval" / ".media"))
    downloader = VideoController(settings)
    transcriber = TranscriptionProviderFactory(settings).create()
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    for video in json.loads(VIDEOS_FILE.read_text(encoding="utf-8")):
        target = CORPUS_DIR / f"{video['youtube_video_id']}.json"
        if target.exists() and not args.force:
            print(f"skip {video['youtube_video_id']} (exists)")
            continue

        started = time.time()
        metadata = downloader.fetch_metadata(video["youtube_video_id"])
        audio_path = downloader.download_audio(video["youtube_video_id"], "eval")
        result = transcriber.transcribe(audio_path, language=video.get("language"))
        os.remove(audio_path)

        target.write_text(json.dumps({
            "youtube_video_id": video["youtube_video_id"],
            "title": metadata.title,
            "duration_seconds": metadata.duration_seconds,
            **result.to_dict(),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{video['youtube_video_id']}: {len(result.segments)} segments, "
              f"{metadata.duration_seconds:.0f}s video in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
