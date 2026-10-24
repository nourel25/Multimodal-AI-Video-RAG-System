"""Streamlit front end: manage a video library, follow processing jobs, chat with citations.

    streamlit run ui/app.py
"""

import os
import re

import streamlit as st

from api_client import APIError, RagClient

st.set_page_config(page_title="Video RAG", page_icon="🎬", layout="wide")

STATUS_BADGE = {"pending": ("gray", "pending"), "processing": ("orange", "processing"),
                "indexed": ("green", "indexed"), "failed": ("red", "failed")}

# ---------- sidebar: connection ----------
with st.sidebar:
    st.title("🎬 Video RAG")
    base_url = st.text_input("API URL", os.getenv("API_BASE_URL", "http://localhost:8000"))
    api_key = st.text_input("API key", os.getenv("API_KEY", ""), type="password")
    user_id = st.text_input("User id", "demo", help="Letters, numbers and underscores")
    client = RagClient(base_url, api_key)

    try:
        health = client.health()
        st.success(f"API {health['status']} · v{health['app_version']}")
        with st.expander("Backends"):
            st.json(health["config"])
    except Exception as exc:  # the API may simply not be running yet
        st.error(f"API unreachable: {exc}")
        st.stop()

    st.divider()
    st.caption("Retrieval")
    mode = st.selectbox("Mode", ["hybrid", "dense", "sparse"])
    rerank = st.toggle("Rerank", value=True)
    limit = st.slider("Sources", 1, 10, 5)

st.session_state.setdefault("messages", [])
st.session_state.setdefault("player", None)


def show_error(exc: Exception) -> None:
    if isinstance(exc, APIError):
        st.error(f"{exc.signal}: {exc.detail or ''}")
    else:
        st.error(str(exc))


def _render_source_list(sources: list[dict], key_prefix: str) -> None:
    for source in sources:
        icon = "🖼️" if source["modality"] == "visual" else "🗣️"
        label = f"[{source['index']}] {icon} {source['video_title']} @ {source['timestamp']}"
        with st.expander(label):
            st.write(source["text"])
            cols = st.columns([2, 2, 3])
            if cols[0].button("▶ Play here", key=f"{key_prefix}-{source['index']}"):
                st.session_state.player = (source["youtube_video_id"], int(source["start"]))
                st.rerun()
            cols[1].link_button("YouTube", source["url"])
            cols[2].caption(f"score {source['score']:.3f}")


def render_sources(sources: list[dict], key_prefix: str, answer: str | None = None) -> None:
    """Show the sources the answer cites; keep the rest out of the way."""
    if answer is None:
        _render_source_list(sources, key_prefix)
        return
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    used = [s for s in sources if s["index"] in cited]
    unused = [s for s in sources if s["index"] not in cited]
    _render_source_list(used, key_prefix)
    if unused:
        with st.popover(f"{len(unused)} retrieved but not cited"):
            _render_source_list(unused, f"{key_prefix}-unused")


library_tab, chat_tab, search_tab = st.tabs(["📚 Library", "💬 Ask", "🔎 Retrieval playground"])

# ---------- library ----------
with library_tab:
    with st.form("add", clear_on_submit=True):
        urls = st.text_area("YouTube URLs (one per line)")
        c1, c2 = st.columns(2)
        auto_process = c1.checkbox("Process immediately", value=True)
        include_visual = c2.checkbox("Analyse frames (needs VISION_BACKEND)", value=True)
        if st.form_submit_button("Add videos") and urls.strip():
            try:
                result = client.add_videos(user_id, urls.split(), auto_process, include_visual)
                st.success(f"Added {len(result['added'])} video(s)")
                for skipped in result["skipped"]:
                    st.warning(f"{skipped['url']}: {skipped['signal']}")
            except Exception as exc:
                show_error(exc)

    @st.fragment(run_every=3)
    def jobs_and_videos():
        jobs = client.list_jobs(user_id)
        for job in [j for j in jobs if j["status"] in ("queued", "running")]:
            st.progress(job["progress"] / 100,
                        text=f"Job {job['job_id'][-6:]} · {job['stage']} · {job['progress']:.0f}%")

        videos = client.list_videos(user_id)
        if not videos:
            st.info("No videos yet — add some above.")
            return
        for video in videos:
            with st.container(border=True):
                thumb, info, actions = st.columns([1, 4, 1])
                if video["thumbnail_url"]:
                    thumb.image(video["thumbnail_url"])
                color, text = STATUS_BADGE[video["status"]]
                info.markdown(f"**{video['title'] or video['youtube_video_id']}**  "
                              f":{color}-badge[{text}]")
                minutes, seconds = divmod(int(video["duration_seconds"]), 60)
                info.caption(f"{video['channel']} · {minutes}:{seconds:02d} · "
                             f"{video['transcript_chunks']} transcript / "
                             f"{video['visual_chunks']} visual chunks")
                if video["error"]:
                    info.caption(f":red[{video['error'][:200]}]")
                if video["status"] in ("pending", "failed", "indexed") and actions.button(
                        "Process" if video["status"] != "indexed" else "Re-process",
                        key=f"p-{video['video_id']}"):
                    try:
                        client.process(user_id, [video["video_id"]],
                                       reprocess=video["status"] == "indexed")
                    except Exception as exc:
                        show_error(exc)
                if actions.button("Delete", key=f"d-{video['video_id']}"):
                    client.delete_video(user_id, video["video_id"])
                    st.rerun()

    jobs_and_videos()

# ---------- chat ----------
with chat_tab:
    chat_col, player_col = st.columns([3, 2])

    with player_col:
        if st.session_state.player:
            youtube_id, start = st.session_state.player
            st.video(f"https://www.youtube.com/watch?v={youtube_id}", start_time=start)
            st.caption(f"Playing from {start // 60}:{start % 60:02d}")
        else:
            st.info("Click ▶ on a source to jump to that moment.")

    with chat_col:
        if st.button("New conversation"):
            st.session_state.messages = []
        # Messages go in a container above the input, so a new answer streams in place.
        conversation = st.container()
        with conversation:
            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message.get("sources"):
                        render_sources(message["sources"], f"h{i}", message["content"])

        if question := st.chat_input("Ask about your videos…"):
            history = [{"role": m["role"], "content": m["content"]}
                       for m in st.session_state.messages]
            st.session_state.messages.append({"role": "user", "content": question})
            # Same key prefix as the history loop will use, so "Play here" clicks on a fresh
            # answer survive the rerun they trigger.
            answer_key = f"h{len(st.session_state.messages)}"
            with conversation, st.chat_message("user"):
                st.markdown(question)

            with conversation, st.chat_message("assistant"):
                sources: list[dict] = []
                errors: list[str] = []

                def tokens():
                    for kind, payload in client.stream_answer(
                            user_id, question, history, limit=limit, mode=mode, rerank=rerank):
                        if kind == "sources":
                            sources.extend(payload)
                        elif kind == "token":
                            yield payload
                        else:
                            errors.append(payload)

                try:
                    answer = st.write_stream(tokens())
                except Exception as exc:
                    show_error(exc)
                    answer = ""
                if errors:
                    st.warning("No matching content found." if "no_relevant_context" in errors
                               else f"Generation failed: {errors[0]}")
                render_sources(sources, answer_key, answer or "")
            st.session_state.messages.append(
                {"role": "assistant", "content": answer or "", "sources": sources})

# ---------- retrieval playground ----------
with search_tab:
    st.caption("Compare what each retrieval strategy returns for the same query.")
    query = st.text_input("Query")
    if query:
        columns = st.columns(4)
        configs = [("Dense", "dense", False), ("Sparse (BM25)", "sparse", False),
                   ("Hybrid", "hybrid", False), ("Hybrid + rerank", "hybrid", True)]
        for column, (title, search_mode, use_rerank) in zip(columns, configs, strict=True):
            with column:
                st.subheader(title)
                try:
                    results = client.search(user_id, query, limit=limit, mode=search_mode,
                                            rerank=use_rerank)
                except Exception as exc:
                    show_error(exc)
                    continue
                for r in results:
                    st.markdown(f"**{r['video_title'][:40]}** @ [{r['timestamp']}]({r['url']})")
                    st.caption(r["text"][:220])
