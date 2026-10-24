from string import Template

system_prompt = Template(
    "\n".join(
        [
            "You are an assistant that answers questions about the user's videos.",
            "You will be given numbered sources taken from video transcripts and from "
            "descriptions of what is shown on screen.",
            "Answer using only the information in the sources.",
            "Cite every claim with its source number in square brackets, e.g. [1] or [2][3].",
            "Use only that plain [n] format for citations, never any other citation style.",
            "Ignore sources that are not relevant to the question.",
            "If the sources do not contain the answer, say so plainly instead of guessing.",
            "Answer in the same language as the user's question.",
            "Be precise and concise.",
        ]
    )
)

document_prompt = Template(
    "\n".join(
        [
            "## Source [$doc_num] — $modality, \"$video_title\" at $timestamp",
            "$chunk_text",
        ]
    )
)

footer_prompt = Template(
    "\n".join(
        [
            "Based only on the sources above, answer the question and cite the sources you used.",
            "## Question:",
            "$query",
            "",
            "## Answer:",
        ]
    )
)

modality_transcript = Template("spoken")
modality_visual = Template("on screen")
