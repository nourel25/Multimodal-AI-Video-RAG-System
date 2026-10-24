from string import Template

frame_prompt = Template(
    "\n".join(
        [
            "This is a frame from a video titled \"$video_title\".",
            "Describe what is shown in 2-4 factual sentences: people, objects, diagrams, "
            "slides, code or charts.",
            "Then transcribe any readable on-screen text verbatim after the label 'Text:'.",
            "If the frame has no meaningful content (blank, blurry, transition), reply only: NONE",
        ]
    )
)
