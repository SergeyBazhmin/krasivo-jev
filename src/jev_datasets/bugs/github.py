# The model stores no facts, so the state has to explain what a GitHub issue is and what each
# label means; otherwise the answer depends on knowing GitHub's triage conventions.
GITHUB_INTRO = (
    "You are a maintainer triaging issues on GitHub, a website where software projects keep their code. Anyone can "
    "open an issue on a project: a short post with a title and a description, used to report a problem, ask for a "
    "change or ask something. Decide which label says what kind of issue this is."
)
