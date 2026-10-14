# The model stores no facts, so the state has to explain what a GitHub issue is and what each
# label means; otherwise the answer depends on knowing GitHub's triage conventions.
GITHUB_INTRO = [
    (
        "You are a maintainer triaging issues on GitHub, a website where software projects keep their code. Anyone "
        "can open an issue on a project: a short post with a title and a description, used to report a problem, "
        "ask for a change or ask something. Decide which label says what kind of issue this is."
    ),
    (
        "You triage incoming issues for a project on GitHub, where software projects host their code. An issue is "
        "a short post with a title and a description that anyone can open to report a problem, request a change or "
        "ask a question. Pick the label that says what kind of issue this one is."
    ),
    (
        "Below is an issue from a GitHub project. GitHub is a website that hosts software code, and anyone may "
        "open an issue there: a titled post that reports a problem, asks for a change or asks a question. As the "
        "project's maintainer, choose the label that fits the kind of issue it is."
    ),
    (
        "As a maintainer of a software project on GitHub (a website that hosts code), you sort the issues people "
        "open. An issue has a title and a description and is used to report a problem, ask for a change, or ask "
        "something. Decide which label describes this issue's kind."
    ),
    (
        "You are given an issue opened on a GitHub project. On GitHub, where software projects keep their code, "
        "anyone can post an issue, with a title and a description, to report a problem, request a change or ask a "
        "question. Label the issue by its kind, as a maintainer triaging it would."
    ),
]
