# The model stores no facts, so the state has to explain what a GitHub issue is and what each
# label means; otherwise the answer depends on knowing GitHub's triage conventions.
GITHUB_INTRO = (
    "GitHub is a website where software projects keep their code. Anyone can open an issue on a "
    "project: a short post with a title and a description, used to report a problem, ask for a change "
    "or ask something. Maintainers then triage the issue by giving it a label that says what kind of "
    "issue it is."
)


def with_github_context(issue: str, labels: dict[str, str]) -> str:
    """The issue text preceded by the GitHub intro and a definition of each label the answer can be."""
    guide = "\n".join(f"- {label}: {meaning}" for label, meaning in labels.items())
    return f"{GITHUB_INTRO}\n\nLabels:\n{guide}\n\nIssue:\n{issue}"
