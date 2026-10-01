import re

from transformers import PreTrainedTokenizerBase

from jev_model.prompt import cut_middle

SYSTEM = "Choose one option. Reply with its number only."
# room left in `max_length` for the system message and the chat template around the user turn
TEMPLATE_TOKENS = 64


class Prompter:
    """The user message of a sample: state, question and the numbered options. The tokenizer only
    measures it, so a long state can lose its middle before the server sees it."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int):
        self.tokenizer = tokenizer
        self.budget = max_length - TEMPLATE_TOKENS - len(self.encode(SYSTEM))

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def __call__(self, state: str, question: str, options: list[str]) -> str | None:
        """None when the question and options alone do not fit."""
        listed = "\n".join(f"{i}. {text}" for i, text in enumerate(options, start=1))
        query = f"{question}\n\n{listed}"
        room = self.budget - len(self.encode(query))
        if not state:
            return query if room >= 0 else None
        context = cut_middle(self.tokenizer, self.encode(f"{state}\n\n"), room)
        return None if context is None else self.tokenizer.decode(context) + query


# the number at the start of an answer ("2", "(2)", "2. text"), or after "answer" anywhere ("The answer is 2")
NUMBER = re.compile(r"[(\[]?(\d+)(?!\d)")
ANSWER_NUMBER = re.compile(r"answer(?:\s+is)?\s*:?\s*[(\[]?(\d+)(?!\d)", re.IGNORECASE)


def parse(answer: str, options: list[str]) -> int | None:
    """The index of the option an answer names by its number; None when it names none in range."""
    found = NUMBER.match(answer.strip()) or ANSWER_NUMBER.search(answer)
    if found and 1 <= (number := int(found.group(1))) <= len(options):
        return number - 1
    return None
