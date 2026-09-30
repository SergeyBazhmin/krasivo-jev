import re
from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

from jev_model.prompt import ELLIPSIS

# <state>, <q>, <opt>, </opt>, <decide>. These are rarely used Qwen special tokens, so no
# embedding rows are added; the LoRA gives them their meaning.
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
SPECIAL_PATTERN = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
# the least state worth keeping around the cut; with less room the sample is rejected
MIN_STATE = 16


@dataclass
class Tokens:
    """One sample tokenized piece by piece, so the options can be put in any order without tokenizing again."""

    state: list[int]
    question: list[int]
    options: list[list[int]]


class Encoder:
    """Lays a sample out as `<state> state <q> question <opt> o1 </opt> <opt> o2 </opt> ... <decide>`."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.state_id, self.question_id, self.open_id, self.close_id, self.decide_id = (
            tokenizer.convert_tokens_to_ids(token) for token in SPECIAL
        )
        if len({self.state_id, self.question_id, self.open_id, self.close_id, self.decide_id, tokenizer.unk_token_id}) != 6:
            raise ValueError(f"the tokenizer lacks some of the delimiter tokens {SPECIAL}")
        self.ellipsis = tokenizer.encode(ELLIPSIS, add_special_tokens=False)

    def tokenize(self, samples: list[tuple[str, str, list[str]]]) -> list[Tokens]:
        """Tokens of each (state, question, options)."""
        texts = [text for state, question, options in samples for text in (state, question, *options)]
        # `<|name|>` in a sample's text must stay text: a real delimiter there would forge an option boundary
        texts = [SPECIAL_PATTERN.sub(r"<¦\1¦>", text) for text in texts]
        ids = iter(self.tokenizer(texts, add_special_tokens=False).input_ids if texts else [])
        return [Tokens(next(ids), next(ids), [next(ids) for _ in options]) for _, _, options in samples]

    def query_length(self, tokens: Tokens) -> int:
        """Tokens of everything but the state text."""
        return 3 + len(tokens.question) + sum(len(option) + 2 for option in tokens.options)

    def fits(self, tokens: Tokens) -> bool:
        room = self.max_length - self.query_length(tokens)
        return room >= min(len(tokens.state), MIN_STATE + len(self.ellipsis))

    def length(self, tokens: Tokens) -> int:
        return min(self.max_length, self.query_length(tokens) + len(tokens.state))

    def assemble(self, tokens: Tokens, order: list[int]) -> tuple[list[int], list[int]]:
        """(ids, the position of each option's closing token) with the options in `order`. The
        `<decide>` token is the last one. A sequence over `max_length` loses the middle of its
        state, since the question and the options must always be seen whole."""
        state = tokens.state
        room = self.max_length - self.query_length(tokens)
        if len(state) > room:
            keep = room - len(self.ellipsis)
            head = keep // 2
            state = state[:head] + self.ellipsis + state[len(state) - (keep - head):]
        ids = [self.state_id, *state, self.question_id, *tokens.question]
        closes = []
        for i in order:
            ids += [self.open_id, *tokens.options[i], self.close_id]
            closes.append(len(ids) - 1)
        ids.append(self.decide_id)
        return ids, closes
