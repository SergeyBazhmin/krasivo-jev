from transformers import PreTrainedTokenizerBase

# Options are numbered so that position i in the prompt is logit i of the head.
# The prompt ends on the answer cue, so the last token's hidden state is the one that
# "decides".
STATE_TEMPLATE = "{state}\n\n"
QUERY_TEMPLATE = "Question: {question}\nOptions:\n{options}\nAnswer:"
ELLIPSIS = "\n...\n"


def render_query(question: str, options: list[str]) -> str:
    listed = "\n".join(f"{i}) {text}" for i, text in enumerate(options, start=1))
    return QUERY_TEMPLATE.format(question=question, options=listed)


def encode(
    tokenizer: PreTrainedTokenizerBase, state: str, question: str, options: list[str], max_length: int
) -> list[int] | None:
    """Token ids of the prompt. When it is too long, the middle of the state is cut, since the
    question and the options must always be seen whole. Returns None if they don't fit on their own."""
    query = tokenizer.encode(render_query(question, options), add_special_tokens=False)
    if not state:
        return query if len(query) <= max_length else None
    context = tokenizer.encode(STATE_TEMPLATE.format(state=state), add_special_tokens=False)
    budget = max_length - len(query)
    if len(context) > budget:
        ellipsis = tokenizer.encode(ELLIPSIS, add_special_tokens=False)
        keep = budget - len(ellipsis)
        if keep < 16:
            return None
        head = keep // 2
        context = context[:head] + ellipsis + context[len(context) - (keep - head):]
    return context + query
