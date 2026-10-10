"""The unary compositional ProntoQA subset in RuleCollection, with checked binary labels."""

import re
from collections import Counter

from datasets import DatasetDict, load_dataset
from loguru import logger

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample, pick_question, with_context

PROMPT = re.compile(
    r"Rules: (.*?)\nFacts: (.*?)\nQuestion: (.*?)\? Please answer the question based on the given rule and facts",
    re.DOTALL,
)
ANSWER = re.compile(r"<answer>(True|False)</answer>")
CLAUSE_SEPARATOR = re.compile(r"\.\s*|(?:,\s*(?:and\s+)?|\s+and\s+)(?=[A-Z][a-z]+ is )")
QUESTIONS = [
    "Given the rules and facts, is the statement true or false?",
    "Do the rules and facts prove the statement true or prove it false?",
    "Based only on the rules and facts, what is the truth value of the statement?",
    "Is the statement supported or contradicted by the rules and facts?",
    "Taking the rules and facts as given, is the statement true or false?",
]
ABOUT = (
    "Use only the stated rules and facts. Concept names and properties have no additional meaning here. "
    "Rules are logical implications; 'or' is inclusive. You may also reason by contradiction: if A implies B "
    "and B is false, A is false. Missing evidence alone does not make a statement false."
)
ANSWERS = {"True": "the statement follows from the rules and facts", "False": "its negation follows"}
OPTIONS = make_options(["True", "False"])
type CNF = list[frozenset[str]]
type ContextKey = tuple[str, tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]


def negate(literal: str) -> str:
    return literal[1:] if literal.startswith("~") else f"~{literal}"


def predicates(text: str) -> tuple[str, list[str]]:
    """Parse only the source's flat unary conjunctions/disjunctions, never arbitrary English."""
    text = text.strip().rstrip(".").lower()
    negative = text.startswith("not ")
    text = text.removeprefix("not ")
    op = "or" if " or " in text else "and"
    if " or " in text and " and " in text:
        raise ValueError("mixed coordination")
    atoms = []
    for token in re.split(r",\s*(?:(?:and|or)\s+)?|\s+(?:and|or)\s+", text):
        token = re.sub(r"^(?:a|an) ", "", token.strip())
        if not re.fullmatch(r"[a-z]+", token) or token in {"and", "or", "not"}:
            raise ValueError("unsupported predicate")
        atoms.append(token[:-2] if token.endswith("puses") else token)
    # Without logical forms, "not A and/or B" has ambiguous scope. Do not invent a reading or a label.
    if negative and len(atoms) > 1:
        raise ValueError("ambiguous negation")
    return op, [negate(a) for a in atoms] if negative else atoms


def as_cnf(op: str, atoms: list[str]) -> CNF:
    return [frozenset([a]) for a in atoms] if op == "and" else [frozenset(atoms)]


def ground_clauses(text: str) -> tuple[str, CNF]:
    subjects = set()
    clauses = []
    for sentence in CLAUSE_SEPARATOR.split(text.strip().rstrip(".")):
        name, expr = sentence.strip().split(" is ", 1)
        if not re.fullmatch(r"[A-Z][a-z]+", name):
            raise ValueError("unsupported subject")
        subjects.add(name)
        clauses.extend(as_cnf(*predicates(expr)))
    if len(subjects) != 1:
        raise ValueError("multiple subjects")
    return subjects.pop(), clauses


def parse_prompt(prompt: str) -> tuple[str, str, str, CNF, CNF, CNF, ContextKey]:
    match = PROMPT.match(prompt)
    if match is None:
        raise ValueError("unsupported prompt")
    rules, facts, statement = match.groups()
    subject, fact_clauses = ground_clauses(facts)
    # All 999 such source questions refer to Polly in the facts; repair only this observed corruption.
    if subject == "Polly" and statement.startswith("lly is "):
        statement = "Po" + statement
    query_subject, query = ground_clauses(statement)
    if query_subject != subject:
        raise ValueError("subject mismatch")
    rule_clauses = []
    for sentence in rules.strip().rstrip(".").split("."):
        sentence = sentence.strip()
        if sentence.startswith("Everything that is "):
            left, right = sentence.removeprefix("Everything that is ").split(" is ")
        elif sentence.startswith(("Each ", "Every ")):
            left, right = sentence.split(" ", 1)[1].split(" is ")
        else:
            left, right = sentence.split(" are ")
        op, atoms = predicates(left)
        negated_left = as_cnf("or" if op == "and" else "and", [negate(a) for a in atoms])
        rule_clauses.extend(a | b for a in negated_left for b in as_cnf(*predicates(right)))
    # Group paraphrases and reordered conjuncts/rules by logical structure, keeping the entity identity.
    key = subject, canonical(rule_clauses), canonical(fact_clauses)
    return rules, facts, statement, rule_clauses, fact_clauses, query, key


def canonical(clauses: CNF) -> tuple[tuple[str, ...], ...]:
    return tuple(sorted({tuple(sorted(c)) for c in clauses}))


def satisfiable(clauses: CNF) -> bool:
    """Complete DPLL for these small unary theories; unit propagation includes contraposition."""
    clauses = [c for c in clauses if not any(negate(a) in c for a in c)]
    while clauses:
        if any(not c for c in clauses):
            return False
        units = {next(iter(c)) for c in clauses if len(c) == 1}
        if any(negate(a) in units for a in units):
            return False
        if not units:
            literal = min(min(clauses, key=lambda c: (len(c), tuple(sorted(c)))))
            return satisfiable(clauses + [frozenset([literal])]) or satisfiable(
                clauses + [frozenset([negate(literal)])]
            )
        opposite = {negate(a) for a in units}
        clauses = [c - opposite for c in clauses if c.isdisjoint(units)]
    return True


def truth_value(theory: CNF, query: CNF) -> str:
    if not satisfiable(theory):
        return "inconsistent"
    if all(not satisfiable(theory + [frozenset([negate(a)]) for a in c]) for c in query):
        return "True"
    if not satisfiable(theory + query):
        return "False"
    return "Unknown"


def context_state(rules: str, facts: str) -> str:
    return with_context(ABOUT, f"Rules: {rules.strip()}\nFacts: {facts.strip().rstrip('.')}.", answers=ANSWERS)


def make_prontoqa_sample(row: dict, states: dict[ContextKey, str]) -> dict:
    prompt = row["prompt"][0]["content"]
    _, _, statement, *_, key = parse_prompt(prompt)
    answer = ANSWER.fullmatch(row["reward_model"]["ground_truth"])[1]
    return make_sample(
        states[key],
        f"Statement: {statement.rstrip('.')}.\n{pick_question(QUESTIONS, prompt)}",
        OPTIONS,
        answer,
    )


class ProntoQADataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        data = load_dataset(self.hf_path)
        return DatasetDict(
            {split: data[split].filter(lambda row: row["data_source"] == "ProntoQA") for split in ("train", "test")}
        )

    def prepare(self) -> None:
        raw = self.data
        seen_contexts: set[ContextKey] = set()
        states: dict[ContextKey, str] = {}
        retained = {}
        # Preserve valid held-out examples; the old shared loader preferred train and shrank test.
        for split in ("test", "train"):
            counts = Counter()
            keep = []
            split_contexts = set()
            seen_inputs = set()
            for i, row in enumerate(raw[split]):
                try:
                    rule_text, fact_text, _, rules, facts, query, context = parse_prompt(row["prompt"][0]["content"])
                except ValueError as error:
                    counts[str(error)] += 1
                    continue
                match = ANSWER.fullmatch(row["reward_model"]["ground_truth"])
                result = truth_value(rules + facts, query)
                if match is None or result != match[1]:
                    counts[result if result in {"Unknown", "inconsistent"} else "wrong label"] += 1
                    continue
                key = context, canonical(query)
                if context in seen_contexts or key in seen_inputs:
                    counts["overlapping context" if context in seen_contexts else "duplicate input"] += 1
                    continue
                seen_inputs.add(key)
                split_contexts.add(context)
                # One original rendering per logical context also keeps hash-carved validation disjoint.
                states.setdefault(context, context_state(rule_text, fact_text))
                keep.append(i)
            seen_contexts.update(split_contexts)
            retained[split] = raw[split].select(keep)
            logger.info("prontoqa {}: kept {}/{}; dropped {}", split, len(keep), len(raw[split]), dict(counts))
        self.data = DatasetDict({split: retained[split] for split in raw}).map(
            lambda row: make_prontoqa_sample(row, states), remove_columns=self.source_columns, features=SAMPLE_FEATURES
        )


prontoqa_dataset = ProntoQADataset(name="prontoqa", hf_path="RuleReasoner/RuleCollection-32K")
