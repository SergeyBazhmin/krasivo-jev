import ast
import operator
import random
import re
from collections.abc import Callable

from datasets import DatasetDict, load_dataset

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.utils import (
    SAMPLE_FEATURES,
    explode,
    make_options,
    make_sample,
    pick_question,
    stable_shuffle,
    with_context,
)

ARITHMETIC_QUESTIONS = [
    "Чему равно значение выражения?",
    "Какой результат получится при вычислении?",
    "Выберите значение данного выражения.",
    "Каков результат вычисления выражения?",
    "Какой из вариантов равен результату?",
]
ARITHMETIC_ABOUT = (
    "Вычислите арифметическое выражение. Сначала выполняются действия в скобках, затем умножение, "
    "затем сложение и вычитание слева направо. Минус перед числом меняет его знак."
)
RULE_QUESTIONS = [
    "Каким числом нужно завершить последнюю строку?",
    "Выберите результат последней строки по общему правилу.",
    "Какой результат соответствует последнему примеру?",
    "Что должно стоять после стрелки в последней строке?",
    "Примените правило примеров к последней строке.",
]
RULE_ABOUT = (
    "В каждой строке вычисляется записанная арифметическая операция. После этого к результату "
    "либо прибавляется 1, либо ничего не прибавляется. Одно и то же правило действует для всех строк. "
    "Определите правило по завершённым примерам и примените его к последней строке."
)
BPS_QUESTIONS = [
    "Сбалансирована ли последовательность скобок?",
    "Правильно ли открываются и закрываются все скобки?",
    "Является ли эта скобочная последовательность правильной?",
    "Можно ли сопоставить каждой скобке парную без нарушения вложенности?",
    "Соблюдены ли правила вложенности и парности скобок?",
]
BPS_ABOUT = (
    "Даны скобки трёх видов: (), [] и {}. Закрывающая скобка должна соответствовать последней ещё не закрытой "
    "открывающей скобке того же вида. В конце не должно оставаться незакрытых скобок. Пробелы игнорируются."
)
LCS_QUESTIONS = [
    "Какова длина наибольшей общей подпоследовательности двух строк?",
    "Сколько символов содержит самая длинная общая подпоследовательность?",
    "Выберите длину наибольшей подпоследовательности, общей для обеих строк.",
    "Какой максимальной длины может быть общая подпоследовательность?",
    "Найдите длину самой длинной общей подпоследовательности этих строк.",
]
LCS_ABOUT = (
    "Подпоследовательность получается удалением любых символов строки без изменения порядка оставшихся. "
    "Она может содержать пропуски. Найдите максимальную длину последовательности символов, "
    "которая является подпоследовательностью обеих строк."
)
MATH_QUESTIONS = [
    "Решите задачу и выберите правильный ответ.",
    "Какой ответ соответствует условиям задачи?",
    "Выберите результат, следующий из условия.",
    "Какой из вариантов является решением задачи?",
    "Ответьте на вопрос задачи, используя приведённые условия.",
]


def arithmetic_value(expression: str) -> int:
    """Evaluate only the integer +, - and * grammar used by these sources, never arbitrary Python."""
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}

    def visit(node: ast.AST) -> int:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError(f"unsupported arithmetic: {expression}")

    return visit(ast.parse(expression.strip(), mode="eval").body)


def numeric_sample(state: str, question: str, answer: int, seed: str, lower: int | None = None) -> dict:
    # Neighbouring integers are distinct, demonstrably incorrect answers; order depends only on the problem.
    start = answer - random.Random(f"distractors:{seed}").randrange(5)
    values = list(range(start, start + 5))
    if lower is not None:
        values = list(range(max(lower, start), max(lower, start) + 5))
    ids = list("ABCDE")
    options = stable_shuffle(make_options(ids, [str(value) for value in values]), seed)
    return make_sample(state, question, options, ids[values.index(answer)])


def arithmetic_samples(row: dict) -> list[dict]:
    if not re.fullmatch(r"-?\d+", row["outputs"]):
        return []
    expression = row["inputs"].split("=")[0].strip()
    try:
        value = arithmetic_value(expression)
    except (SyntaxError, ValueError):
        return []
    # Discard unsupported expressions or inconsistent source labels rather than inventing an answer.
    if value != int(row["outputs"]):
        return []
    state = with_context(ARITHMETIC_ABOUT, expression, "Выражение")
    return [numeric_sample(state, pick_question(ARITHMETIC_QUESTIONS, expression), value, expression)]


def rumodar_samples(row: dict) -> list[dict]:
    if not re.fullmatch(r"-?\d+", row["outputs"]):
        return []
    lines = row["inputs"].splitlines()
    offsets = set()
    try:
        for line in lines[:-1]:
            expression, result = line.split("->")
            offsets.add(int(result.strip()) - arithmetic_value(expression))
        if len(offsets) != 1 or not offsets <= {0, 1}:
            return []
        value = arithmetic_value(lines[-1].split("->")[0]) + next(iter(offsets))
    except (SyntaxError, ValueError):
        return []
    if value != int(row["outputs"]):
        return []
    # The restricted rule family makes the few-shot inference well-defined; arbitrary patterns do not.
    state = with_context(RULE_ABOUT, row["inputs"], "Примеры")
    return [numeric_sample(state, pick_question(RULE_QUESTIONS, row["inputs"]), value, row["inputs"])]


def bps_samples(row: dict) -> list[dict]:
    if row["outputs"] not in ("0", "1"):
        return []
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    valid = True
    for char in row["inputs"].replace(" ", ""):
        if char in "([{":
            stack.append(char)
        elif char not in pairs or not stack or stack.pop() != pairs[char]:
            valid = False
            break
    valid = valid and not stack
    if valid != (row["outputs"] == "1"):
        return []
    return [
        make_sample(
            with_context(BPS_ABOUT, row["inputs"], "Последовательность"),
            pick_question(BPS_QUESTIONS, row["inputs"]),
            make_options(["yes", "no"], ["да", "нет"]),
            "yes" if valid else "no",
        )
    ]


def lcs_samples(row: dict) -> list[dict]:
    if not row["outputs"].isdigit() or len(strings := row["inputs"].split()) != 2:
        return []
    first, second = strings
    previous = [0] * (len(second) + 1)
    for a in first:
        current = [0]
        for i, b in enumerate(second):
            current.append(previous[i] + 1 if a == b else max(previous[i + 1], current[-1]))
        previous = current
    value = previous[-1]
    if value != int(row["outputs"]):
        return []
    state = with_context(LCS_ABOUT, f"Строка 1: {first}\nСтрока 2: {second}")
    return [numeric_sample(state, pick_question(LCS_QUESTIONS, row["inputs"]), value, row["inputs"], lower=0)]


def mathlogicqa_samples(row: dict) -> list[dict]:
    # The 'logic' subset includes unstated facts such as animal leg counts and units. Keep explicit algebra only.
    if row["meta"]["task"] != "math" or row["outputs"] not in "ABCD" or not row["outputs"]:
        return []
    inputs = row["inputs"]
    texts = [inputs[f"option_{id.lower()}"] for id in "ABCD"]
    if len(set(texts)) != len(texts):
        return []
    options = stable_shuffle(make_options(list("ABCD"), texts), inputs["text"])
    state = with_context(
        "Решите алгебраическую задачу, используя только приведённые равенства и условия.", inputs["text"]
    )
    return [make_sample(state, pick_question(MATH_QUESTIONS, inputs["text"]), options, row["outputs"])]


class MERADataset(JevDataset):
    type = DatasetType.CHOICE

    def __init__(self, name: str, task: str, convert: Callable[[dict], list[dict]]):
        super().__init__(name=name, hf_path="ai-forever/MERA", hf_name=task)
        self.convert = convert

    def load(self) -> DatasetDict:
        # MERA's public tests have blank targets; loading them as training would leak benchmark inputs.
        url = f"https://huggingface.co/datasets/{self.hf_path}/resolve/main/data/{self.hf_name}/train.jsonl"
        return load_dataset("json", data_files={"train": url})

    def prepare(self):
        self.data = self.data.map(
            explode(self.convert), batched=True, remove_columns=self.source_columns, features=SAMPLE_FEATURES
        )


mera_bps_dataset = MERADataset(name="mera_bps", task="bps", convert=bps_samples)
mera_lcs_dataset = MERADataset(name="mera_lcs", task="lcs", convert=lcs_samples)
mera_mathlogicqa_dataset = MERADataset(name="mera_mathlogicqa", task="mathlogicqa", convert=mathlogicqa_samples)
mera_rumodar_dataset = MERADataset(name="mera_rumodar", task="rumodar", convert=rumodar_samples)
mera_rumultiar_dataset = MERADataset(name="mera_rumultiar", task="rumultiar", convert=arithmetic_samples)
mera_simplear_dataset = MERADataset(name="mera_simplear", task="simplear", convert=arithmetic_samples)

# RCB is taken from Russian SuperGLUE only. Knowledge, commonsense, preferences, and expert code tasks are excluded.
# ruTiE mixes outside facts and irony even in math turns, so its category flags do not guarantee state answerability.
