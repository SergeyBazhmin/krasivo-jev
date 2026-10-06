import json
import re
import tarfile
from collections import defaultdict
from collections.abc import Iterator
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

from datasets import DatasetDict, DownloadManager

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.structured.source import content_group, grouped_records, map_records, record
from jev_datasets.utils import pick_question, stable_shuffle, text_choice

REVISION = "cffb423077756d04c1bac5bcd45167c86903fbcb"
URL = f"https://raw.githubusercontent.com/salesforce/WikiSQL/{REVISION}/data.tar.bz2"
QUESTIONS = [
    "Use the table to answer: {request}",
    "Considering only the given table, {request}",
    "Calculate or read the following result from the table: {request}",
    "Select the result of this operation on the table: {request}",
    "What result does the table give for this request? {request}",
]
QUESTIONS_RU = [
    "Используя таблицу, ответьте: {request}",
    "Рассматривая только данную таблицу: {request}",
    "Вычислите или прочитайте результат по таблице: {request}",
    "Выберите результат этой операции над таблицей: {request}",
    "Какой результат даёт таблица для запроса? {request}",
]
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def number(value: object) -> Decimal | None:
    # No implicit unit stripping, comma/thousands conventions, dates or missing-value coercion.
    text = str(value).strip()
    if len(text) > 30 or not NUMBER.fullmatch(text):
        return None
    try:
        result = Decimal(text)
    except InvalidOperation:
        return None
    return (
        result if result.is_finite() and result.as_tuple().exponent >= -6 and abs(result) <= Decimal("1e12") else None
    )


def formatted(value: Decimal) -> str:
    return format(value, "f").rstrip("0").rstrip(".") if value % 1 else str(int(value))


def normalized_table(table: dict) -> dict | None:
    headers, rows, types = table["header"], table["rows"], table["types"]
    if not 2 <= len(headers) <= 12 or not 2 <= len(rows) <= 40 or len(types) != len(headers):
        return None
    if any(not isinstance(header, str) or not header.strip() or len(header) > 100 for header in headers):
        return None
    if len({header.strip().casefold() for header in headers}) != len(headers):
        return None
    if any(len(row) != len(headers) for row in rows):
        return None
    if any(isinstance(value, (bool, dict, list)) or value is None for row in rows for value in row):
        return None
    rows = [[str(value).strip() for value in row] for row in rows]
    if any(not value or len(value) > 300 for row in rows for value in row):
        return None
    cleaned = {"header": headers, "rows": rows, "types": types}
    return cleaned if len(json.dumps(cleaned, ensure_ascii=False)) <= 14000 else None


def table_group(table: dict) -> str:
    # Row order and source table ids cannot hide duplicate tables in different partitions.
    return content_group(
        {
            "header": [value.strip().casefold() for value in table["header"]],
            "rows": sorted([[str(value).strip().casefold() for value in row] for row in table["rows"]]),
        }
    )


def wikisql_records(path: str, excluded_groups: tuple[str, ...] = ()) -> Iterator[dict]:
    excluded = set(excluded_groups)
    with tarfile.open(path) as archive:
        for split, source_split in (("train", "train"), ("validation", "dev"), ("test", "test")):
            tables = {
                row["id"]: row
                for line in archive.extractfile(f"data/{source_split}.tables.jsonl")
                if (row := json.loads(line))
            }
            queries = defaultdict(list)
            for line in archive.extractfile(f"data/{source_split}.jsonl"):
                row = json.loads(line)
                queries[row["table_id"]].append(row["sql"])
            for table_id, sql in queries.items():
                raw_table = tables[table_id]
                group = table_group(raw_table)
                if group in excluded:
                    continue  # Shared Wikipedia tables belong to ToTTo only, across every split.
                # Reserve even rejected holdout tables, in case a train copy has different metadata.
                table = normalized_table(raw_table)
                yield {**record({"table": table, "queries": sql}, group), "split": split}


def execute(table: dict, query: dict) -> tuple[str, list[str]] | None:
    headers, types, rows = table["header"], table["types"], table["rows"]
    selected, aggregation, conditions = query["sel"], query["agg"], query["conds"]
    if not 0 <= selected < len(headers) or aggregation not in range(6) or len(conditions) > 4:
        return None
    matching = rows
    for column, operator, value in conditions:
        if not 0 <= column < len(headers) or operator not in {0, 1, 2}:
            return None
        if types[column] == "real":
            threshold = number(value)
            values = [number(row[column]) for row in rows]
            if threshold is None or any(value is None for value in values):
                return None

            def satisfies(row: list[str]) -> bool:
                candidate = number(row[column])
                return (candidate == threshold, candidate > threshold, candidate < threshold)[operator]

        elif operator == 0:

            def satisfies(row: list[str]) -> bool:
                return row[column].strip().casefold() == str(value).strip().casefold()

        else:
            return None  # Lexical ordering of arbitrary text is not a useful numeric-comparison task.
        matching = [row for row in matching if satisfies(row)]
    if aggregation == 3:
        answer = Decimal(len(matching))
        alternatives = [Decimal(len(rows)), answer - 1, answer + 1, answer + 2, Decimal(0)]
    elif aggregation == 0:
        values = {row[selected].casefold(): row[selected] for row in matching}
        if len(values) != 1:
            return None  # A list-valued SQL answer must not become an arbitrarily chosen one-hot.
        answer_text = next(iter(values.values()))
        if types[selected] == "real":
            answer = number(answer_text)
            if answer is None:
                return None
            alternatives = [value for row in rows if (value := number(row[selected])) is not None]
        else:
            candidates = [row[selected] for row in rows if row[selected].casefold() not in values]
            return answer_text, candidates
    else:
        if types[selected] != "real" or not matching:
            return None
        values = [number(row[selected]) for row in matching]
        if any(value is None for value in values):
            return None
        statistics = {1: max(values), 2: min(values), 4: sum(values), 5: sum(values) / len(values)}
        answer = statistics[aggregation]
        if aggregation == 5:
            answer = answer.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        alternatives = [*values, *statistics.values(), answer - 1, answer + 1]
    if abs(answer) > Decimal("1e12"):
        return None
    answer_text = formatted(answer)
    candidates = {
        formatted(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        for value in alternatives
        if value >= 0 or answer < 0
    }
    candidates.discard(answer_text)
    return answer_text, sorted(candidates)


def request_text(table: dict, query: dict, russian: bool) -> str:
    headers = table["header"]
    selected, aggregation = query["sel"], query["agg"]
    operations = (
        [
            "прочитайте значение столбца «{column}»",
            "найдите максимальное значение столбца «{column}»",
            "найдите минимальное значение столбца «{column}»",
            "посчитайте количество строк (повторы тоже считаются)",
            "найдите сумму значений столбца «{column}»",
            "найдите среднее арифметическое столбца «{column}», округлив до сотых",
        ]
        if russian
        else [
            'read the value of column "{column}"',
            'find the maximum value in column "{column}"',
            'find the minimum value in column "{column}"',
            "count the rows (including repeated rows)",
            'sum the values in column "{column}"',
            'find the arithmetic mean of column "{column}", rounded to two decimal places',
        ]
    )
    operation = operations[aggregation].format(column=headers[selected])
    filters = []
    for column, operator, value in query["conds"]:
        comparison = ["равно", "больше", "меньше"] if russian else ["equals", "is greater than", "is less than"]
        filters.append(f"«{headers[column]}» {comparison[operator]} «{value}»")
    if filters:
        scope = "для строк, где одновременно " if russian else "for rows where all these conditions hold: "
        scope += (" И " if russian else " AND ").join(filters)
    else:
        scope = "для всех строк" if russian else "for all rows"
    return f"{operation} {scope}."


def wikisql_samples(row: dict, russian: bool = False) -> list[dict]:
    payload = json.loads(row["payload"])
    table = payload["table"]
    if table is None:
        return []
    guide = (
        "Таблица. Строки даны как списки значений в порядке столбцов. При сравнении текста не учитывайте регистр "
        "и пробелы по краям. Числа записаны с десятичной точкой. При округлении ровно половину округляйте от нуля."
        if russian
        else "Table. Each row lists values in column order. Text equality ignores case and surrounding spaces. "
        "Numbers use a decimal point. Round exact halves away from zero."
    )
    state = guide + "\n\n" + ("Столбцы: " if russian else "Columns: ")
    state += json.dumps(table["header"], ensure_ascii=False)
    for index, values in enumerate(table["rows"], 1):
        state += f"\n{index}: " + json.dumps(values, ensure_ascii=False)
    samples, seen = [], set()
    for query in stable_shuffle(payload["queries"], row["group"]):
        identity = json.dumps(query, sort_keys=True)
        if identity in seen:
            continue
        seen.add(identity)
        if result := execute(table, query):
            answer, candidates = result
            seed = row["group"] + identity
            request = request_text(table, query, russian)
            question = pick_question(QUESTIONS_RU if russian else QUESTIONS, seed).format(request=request)
            if sample := text_choice(state, question, answer, candidates, seed):
                samples.append(sample)
        if len(samples) == 3:
            break
    return samples


class WikiSQLQueriesDataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        manifest = json.loads(Path(__file__).with_name("wikisql_totto_overlap.json").read_text())
        if manifest["wikisql_revision"] != REVISION:
            raise ValueError("Regenerate the ToTTo overlap manifest for this WikiSQL revision")
        return grouped_records(
            wikisql_records,
            path=DownloadManager().download(URL),
            excluded_groups=tuple(manifest["excluded_groups"]),
        )

    def prepare(self):
        russian = self.hf_name == "ru"
        self.data = map_records(self.data, lambda row: wikisql_samples(row, russian), __file__)


wikisql_queries_dataset = WikiSQLQueriesDataset(name="wikisql_queries", hf_path="salesforce/WikiSQL", hf_name="en")
wikisql_queries_ru_dataset = WikiSQLQueriesDataset(
    name="wikisql_queries_ru", hf_path="salesforce/WikiSQL", hf_name="ru"
)
