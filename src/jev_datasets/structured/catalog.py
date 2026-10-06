import json
import random
from collections.abc import Iterator

from datasets import Dataset, DatasetDict

from jev_datasets.base import DatasetType, JevDataset
from jev_datasets.structured.source import record
from jev_datasets.utils import (
    SAMPLE_FEATURES,
    explode,
    make_options,
    make_sample,
    pick_question,
    stable_shuffle,
    text_choice,
)

QUESTIONS = {
    "filter": [
        "Which item meets both conditions?",
        "Choose the item satisfying both requirements.",
        "Which item matches the two constraints?",
        "Select the item that passes both conditions.",
        "Which listed item qualifies under both requirements?",
    ],
    "cheapest": [
        "Which eligible item is cheapest?",
        "Choose the least expensive eligible item.",
        "Which item has the lowest price among those with sufficient stock?",
        "Select the cheapest item that is available in the requested quantity.",
        "Which available item minimizes the price?",
    ],
    "update": [
        "How many units remain after these changes?",
        "What is the resulting stock?",
        "Choose the final number of units in stock.",
        "How many units are available after both operations?",
        "What is the stock after the sale and return?",
    ],
}
QUESTIONS_RU = {
    "filter": [
        "Какой товар удовлетворяет обоим условиям?",
        "Выберите товар, подходящий под оба требования.",
        "Какой товар соответствует двум ограничениям?",
        "Укажите товар, выполняющий оба условия.",
        "Какой из перечисленных товаров проходит оба требования?",
    ],
    "cheapest": [
        "Какой подходящий товар дешевле всего?",
        "Выберите самый дешёвый подходящий товар.",
        "У какого товара минимальная цена среди товаров с достаточным остатком?",
        "Укажите самый дешёвый товар, доступный в нужном количестве.",
        "Какой доступный товар имеет наименьшую цену?",
    ],
    "update": [
        "Сколько единиц останется после этих изменений?",
        "Каким станет остаток товара?",
        "Выберите итоговое количество товара на складе.",
        "Сколько единиц доступно после обеих операций?",
        "Каков остаток после продажи и возврата?",
    ],
}


def catalog_records(split: str, count: int) -> Iterator[dict]:
    for index in range(count):
        identity = f"catalog:{split}:{index}"
        rng = random.Random(identity)
        size = rng.randrange(4, 9)
        prices = rng.sample(range(20, 500), size)
        items = [
            {
                "id": f"P{i + 1}",
                "price": prices[i],
                "warranty": rng.choice([6, 12, 18, 24, 36]),
                "stock": rng.randrange(0, 41),
            }
            for i in range(size)
        ]
        target = rng.choice(items)
        payload = {
            "items": items,
            "budget": rng.randrange(40, 500),
            "warranty": rng.choice([12, 18, 24, 36]),
            "quantity": rng.randrange(1, 41),
            "target": target["id"],
            "sold": rng.randrange(target["stock"] + 1),
            "returned": rng.randrange(1, 11),
        }
        yield record(payload, identity)


def catalog_samples(row: dict, russian: bool) -> list[dict]:
    payload = json.loads(row["payload"])
    items = payload["items"]
    none = "ни один" if russian else "none"
    guide = (
        "У каждого товара указаны цена за единицу, гарантия в месяцах и количество единиц на складе. "
        "Используйте только эти сведения. Продажа уменьшает остаток, возврат увеличивает его."
        if russian
        else "Each item has a unit price, a warranty in months and a stock quantity. "
        "Use only these values. A sale reduces stock; a return increases it."
    )
    lines = [
        f"{item['id']}: цена {item['price']}; гарантия {item['warranty']}; остаток {item['stock']}"
        if russian
        else f"{item['id']}: price {item['price']}; warranty {item['warranty']}; stock {item['stock']}"
        for item in items
    ]
    state = guide + "\n\n" + "\n".join(lines)
    variants = QUESTIONS_RU if russian else QUESTIONS
    samples = []

    def choose(task: str, question: str, answer: str):
        texts = stable_shuffle([*[item["id"] for item in items], none], row["group"] + task)
        # Keep every item as an option, so the model must inspect all records; exactly one answer is accepted.
        ids = [str(i) for i in range(len(texts))]
        samples.append(make_sample(state, question, make_options(ids, texts), ids[texts.index(answer)]))

    matching = [item for item in items if item["price"] < payload["budget"] and item["warranty"] >= payload["warranty"]]
    if len(matching) <= 1:
        conditions = (
            f"Цена строго меньше {payload['budget']}; гарантия не меньше {payload['warranty']} месяцев."
            if russian
            else f"Price strictly below {payload['budget']}; warranty at least {payload['warranty']} months."
        )
        question = conditions + "\n" + pick_question(variants["filter"], row["group"])
        choose("filter", question, matching[0]["id"] if matching else none)

    eligible = [item for item in items if item["stock"] >= payload["quantity"]]
    answer = min(eligible, key=lambda item: item["price"])["id"] if eligible else none
    condition = (
        f"Нужно {payload['quantity']} единиц." if russian else f"Required quantity: {payload['quantity']} units."
    )
    choose("cheapest", condition + "\n" + pick_question(variants["cheapest"], row["group"]), answer)

    target = next(item for item in items if item["id"] == payload["target"])
    result = target["stock"] - payload["sold"] + payload["returned"]
    operations = (
        f"Для {target['id']}: продано {payload['sold']} единиц, затем возвращено {payload['returned']} единиц."
        if russian
        else f"For {target['id']}: {payload['sold']} units sold, then {payload['returned']} units returned."
    )
    question = operations + "\n" + pick_question(variants["update"], row["group"])
    wrong = sorted(
        {
            max(0, result - 1),
            result + 1,
            target["stock"],
            target["stock"] + payload["returned"],
            target["stock"] - payload["sold"],
            target["stock"] + payload["sold"] - payload["returned"],
        }
    )
    if sample := text_choice(state, question, str(result), [str(n) for n in wrong if n >= 0], row["group"] + "update"):
        samples.append(sample)
    return samples


class CatalogDataset(JevDataset):
    type = DatasetType.CHOICE

    def load(self) -> DatasetDict:
        # Language variants use identical source groups and splits, so their translations cannot cross partitions.
        return DatasetDict(
            {
                split: Dataset.from_generator(catalog_records, gen_kwargs={"split": split, "count": count})
                for split, count in (("train", 6000), ("validation", 1000), ("test", 1000))
            }
        )

    def prepare(self):
        self.data = self.data.map(
            explode(lambda row: catalog_samples(row, self.hf_name == "ru")),
            batched=True,
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


catalog_dataset = CatalogDataset(name="catalog", hf_path="synthetic/catalog", hf_name="en")
catalog_ru_dataset = CatalogDataset(name="catalog_ru", hf_path="synthetic/catalog", hf_name="ru")
