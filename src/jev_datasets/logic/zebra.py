import re

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, explode, make_options, make_sample, pick_question

# the attribute listing in the introduction: "Pets: cat, dog, rabbit and snail."
CATEGORY = re.compile(r"^([^:\n]+): .+\.$", re.M)
SPLITS = {"train": "train", "val": "validation", "test": "test"}

# per language: its configs (2x3 and 4x5 grids, five red-herring clues each), and the
# questions asked of a solved grid, "{category}" being the name as the puzzle spells it
LANGUAGES = {
    "en": {
        "configs": ["dataset_en_houses_2x3_5rh", "dataset_en_houses_4x5_5rh"],
        "house": "house {house}",
        "value_questions": [
            "Which of the {category} belongs to the person in house {house}?",
            "What does the person in house {house} have among the {category}?",
            "For the person in house {house}, which option from the {category} applies?",
        ],
        "house_questions": [
            "Which house does the person with \"{value}\" ({category}) live in?",
            "In which house is \"{value}\" from the {category}?",
            "Where does the person with \"{value}\" ({category}) live?",
        ],
    },
    "ru": {
        "configs": ["dataset_ru_doma_2x3_5rh", "dataset_ru_doma_4x5_5rh"],
        "house": "дом {house}",
        "value_questions": [
            "Что из категории «{category}» относится к жителю дома {house}?",
            "Какая характеристика из категории «{category}» у человека в доме {house}?",
            "Выберите для жителя дома {house} значение из категории «{category}».",
        ],
        "house_questions": [
            "В каком доме живёт человек с характеристикой «{value}» ({category})?",
            "В каком доме находится «{value}» из категории «{category}»?",
            "Где живёт человек, у которого «{value}» ({category})?",
        ],
    },
}


def grid_samples(x: dict, language: str) -> list[dict]:
    """
    A solved grid as many single-cell questions, both ways round: which value of a category
    a house has, and which house has a value. The source asks for the whole grid at once.
    """
    templates = LANGUAGES[language]
    state = x["introduction"] + "\n".join(x["clues"])
    categories = CATEGORY.findall(x["introduction"])
    houses = x["houses"]
    if any(len(values) != len(categories) for values in houses):
        return []
    house_names = [templates["house"].format(house=n) for n in range(1, len(houses) + 1)]
    house_options = make_options(house_names)
    samples = []
    for c, category in enumerate(categories):
        name = category.lower() if language == "en" else category
        options = make_options(sorted(values[c] for values in houses))
        for n, values in enumerate(houses, start=1):
            seed = f"{state}{c}{n}"
            question = pick_question(templates["value_questions"], seed).format(category=name, house=n)
            samples.append(make_sample(state, question, options, values[c]))
            question = pick_question(templates["house_questions"], seed).format(category=name, value=values[c])
            samples.append(make_sample(state, question, house_options, house_names[n - 1]))
    return samples


def with_houses(rows: Dataset) -> Dataset:
    """`solution` has one field per house ("object_1", ...), so its type differs between grid
    sizes; as a list of houses the configs can be joined."""
    return rows.map(
        lambda x: {"houses": [x["solution"][f"object_{n}"] for n in range(1, len(x["solution"]) + 1)]},
        remove_columns=["solution"],
    )


class MultiZebraLogicDataset(JevDataset):
    def __init__(self, language: str):
        super().__init__(name=f"multi_zebra_logic_{language}", hf_path="alexandrainst/multi-zebra-logic")
        self.language = language

    def load(self) -> DatasetDict:
        configs = [load_dataset(self.hf_path, config) for config in LANGUAGES[self.language]["configs"]]
        return DatasetDict(
            {
                SPLITS[split]: concatenate_datasets([with_houses(config[split]) for config in configs])
                for split in SPLITS
            }
        )

    def prepare(self):
        self.data = self.data.map(
            explode(lambda x: grid_samples(x, self.language)),
            batched=True,
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


multi_zebra_logic_en_dataset = MultiZebraLogicDataset("en")
multi_zebra_logic_ru_dataset = MultiZebraLogicDataset("ru")
