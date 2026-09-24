from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

QUESTION = "К какому врачу следует обратиться пациенту?"

# `to_doctor` is free text (~95 spellings); these specialties cover ~95% of dialogues once
# case is folded. Compound ("уролог или андролог") and rare referrals are dropped.
DOCTORS = {
    "терапевт": "therapist",
    "эндокринолог": "endocrinologist",
    "кардиолог": "cardiologist",
    "гинеколог": "gynecologist",
    "дерматолог": "dermatologist",
    "невролог": "neurologist",
    "травматолог": "traumatologist",
    "проктолог": "proctologist",
    "педиатр": "pediatrician",
    "психиатр": "psychiatrist",
    "уролог": "urologist",
    "аллерголог": "allergist",
    "стоматолог": "dentist",
    "офтальмолог": "ophthalmologist",
    "хирург": "surgeon",
    "онколог": "oncologist",
    "гастроэнтеролог": "gastroenterologist",
    "андролог": "andrologist",
    "диетолог": "dietitian",
    "косметолог": "cosmetologist",
    "репродуктолог": "reproductologist",
    "нефролог": "nephrologist",
}
OPTIONS = make_options(list(DOCTORS.values()), list(DOCTORS))


def normalize(doctor: str) -> str:
    return doctor.strip().lower()


class RusMedDialoguesDataset(JevDataset):
    def prepare(self):
        self.data = self.data.filter(lambda x: normalize(x["to_doctor"]) in DOCTORS)
        self.data = self.data.map(
            lambda x: make_sample(
                x["user_question"], QUESTION, OPTIONS, DOCTORS[normalize(x["to_doctor"])]
            ),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


rus_med_dialogues_dataset = RusMedDialoguesDataset(
    name="rus_med_dialogues", hf_path="Mykes/rus_med_dialogues"
)
