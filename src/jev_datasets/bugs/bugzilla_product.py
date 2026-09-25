from jev_datasets.base import JevDataset
from jev_datasets.utils import SAMPLE_FEATURES, make_options, make_sample

QUESTION = "Which product team should this bug report be routed to?"

# Mozilla (Core, Firefox, Thunderbird, Bugzilla) and Eclipse (CDT, JDT, Platform) products
PRODUCTS = {
    "Core": "Mozilla Core (Gecko engine)",
    "Firefox": "Firefox",
    "Thunderbird": "Thunderbird",
    "Bugzilla": "Bugzilla",
    "CDT": "Eclipse CDT (C/C++ tools)",
    "JDT": "Eclipse JDT (Java tools)",
    "Platform": "Eclipse Platform",
}
OPTIONS = make_options(list(PRODUCTS), list(PRODUCTS.values()))


class BugzillaProductDataset(JevDataset):
    def prepare(self):
        # only the one-line summary is available, not the full report
        self.data = self.data.map(
            lambda x: make_sample(x["Short Description"], QUESTION, OPTIONS, x["Project"]),
            remove_columns=self.source_columns,
            features=SAMPLE_FEATURES,
        )


bugzilla_product_dataset = BugzillaProductDataset(
    name="bugzilla_product", hf_path="AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset"
)
