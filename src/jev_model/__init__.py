from jev_model.base import JevModel
from jev_model.frozen_head.pipeline import frozen_head_model

# {name: model}, alphabetical
models: dict[str, JevModel] = {
    model.name: model
    for model in [
        frozen_head_model,
    ]
}
