from jev_model.base import JevModel
from jev_model.pointer.pipeline import pointer_model
from jev_model.typesafe.pipeline import typesafe_model

# {name: model}, alphabetical. `zero_shot/` is kept as code but not registered: import `zero_shot_model` here to use it
models: dict[str, JevModel] = {
    model.name: model
    for model in [
        pointer_model,
        typesafe_model,
    ]
}
