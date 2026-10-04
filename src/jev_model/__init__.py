from jev_model.base import JevModel
from jev_model.frozen_head.pipeline import frozen_head_model
from jev_model.pointer.pipeline import pointer_model
from jev_model.typesafe.pipeline import typesafe_model
from jev_model.zero_shot.pipeline import zero_shot_model

# {name: model}, alphabetical
models: dict[str, JevModel] = {
    model.name: model
    for model in [
        frozen_head_model,
        pointer_model,
        typesafe_model,
        zero_shot_model,
    ]
}
