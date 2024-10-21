from .base_model import BaseModel


class Glove(BaseModel):
    def __init__(self, model_name: str = "glove", *args, **kwargs) -> None:
        super().__init__(model_name, *args, **kwargs)
