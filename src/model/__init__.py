from src.utils.registry import Registry

MODEL_REGISTRY = Registry("MODEL")

from src.model.baseline import BaselineModel
from src.model.bce_baseline import BCEPointCloudTextModel
MODEL_REGISTRY.register(BaselineModel)
MODEL_REGISTRY.register(BCEPointCloudTextModel)
