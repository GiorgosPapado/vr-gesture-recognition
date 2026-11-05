from utils.preproc import TSControllerTranslationCoordinateTransform, TSControllerCoordinateTransform
from export.torch.preproc.controller_transform import TSControllerTranslationCoordinateTransform as TTSControllerTranslationCoordinateTransform
from export.torch.preproc.controller_transform import TSControllerCoordinateTransform as TTSControllerCoordinateTransform
from .registry import register_op, get_op
import torch

@register_op(name = TSControllerTranslationCoordinateTransform.__name__)
def controller_translation_transform_factory(transform: TSControllerTranslationCoordinateTransform):    
    return TTSControllerTranslationCoordinateTransform()

@register_op(name = TSControllerCoordinateTransform.__name__)
def controller_transform_factory(transform: TSControllerCoordinateTransform):
    return TTSControllerCoordinateTransform(
        torch.tensor(transform.position_indices, dtype=torch.long),
        torch.tensor(transform.rotation_indices, dtype=torch.long),
        torch.tensor(transform.reference_axis, dtype=torch.float),
        torch.tensor(transform.additional_axis, dtype = torch.long) if transform.additional_axis is not None else None
    )