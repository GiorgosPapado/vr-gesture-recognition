from sktime.transformations.panel.interpolate import TSInterpolator
from sktime.classification.kernel_based import RocketClassifier
from sktime.transformations.panel.rocket import Rocket
from export.torch.tsinterpolator import TSInterpolator as TTSInterpolator
from export.torch.rocket import Rocket as TRocket
from export.torch.linear import LinearClassifier as TLinearClassifier
from export.torch.pipeline import Pipeline as TPipeline
from .sklearn import TStandardScaler
from .registry import register_op, get_op
import torch

@register_op(name = TSInterpolator.__name__)
def interpolator_factory(interpolator: TSInterpolator):
    return TTSInterpolator(length = interpolator.length)

@register_op(name = Rocket.__name__)
def rocket_transform_factory(rocket: Rocket):
    weights, \
    lengths, \
    biases, \
    dilations, \
    paddings, \
    num_channel_indices, \
    channel_indices = rocket.kernels

    trocket = TRocket(
        normalize = rocket.normalise,
        weights = torch.from_numpy(weights),
        lengths = torch.from_numpy(lengths),
        biases = torch.from_numpy(biases),
        dilations = torch.from_numpy(dilations),
        paddings = torch.from_numpy(paddings),
        num_channel_indices = torch.from_numpy(num_channel_indices),
        channel_indices = torch.from_numpy(channel_indices)
    )

    return trocket

@register_op(name = RocketClassifier.__name__)
def rocket_classifier_factory(rocket_clf: RocketClassifier):
    rocket_trans : TRocket = get_op(rocket_clf.multivar_rocket_.transformers_.steps_[0][1])    
    scaler_trans : TStandardScaler = get_op(rocket_clf.multivar_rocket_.transformers_.steps_[1][1].transformer_)
    linear_clf : TLinearClassifier = get_op(rocket_clf.multivar_rocket_.classifier_)

    pipeline = TPipeline(
        [
            rocket_trans,
            scaler_trans,
            linear_clf
        ]
    )
    pipeline.eval()
    return pipeline
    
    
