from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifierCV
from export.torch.pipeline import Pipeline as TPipeline
from sklearn.preprocessing import StandardScaler
from export.torch.standard_scaler import StandardScaler as TStandardScaler
from export.torch.linear import LinearClassifier as TLinearClassifier
from .registry import register_op, get_op
from operator import itemgetter
import torch

@register_op(name = Pipeline.__name__)
def pipeline_factory(pipeline: Pipeline):
    steps = list(map(get_op, map(itemgetter(1), pipeline.steps)))
    return TPipeline(steps = steps)

@register_op(name = StandardScaler.__name__)
def standard_scaler_factory(scaler: StandardScaler):
    return TStandardScaler(
        with_mean = scaler.with_mean,
        with_std = scaler.with_std,
        mean = torch.from_numpy(scaler.mean_),
        scale = torch.from_numpy(scaler.scale_)
    )

@register_op(name = RidgeClassifierCV.__name__)
def ridge_classifier_factory(clf: RidgeClassifierCV):
    return TLinearClassifier(
        weights = torch.from_numpy(clf.coef_),
        intercept = torch.from_numpy(clf.intercept_)
    )
