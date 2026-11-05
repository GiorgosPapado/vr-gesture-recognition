import pandas as pd
import numpy as np
from sktime.transformations.base import BaseTransformer
from sktime.transformations.compose import CORE_MTYPES
from sktime.datatypes._panel._convert import from_nested_to_2d_array, from_2d_array_to_nested
from sklearn.decomposition import PCA
from sktime.utils.validation.panel import check_X
from .quaternion import quaternion_vec_mul

class TSControllerCoordinateTransform(BaseTransformer):
    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:transform-labels": "None",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "capability:inverse_transform": False,  # can the transformer inverse transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "X_inner_mtype": "pd.DataFrame",  # which mtypes do _fit/_predict support for X?
        # this can be a Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "requires_y": False,  # does y need to be passed in fit?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "transform-returns-same-time-index": False,
        # does transform return have the same time index as input X
        "skip-inverse-transform": False,  # is inverse-transform skipped when called?
        "capability:unequal_length": True,
        # can the transformer handle unequal length time series (if passed Panel)?
        "capability:unequal_length:removes": True,
        # is transform result always guaranteed to be equal length (and series)?
        "handles-missing-data": False,  # can estimator handle missing data?
        # todo: rename to capability:missing_values
        "capability:missing_values:removes": False,
        # is transform result always guaranteed to contain no missing values?
        "python_version": None,  # PEP 440 python version specifier to limit versions
        "remember_data": False,  # whether all data seen is remembered as self._X
    }
    
    def __init__(self, position_indices = (0,1,2), rotation_indices = (3,4,5,6), reference_axis = (0,1,0), additional_axis = (7,)):
        assert len(position_indices) == 3
        assert len(rotation_indices) == 4
        self.position_indices = position_indices
        self.rotation_indices = rotation_indices
        self.reference_axis = reference_axis
        self.additional_axis = additional_axis
        super().__init__()

    def _transform_instance(self, instance: pd.DataFrame) -> np.ndarray:
        X = from_nested_to_2d_array(instance, return_numpy=True)

        Xpos = X[self.position_indices,:].T
        Qrot = X[self.rotation_indices,0]
        if self.additional_axis is not None:
            Add = X[self.additional_axis,:]

        refPos = Xpos[0,:]

        pca = PCA(n_components=3)
        pca.fit(Xpos)
        basis = pca.components_
        
        pca_z = basis[-1,:]
        world_y = np.array(self.reference_axis, dtype=float)

        ZFwd = quaternion_vec_mul(Qrot, np.array([0,0,1], dtype=float)).squeeze()     # Z forward
        
        # Flip the sign of pca_z to point towards the forward direction of the controller
        if np.dot(ZFwd,pca_z) < 0:
            pca_z *= -1.0
        
        # new_up is the world_y
        new_up = world_y
        # the new_right vector is parallel to the gesture plane
        new_right = np.cross(world_y, pca_z)
        # the new forward looks towards the fwd direction of the controller while remaining perpendicular to world_y and the new_right vector
        new_fwd = np.cross(new_right,new_up)
        # construct the new basis
        new_basis = np.stack((new_right, new_up, new_fwd))
        # the origin of the new basis is located at the first point of the gesture's trajectory
        newX = new_basis @ (Xpos - refPos).T
        
        if self.additional_axis is not None:
            return [pd.Series(newX[index,:]) for index in range(newX.shape[0])] + [pd.Series(Add[index,:]) for index in range(Add.shape[0])]
        else:
            return [pd.Series(newX[index,:]) for index in range(newX.shape[0])]

    def fit(self, X, y = None):
        return super().fit(X,y)

    def transform(self, X: pd.DataFrame, y = None):
        self.check_is_fitted()
        X = check_X(X,coerce_to_pandas=True)
        newX = X.apply(self._transform_instance,axis=1,result_type='expand')
        return newX   


class TSControllerTranslationCoordinateTransform(BaseTransformer):
    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:transform-labels": "None",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "capability:inverse_transform": False,  # can the transformer inverse transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "X_inner_mtype": "pd.DataFrame",  # which mtypes do _fit/_predict support for X?
        # this can be a Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "requires_y": False,  # does y need to be passed in fit?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "transform-returns-same-time-index": False,
        # does transform return have the same time index as input X
        "skip-inverse-transform": False,  # is inverse-transform skipped when called?
        "capability:unequal_length": True,
        # can the transformer handle unequal length time series (if passed Panel)?
        "capability:unequal_length:removes": True,
        # is transform result always guaranteed to be equal length (and series)?
        "handles-missing-data": False,  # can estimator handle missing data?
        # todo: rename to capability:missing_values
        "capability:missing_values:removes": False,
        # is transform result always guaranteed to contain no missing values?
        "python_version": None,  # PEP 440 python version specifier to limit versions
        "remember_data": False,  # whether all data seen is remembered as self._X
    }
    
    def __init__(self, position_indices = (0,1,2), additional_axis = None):
        assert len(position_indices) == 3        
        self.position_indices = position_indices
        self.additional_axis = additional_axis
        super().__init__()

    def _transform_instance(self, instance: pd.DataFrame) -> np.ndarray:
        X = from_nested_to_2d_array(instance, return_numpy=True)

        Xpos = X[self.position_indices,:].T
        refPos = Xpos[0,:]

        if self.additional_axis is not None:
            Add = X[self.additional_axis,:]     # Features X Time

        newX = (Xpos - refPos).T        # 3 x Time
        
        if self.additional_axis is not None:
            return [pd.Series(newX[index,:]) for index in range(newX.shape[0])] + [pd.Series(Add[index,:]) for index in range(Add.shape[0])]
        else:
            return [pd.Series(newX[index,:]) for index in range(newX.shape[0])]

    def fit(self, X, y = None):
        return super().fit(X,y)

    def transform(self, X: pd.DataFrame, y = None):
        self.check_is_fitted()
        X = check_X(X,coerce_to_pandas=True)
        newX = X.apply(self._transform_instance,axis=1,result_type='expand')
        return newX          

class TSHandCoordinateTransform(BaseTransformer):
    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:transform-labels": "None",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "capability:inverse_transform": False,  # can the transformer inverse transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "X_inner_mtype": "pd.DataFrame",  # which mtypes do _fit/_predict support for X?
        # this can be a Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "requires_y": False,  # does y need to be passed in fit?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "transform-returns-same-time-index": False,
        # does transform return have the same time index as input X
        "skip-inverse-transform": False,  # is inverse-transform skipped when called?
        "capability:unequal_length": True,
        # can the transformer handle unequal length time series (if passed Panel)?
        "capability:unequal_length:removes": True,
        # is transform result always guaranteed to be equal length (and series)?
        "handles-missing-data": False,  # can estimator handle missing data?
        # todo: rename to capability:missing_values
        "capability:missing_values:removes": False,
        # is transform result always guaranteed to contain no missing values?
        "python_version": None,  # PEP 440 python version specifier to limit versions
        "remember_data": False,  # whether all data seen is remembered as self._X
    }
    
    def __init__(self, global_position_indices = (0,1,2)):
        assert len(global_position_indices) == 3        
        self.global_position_indices = global_position_indices
        super().__init__()

    def _transform_instance(self, instance: pd.DataFrame) -> np.ndarray:
        X = from_nested_to_2d_array(instance, return_numpy=True)
        
        newX = X.T
        newX[:,self.global_position_indices] -= newX[0,self.global_position_indices]
        newX = newX.T
        return [pd.Series(newX[index,:]) for index in range(newX.shape[0])]

    def fit(self, X, y = None):
        return super().fit(X,y)

    def transform(self, X: pd.DataFrame, y = None):
        self.check_is_fitted()
        X = check_X(X,coerce_to_pandas=True)
        newX = X.apply(self._transform_instance,axis=1,result_type='expand')
        return newX        

