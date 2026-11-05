import torch
import torch.nn as nn
import torch.linalg as tl
from typing import Tuple, Optional

class TSControllerTranslationCoordinateTransform(nn.Module):
    # TODO: implement hyper-parameters of the original transform (like position_indices and additional_feature_indices)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: size BxFxT (Batch X Features X Time)
        :returns: y size BxFxT removing offset of time series with respect to the first sample in the time-sequence
        """

        y = x.permute(2,0,1)
        y = y - y[0,...]
        y = y.permute(1,2,0)
        return y
    


def quaternion_vec_mul(q : torch.Tensor, v : torch.Tensor) -> torch.Tensor:
    """
    q: tensor containing a Unity3D quaternion (4 elements, x,y,z,w)
    v: tensor containing a set of points to be transformed via the quaternion (Nx3 for N points with x,y,z coordinates)
    """
    if q.ndim == 1:
        q = q.unsqueeze(0)
    if v.ndim == 1:
        v = v.unsqueeze(0)

    assert q.shape[0] == 1 and q.shape[1] == 4
    assert v.shape[1] == 3

    num = q[:,0] * 2.0
    num2 = q[:,1] * 2.0
    num3 = q[:,2] * 2.0
    num4 = q[:,0]*num
    num5 = q[:,1]*num2
    num6 = q[:,2]*num3
    num7 = q[:,0]*num2
    num8 = q[:,0]*num3
    num9 = q[:,1]*num3
    num10 = q[:,3]*num
    num11 = q[:,3]*num2
    num12 = q[:,3]*num3

    result = torch.zeros_like(v)

    result[:,0] = (1 - (num5+num6))*v[:,0] + (num7-num12)*v[:,1] + (num8+num11)*v[:,2]
    result[:,1] = (num7+num12)*v[:,0] + (1-(num4+num6))*v[:,1] +   (num9-num10)*v[:,2]
    result[:,2] = (num8-num11)*v[:,0] + (num9+num10)*v[:,1]  +   (1-(num4+num5))*v[:,2]
    return result


class TSControllerCoordinateTransform(nn.Module):

    def __init__(self, position_indices: torch.Tensor, rotation_indices: torch.Tensor,
                 reference_axis: torch.Tensor, additional_feature_indices: Optional[torch.Tensor] = None):
        
        super().__init__()
        self.position_indices = position_indices
        self.rotation_indices = rotation_indices
        self.reference_axis = reference_axis
        self.additional_feature_indices = additional_feature_indices

    def _get_pca_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: 1x3xT tensor with positions in first dimension (x,y,z) and time in the second dimension
        """
        assert x.shape[0:2] == (1,3)
        
        # build covariance matrix
        xm = x - torch.mean(x, dim = 2, keepdim = True)
        T = x.shape[2]
        cov = 1/(T-1)*torch.bmm(xm,xm.permute(0,2,1))
        e, v = tl.eigh(cov)     # v's columns are the eigenvectors (i.e the pca basis.T)
        basis = v.permute(0,2,1).squeeze()
        return basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: size 1xFxT with position and orientation indices defined in the __init__ function. This function does not support batch size > 1
        :returns: transformed features 1x(P+A)xT with P the position features and A the additional features, unaltered
        """

        Xpos = x[:,self.position_indices,:]
        Qrot = x[:,self.rotation_indices,0]
        if self.additional_feature_indices is not None:
            Add = x[:,self.additional_feature_indices,:]

        refPos = Xpos[:,:,0]
        basis = self._get_pca_basis(Xpos)

        pca_z = basis[-1,:]
        world_y = self.reference_axis

        ZFwd = quaternion_vec_mul(Qrot, torch.tensor([0,0,1], dtype = torch.float)).squeeze()

        # Flip the sign of pca_z to point towards the forward direction of the controller
        if torch.dot(ZFwd,pca_z) < 0:
            pca_z *= -1.0
        
        # new_up is the world_y
        new_up = world_y
        # the new_right vector is parallel to the gesture plane
        new_right = torch.cross(world_y, pca_z)
        # the new forward looks towards the fwd direction of the controller while remaining perpendicular to world_y and the new_right vector
        new_fwd = torch.cross(new_right,new_up)
        # construct the new basis
        new_basis = torch.stack((new_right, new_up, new_fwd))
        # the origin of the new basis is located at the first point of the gesture's trajectory
        newX = torch.bmm(new_basis.unsqueeze(0), Xpos - refPos.unsqueeze(-1))

        if self.additional_feature_indices is not None:
            newX = torch.cat((newX, Add))

        return newX