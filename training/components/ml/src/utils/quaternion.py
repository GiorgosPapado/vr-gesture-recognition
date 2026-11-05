import numpy as np
import nptyping as npt
import typing as tp

# private static Quaternion QuatMul(Quaternion q1, Quaternion q2)
# {
#     float w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
#     float x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
#     float y = q1.w * q2.y + q1.y * q2.w + q1.z * q2.x - q1.x * q2.z;
#     float z = q1.w * q2.z + q1.z * q2.w + q1.x * q2.y - q1.y * q2.x;
#     Quaternion r = new Quaternion(x, y, z, w);
#     return r;
# }

def quaternion_mul(q1 : tp.Union[npt.NDArray[4,float], npt.NDArray[(tp.Any,4),float]],q2 : tp.Union[npt.NDArray[4,float],npt.NDArray[(tp.Any,4),float]]) -> npt.NDArray[(tp.Any,4),float]:

    if q1.ndim == 1:
        q1 = np.expand_dims(q1,axis=0)
    if q2.ndim == 1:
        q2 = np.expand_dims(q2,axis=0)

    # quaternions in form x,y,z,w

    w = q1[:,3]*q2[:,3] - q1[:,0]*q2[:,0] - q1[:,1]*q2[:,1] - q1[:,2]*q2[:,2]
    x = q1[:,3]*q2[:,0] + q1[:,0]*q2[:,3] + q1[:,1]*q2[:,2] - q1[:,2]*q2[:,1]
    y = q1[:,3]*q2[:,1] + q1[:,1]*q2[:,3] + q1[:,2]*q2[:,0] - q1[:,0]*q2[:,2]
    z = q1[:,3]*q2[:,2] + q1[:,2]*q2[:,3] + q1[:,0]*q2[:,1] - q1[:,1]*q2[:,0]

    r = np.stack([x,y,z,w],axis=1)
    return r

# Directly ripped from Unity3D
# public static Vector3 operator *(Quaternion rotation, Vector3 point)
# {
#     float num = rotation.x * 2f;
#     float num2 = rotation.y * 2f;
#     float num3 = rotation.z * 2f;
#     float num4 = rotation.x * num;
#     float num5 = rotation.y * num2;
#     float num6 = rotation.z * num3;
#     float num7 = rotation.x * num2;
#     float num8 = rotation.x * num3;
#     float num9 = rotation.y * num3;
#     float num10 = rotation.w * num;
#     float num11 = rotation.w * num2;
#     float num12 = rotation.w * num3;
#     Vector3 result = default(Vector3);
#     result.x = (1f - (num5 + num6)) * point.x + (num7 - num12) * point.y + (num8 + num11) * point.z;
#     result.y = (num7 + num12) * point.x + (1f - (num4 + num6)) * point.y + (num9 - num10) * point.z;
#     result.z = (num8 - num11) * point.x + (num9 + num10) * point.y + (1f - (num4 + num5)) * point.z;
#     return result;
# }
def quaternion_vec_mul(q : tp.Union[npt.NDArray[4,float], npt.NDArray[(tp.Any,4),float]], v : tp.Union[npt.NDArray[3,float],npt.NDArray[(tp.Any,3),float]]) -> npt.NDArray[(tp.Any,3),float]:
    if q.ndim == 1:
        q = np.expand_dims(q,axis=0)
    if v.ndim == 1:
        v = np.expand_dims(v,axis=0)


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

    result = np.zeros_like(v)

    result[:,0] = (1 - (num5+num6))*v[:,0] + (num7-num12)*v[:,1] + (num8+num11)*v[:,2]
    result[:,1] = (num7+num12)*v[:,0] + (1-(num4+num6))*v[:,1] +   (num9-num10)*v[:,2]
    result[:,2] = (num8-num11)*v[:,0] + (num9+num10)*v[:,1]  +   (1-(num4+num5))*v[:,2]
    return result

def quaternion_inv(q : npt.NDArray[(tp.Any,4),float], eps = 1e-18) -> npt.NDArray[(tp.Any,4),float]:
    if q.ndim == 1:
        q = np.expand_dims(q,axis=0)

    norm = q[:,3]*q[:,3] + q[:,0]*q[:,0] + q[:,1]*q[:,1] + q[:,2]*q[:,2] + eps
    w =  q[:,3] / norm
    x = -q[:,0] / norm
    y = -q[:,1] / norm
    z = -q[:,2] / norm

    r = np.stack([x,y,z,w],axis=1)
    return r

# //https://www.mathworks.com/help/aeroblks/quaternioninverse.html
# private static Quaternion QuatInv(Quaternion q)
# {
#     float norm = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z + 1e-10f;
#     float w = q.w/norm;
#     float x = -q.x/norm;
#     float y = -q.y/norm;
#     float z = -q.z/norm;
#     return new Quaternion(x, y, z, w);
# }

# 
# public void RotateAround(Vector3 point, Vector3 axis, float angle)
# {
# 	Vector3 vector = position;
# 	Quaternion quaternion = Quaternion.AngleAxis(angle, axis);
# 	Vector3 vector2 = vector - point;
# 	vector2 = quaternion * vector2;
# 	vector = (position = point + vector2);
# 	RotateAroundInternal(axis, angle * ((float)Math.PI / 180f));
# }

