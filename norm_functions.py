import torch
import torch.nn.functional as F

def angle_normalize(vector, dim=1):
    """Normalizes network output by interpreting them as angles
    and using the sine and cosine functions to generate the coordinates on an n-sphere (in R^(n+1))
    
    :param vector: (batch_dim, feature_dim) or (epoch_dim, batch_dim, feature_dim)
    :param int dim: the dimension over which to normalize
    """
    device = vector.device
    angles = vector
    sample_n = angles.shape[-2]
    padding_shape = [sample_n, 1]
    if len(angles.shape) >= 3:
        epoch_dim = angles.shape[-3]
        padding_shape.insert(0, epoch_dim)
    
    cum_sin = torch.cumprod(torch.sin(angles), dim=dim)
    cum_sin = torch.cat([torch.ones(padding_shape, device=device), cum_sin], dim=dim)
    cos = torch.cos(angles)
    cos = torch.cat([cos, torch.ones(padding_shape, device=device)], dim=dim)
    coordinates = cum_sin * cos
    return coordinates

def angle2_normalize(vector, dim=1):
    """same as angle_normalization, but sin and cosine are switched
    Normalizes network output by interpreting them as angles
    and using the sine and cosine functions to generate the coordinates on an n-sphere (in R^(n+1))
    
    :param torch.tensor vector: (batch_dim, feature_dim) or (epoch_dim, batch_dim, feature_dim)
    :param int dim: the dimension over which to normalize
    """
    device = vector.device
    angles = vector
    sample_n = angles.shape[-2]
    padding_shape = [sample_n, 1]
    if len(angles.shape) >= 3:
        epoch_dim = angles.shape[-3]
        padding_shape.insert(0, epoch_dim)
    
    cum_cos = torch.cumprod(torch.cos(angles), dim=dim)
    cum_cos = torch.cat([torch.ones(padding_shape, device=device), cum_cos], dim=dim)
    sin = torch.sin(angles)
    sin = torch.cat([sin, torch.ones(padding_shape, device=device)], dim=dim)
    coordinates = cum_cos * sin
    return coordinates

def extra_dim_normalize(
        vector,
        dim = 1,
        function_of_norm = torch.cos # this function should map [0,inf) -> [-1,1]
        ):
    norm = vector.norm(dim=dim, keepdim=True)
    extra_dim = function_of_norm(norm)
    other_dims = torch.sqrt(1-extra_dim**2) * F.normalize(vector)
    extended_vec = torch.cat([extra_dim, other_dims], dim=dim)
    
    return extended_vec

def exp_map_normalize(vector, dim=1, norm_scaling=1):
    """Normalizes vector by using exponential map representation, the resulting vector will lie on the sphere.
    Almost the same as the normalization function but adds a new dimension
    based on the cosine of the norm 
    """
    norm = vector.norm(dim=dim, keepdim=True) * norm_scaling
    vec_normalized = F.normalize(vector, dim=dim)
    extra_dim = torch.cos(norm)
    output = torch.cat([extra_dim, torch.sin(norm)*vec_normalized], dim=dim)
    return output

def minus_exp_map_normalize(vector, dim=1, norm_scaling=1):
    """Normalizes vector by using exponential map representation (but the extra dim is multiplied with -1), the resulting vector will lie on the sphere.
    Almost the same as the normalization function but adds a new dimension
    based on the cosine of the norm 
    """
    norm = vector.norm(dim=dim, keepdim=True) * norm_scaling
    vec_normalized = F.normalize(vector, dim=dim)
    extra_dim = -torch.cos(norm)
    output = torch.cat([extra_dim, torch.sin(norm)*vec_normalized], dim=dim)
    return output

def stereo_normalize(vector, dim=1, norm_scaling=1):
    """Normalizes vector by using stereographic projection, the resulting vector will lie on the sphere.
    Adds new dimension based on norm"""
    def extra_dim(norm):
        s_norm = norm*norm_scaling
        return ((s_norm)**2 - 1)/((s_norm)**2 +1)
    
    return extra_dim_normalize(vector, dim, extra_dim)

def stereo_minus_mean_normalize(vector, dim=1):
    """Normalizes vector by using stereographic projection, the resulting vector will lie on the sphere.
    Adds new dimension based on norm"""
    def extra_dim(norm):
        s_norm = norm - norm.mean().detach()
        print("norm", norm.mean().detach(), "s_norm min:", s_norm.min())
        return ((s_norm)**2 - 1)/((s_norm)**2 +1)
    
    return extra_dim_normalize(vector, dim, extra_dim)

def stereo_times_mean_normalize(vector, dim=1):
    """Normalizes vector by using stereographic projection, the resulting vector will lie on the sphere.
    Adds new dimension based on norm"""
    def extra_dim(norm):
        s_norm = norm * norm.mean().detach()
        return ((s_norm)**2 - 1)/((s_norm)**2 +1)
    
    return extra_dim_normalize(vector, dim, extra_dim)

def stereo_div_mean_normalize(vector, dim=1):
    """Normalizes vector by using stereographic projection, the resulting vector will lie on the sphere.
    Adds new dimension based on norm"""
    def extra_dim(norm):
        s_norm = norm / norm.mean().detach()
        print(norm.mean().detach())
        return ((s_norm)**2 - 1)/((s_norm)**2 +1)
    
    return extra_dim_normalize(vector, dim, extra_dim)

def mono_normalize(vector, dim=1, norm_scaling=1):
    """Normalizes vector, like stereographic projection, but without squaring the norm.  
    The resulting vector will lie on the sphere.
    Adds new dimension based on norm"""
    def extra_dim(norm):
        s_norm = norm*norm_scaling
        return (s_norm - 1)/(s_norm +1)
    
    return extra_dim_normalize(vector, dim, extra_dim)

def line_normalize(vector, dim=1, norm_scaling=1):
    """Adds a new dimension, linearly based on the norm"""
    def extra_dim(norm):
        s_norm = norm*norm_scaling
        return s_norm * (s_norm <= 2) - 1 + 1 * (s_norm > 2)
    
    return extra_dim_normalize(vector, dim, extra_dim)

def exp2_normalize(vector, dim=1):
    """Adds a new dimension, based on the norm"""
    def extra_dim(norm):
        s_norm = norm #*norm_scaling
        return (torch.exp2(s_norm) - 2)/(torch.exp2(s_norm))

    return extra_dim_normalize(vector, dim, extra_dim)

def exponent_normalize(vector, dim=1, exponent=1):
    """Adds a new dimension, based on the norm"""
    def extra_dim(norm):
        s_norm = norm
        return (s_norm**exponent - 1)/(s_norm**exponent +1)
    
    return extra_dim_normalize(vector, dim, extra_dim)

def torus_norm(
        vector,
        dim = -1
        ):
    """Uses each v_i in the input vector as an angle for a circle i.e. computing sin(v_i) and cos(v_i)
    this is equivalent to torus. The resulting vectors are concatenated,
    thus the x and y coordinates are seperated by n
    e.g x_i = torus_coordinates[..,i], y_i = torus_coordinates[..,i+n]
    finally the coordinates get normalized by deviding by the root of n,
    thus the product of two vectors normalized is between -1 and +1"""
    n = vector.shape[dim]
    sin = torch.sin(vector)
    cos = torch.cos(vector)
    torus_coordinates = torch.cat((sin,cos), dim=dim)
    normalized_coordinates = torus_coordinates / (n**0.5)
    return normalized_coordinates

# class Stereo:
#     def __call__(self, norm):
#         return (norm**2 - 1)/(norm**2 + 1)
#     def __str__(self):
#         return "(‖v‖² - 1)/(‖v‖² + 1)"
#     def __repr__(self):
#         return self.__str__()

# class ExtraDimNormalize:
#     def __init__(self, FunctionOfNorm, dim=1):
#         self.dim = dim
#         self.function_of_norm = FunctionOfNorm()

#     def __call__(self, vector):
#         norm = vector.norm(dim=self.dim, keepdim=True)
#         extra_dim = self.function_of_norm(norm)
#         other_dims = torch.sqrt(1-extra_dim**2) * F.normalize(vector)
#         extended_vec = torch.cat([extra_dim, other_dims], dim=self.dim)
#         return extended_vec
    
#     def __str__(self):
#         return f"""f(v)_[0] = {self.function_of_norm}
# f(v)_[1,n] = sqrt(1 - {self.function_of_norm} * v ╱ ‖v‖"""
    
#     def __repr__(self):
#         return self.__str__()

# def remove(value, deletechars='\\/:*?"<>| '):
#     for c in deletechars:
#         value = value.replace(c,'')
#     return value

# # print(remove(str(ExtraDimNormalize(Stereo)), '\\/:*?"<>| '))