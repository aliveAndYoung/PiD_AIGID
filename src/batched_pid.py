import torch

def get_residuals(x: torch.Tensor) -> torch.Tensor:
    """
    calculates the PiD residual for a batch of tensors
    ( b , c , h , w )  -> ( b , c ,  h*w  ) -> ( b , h*w , c(3) )  -> ( b , c ,  h , w  )
    """
# transform matrix Myuv in GPU
    m_t = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.147, -0.289, 0.436],
        [0.615, -0.515, -0.100]
    ], device=x.device, dtype=x.dtype)
    
# inverse
    m_inv = torch.inverse(m_t)

# reshaping to a linear flat form to apply matrix multiplication on GPU
    b, c, h, w = x.shape

#  ( b , c , h , w )  -> ( b , c ,  h*w  ) -> ( b , h*w , c(3) ) 
    x_flat = x.view(b, c, -1).transpose(1, 2) 

#   X * Myuv
    yuv = torch.matmul(x_flat, m_t.T)

# applying Q(x)  [ floor ]
    yuv_quantized = torch.floor(yuv * 255) / 255

# X` * (Myuv)^-1
    rgb_quantized_flat = torch.matmul(yuv_quantized, m_inv.T)

#  ( b , h*w , c(3) )  -> ( b , c ,  h , w  )
    rgb_quantized = rgb_quantized_flat.transpose(1, 2).view(b, c, h, w)

    # retrieve the residual
    residual = x - rgb_quantized

    # Normalize to fit ImageNet standard ( NN inputs )
    return normalize_batch(residual)

def normalize_batch(tensor: torch.Tensor) -> torch.Tensor:
    """
    normalizes a batch of images to fit the ImageNet standard  ( Xnorm = (X - mean) / std )
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std