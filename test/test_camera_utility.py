import torch

import sys

sys.path.insert(0, "src/")
from broadcast.camera_utility import invert_intrinsic, scale_intrinsic
from utility.benchmarking_utility import evaluate_time


def test_invert_intrinsic(K: torch.Tensor) -> None:
    K_inv = evaluate_time(invert_intrinsic)(K)
    K_inv_torch = evaluate_time(torch.linalg.inv)(K) # better than mine for small matrices
    assert torch.allclose(K_inv, K_inv_torch)

def test_scale_intrinsic(K: torch.Tensor, scale_factor: int) -> None:
    K_scaled = scale_intrinsic(K, scale_factor)
    K_result = torch.eye(3)
    K_result[0, 0] = K[0, 0] * scale_factor
    K_result[0, 2] = K[0, 2] * scale_factor
    K_result[1, 1] = K[1, 1] * scale_factor
    K_result[1, 2] = K[1, 2] * scale_factor

    assert torch.allclose(K_scaled, K_result)

if __name__ == "__main__":
    device = "cpu"
    K = torch.tensor([960., 0., 960., 0., 960., 540., 0., 0., 1.], device=device).reshape((3, 3))
    
    test_invert_intrinsic(K)
    
    scale_factor = 2
    test_scale_intrinsic(K, scale_factor)
    
    scale_factor = 0.5
    test_scale_intrinsic(K, scale_factor)
    
    