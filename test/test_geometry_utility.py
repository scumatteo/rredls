import torch

import sys

sys.path.insert(0, "src/")

from broadcast.geometry_utility import (
    invert_transformation_matrices, compose_transformation_matrices, concatenate_transformations, compute_relative_transformations)
from utility.benchmarking_utility import evaluate_time



def test_invert_transformation_matrices(T: torch.Tensor) -> torch.Tensor:
    T_inv = invert_transformation_matrices(T)
    true_T_inv = torch.tensor([0.6660276, 0.7222504, 0.1864445, -0.1864445, -0.1056860, -0.1560583,
                              0.9820775, -0.9820775, 0.7384021, -0.6737953, -0.0276074, 0.0276074, 0, 0, 0, 1], device=T.device).view((4, 4))
    assert torch.allclose(T_inv, true_T_inv)
    return T_inv


def test_compose_transformation_matrices(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    T = compose_transformation_matrices(R, t)
    assert torch.allclose(T[:3, :3], R)
    assert torch.allclose(T[:3, 3], t)


def test_concatenate_transformations(initial: torch.Tensor, T: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
    new_T = concatenate_transformations(initial, T)
    assert torch.allclose(new_T, result)
    return new_T

def test_compute_relative_transformations(T: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
    relative_T = compute_relative_transformations(T)
    assert torch.allclose(relative_T, result)
    return relative_T


if __name__ == "__main__":
    device = "cpu"
    T = torch.tensor([0.6660276, -0.1056860,  0.7384021, 0, 0.7222504, -0.1560583,
                      -0.6737953, 0, 0.1864445,  0.9820775, -0.0276074, 1, 0, 0, 0, 1], device=device).view((4, 4))
    T_inv = test_invert_transformation_matrices(T)

    R = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1],
                     device=device).view((3, 3)).float()
    t = torch.tensor([0, 0, 0], device=device).float()

    T_eye = test_compose_transformation_matrices(R, t)
    # another test
    T = test_compose_transformation_matrices(T[:3, :3], T[:3, 3])
    
    # create 5 transformation matrices with no rotation and with a translation of 1 on z-axis.
    initial = torch.eye(4, device=device)
    relative_T = torch.eye(4, device=device)
    relative_T[2, 3] = 1
    T = [] # matrices of relative transformations, all identical
    result = [initial] # matrices that are the result of the concatenation
    for i in range(1, 6):
        T.append(relative_T)
        T_res = torch.eye(4)
        T_res[2, 3] = i
        result.append(T_res)
    T = torch.stack(T)
    result = torch.stack(result)
    final_T = test_concatenate_transformations(initial, T, result)
    
    # given the matrices of the concatenation final_T, the relative transformations should be equals to T
    test_compute_relative_transformations(final_T, T) 
    
    
    
    
