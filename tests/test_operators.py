import numpy as np

from eigendiffusion.operators import eigendecompose, neumann_laplacian_1d


def test_neumann_operator_is_symmetric_and_conserves_mass():
    operator = neumann_laplacian_1d(11, length=4.0, diffusion_coefficient=2.2e-4)
    assert np.allclose(operator, operator.T)
    assert np.allclose(operator.sum(axis=1), 0.0)


def test_eigenvalues_are_nonnegative_and_zero_mode_is_constant():
    operator = neumann_laplacian_1d(11, length=4.0, diffusion_coefficient=2.2e-4)
    basis = eigendecompose(operator)
    assert np.all(basis.eigenvalues >= -1e-14)
    assert basis.eigenvalues[0] < 1e-12
    assert np.allclose(np.abs(basis.eigenvectors[:, 0]), 1.0 / np.sqrt(11))
