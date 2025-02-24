# import required packages
import numpy as np
import torch


def qcdml_eigvals(dimensions, op, verbose=False):
    """calculate eigenvalues of a qcdml operator

    Args:
        dimensions: [L0, L1, .., Ld]
        op: operator
        verbose (bool, optional): verbose mode. defaults to false

    Returns:
        eigenvalues
    """
    # get numpy matrix dimension if not specified
    N = np.prod(dimensions) * 4 * 3

    # compute matrix D
    xn = np.zeros(N, complex)
    D_full = np.zeros((N, N), complex)
    for i in range(N):
        if verbose and i % 32 == 0: 
            print(f"\rFilling matrix {i/N * 100:.2f}%", end="")
        xn[i] = 1
        xn_rs = xn.reshape(dimensions + [4, 3])
        xt = torch.from_numpy(xn_rs)
        yt = op(xt)
        yn = yt.numpy()
        yn_rs = yn.reshape(N)
        D_full[:, i] = yn_rs
        xn[i] = 0
    if verbose:
        print("\rFilling matrix 100%\033[K")

    # compute eigenvalues
    evals = np.linalg.eigvals(D_full)

    # return eigenvalues
    return evals
