# import required packages
import gpt as g
import numpy as np


def create_copy_plan(grid):
    """create copy plan between gpt and numpy for the fine operator"""
    # get numpy matrix dimension
    N = grid.fsites * 4 * 3

    # allocate gpt vectors and numpy arrays (including dense matrix for operator)
    xg = g.vector_spin_color(grid, 4, 3)
    yg = g.vector_spin_color(grid, 4, 3)
    xn = np.zeros(N, complex)
    yn = np.empty(N, complex)
    D5 = np.empty((N, N), complex, order="F")  # "F" for column-major storage

    # create copy_plans between gpt and numpy
    g2n = g.copy_plan(yn, yg)
    g2n.source += yg.view[:]
    g2n.destination += g.global_memory_view(
        yg.grid,
        [[yg.grid.processor, yn, 0, yn.nbytes]] if yn.nbytes > 0 else None,
    )
    g2n = g2n()
    n2g = g.copy_plan(xg, xn)
    n2g.source += g.global_memory_view(
        xg.grid,
        [[xg.grid.processor, xn, 0, xn.nbytes]] if xn.nbytes > 0 else None,
    )
    n2g.destination += xg.view[:]
    n2g = n2g()

    # put everything in a dictionary and return
    copy_plan = {
        "g2n": g2n,
        "n2g": n2g,
        "xg": xg,
        "yg": yg,
        "xn": xn,
        "yn": yn,
        "D5": D5,
    }
    return copy_plan


def create_copy_plan_coarse(grid, nbasisvectors):
    """create copy plan between gpt and numpy for the coarse operator"""
    # get numpy matrix dimension
    N = grid.fsites * nbasisvectors

    # allocate gpt vectors and numpy arrays (including dense matrix for operator)
    xg = g.vector_complex_additive(grid, nbasisvectors)
    yg = g.vector_complex_additive(grid, nbasisvectors)
    xn = np.zeros(N, complex)
    yn = np.empty(N, complex)
    D5 = np.empty((N, N), complex, order="F")  # "F" for column-major storage

    # create copy_plans between gpt and numpy
    g2n = g.copy_plan(yn, yg)
    g2n.source += yg.view[:]
    g2n.destination += g.global_memory_view(
        yg.grid,
        [[yg.grid.processor, yn, 0, yn.nbytes]] if yn.nbytes > 0 else None,
    )
    g2n = g2n()
    n2g = g.copy_plan(xg, xn)
    n2g.source += g.global_memory_view(
        xg.grid,
        [[xg.grid.processor, xn, 0, xn.nbytes]] if xn.nbytes > 0 else None,
    )
    n2g.destination += xg.view[:]
    n2g = n2g()

    # put everything in a dictionary and return
    copy_plan = {
        "g2n": g2n,
        "n2g": n2g,
        "xg": xg,
        "yg": yg,
        "xn": xn,
        "yn": yn,
        "D5": D5,
    }
    return copy_plan


def eigvals(grid, op, N=None, copy_plan=None):
    """calculate full spectrum for a GPT operator

    Args:
        grid: GPT operator
        op: GPT grid
        N (optional): operator dimension i.e. number of eigenvalues. defaults to grid.fsites * 4 * 3.
        copy_plan (optional): copy plan between GPT and numpy. defaults to fine grid plan.

    Returns:
        _type_: _description_
    """
    # get numpy matrix dimension if not specified
    if N is None:
        N = grid.fsites * 4 * 3

    # get copy plan if not specified
    if copy_plan is None:
        copy_plan = create_copy_plan(grid)

    # get copy_plans and associated arrays
    g2n = copy_plan["g2n"]
    n2g = copy_plan["n2g"]
    xg = copy_plan["xg"]
    yg = copy_plan["yg"]
    xn = copy_plan["xn"]
    yn = copy_plan["yn"]
    D5 = copy_plan["D5"]

    # compute matrix D
    for i in range(N):
        xn[i] = 1
        n2g(xg, xn)
        xn[i] = 0
        yg @= op * xg
        g2n(yn, yg)
        D5[:, i] = yn

    # compute eigenvalues
    evals = np.linalg.eigvals(D5)

    # return eigenvalues
    return evals
