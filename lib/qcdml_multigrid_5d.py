"""
Provides Multigrid with zero point projection. Copied from qcd_ml.util.qcd.multigrid and modified for 5d.
"""


import torch
import itertools
import numpy as np


def innerproduct(x, y): return (x.conj() * y).sum()
def norm(x): return torch.sqrt(innerproduct(x, x).real)


def orthonormalize(vecs):
    basis = []
    for vec in vecs:
        for b in basis:
            vec = vec - innerproduct(b, vec) * b
        vec = vec / norm(vec)
        basis.append(vec)
    return basis


class ZPP_Multigrid_5d:
    """
    Multigrid with zeropoint projection.

    Use ``.v_project`` and ``.v_prolong`` to project and prolong vectors.
    Use ``.get_coarse_operator`` to construct a coarse operator.

    use ``ZPP_Multigrid.gen_from_fine_vectors([random vectors], [i, j, k, l], lambda b, xo: <solve Dx = b for x>)``
    to construct a ``ZPP_Multigrid``.
    """

    def __init__(self, block_size, ui_blocked, n_basis, L_coarse, L_fine):
        self.block_size = block_size
        self.ui_blocked = ui_blocked
        self.n_basis = n_basis
        self.L_coarse = L_coarse
        self.L_fine = L_fine

    def cuda(self):
        ui_blocked = list(np.empty(self.L_coarse, dtype=object))
        for bs, bx, by, bz, bt in itertools.product(*(range(li) for li in self.L_coarse)):  # ! 5d change
            ui_blocked[bs][bx][by][bz][bt] = [uib.cuda() for uib in self.ui_blocked[bs][bx][by][bz][bz]]  # ! 5d change

        return self.__class__(self.block_size, ui_blocked, self.n_basis, self.L_coarse, self.L_fine)

    @classmethod
    def from_basis_vectors(cls, basis_vectors, block_size):
        """
        Used to generate a multigrid setup using basis vectors and a block size.
        The basis vectors can be obtained using ``.get_basis_vectors()`` method.
        """
        n_basis = len(basis_vectors)
        L_fine = list(basis_vectors[0].shape[:5])  # ! 5d change
        L_coarse = [lf // bs for lf, bs in zip(L_fine, block_size)]

        # Perform blocking
        ls, lx, ly, lz, lt = block_size  # ! 5d change
        ui_blocked = list(np.empty(L_coarse, dtype=object))

        for bs, bx, by, bz, bt in itertools.product(*(range(li) for li in L_coarse)):  # ! 5d change
            for uk in basis_vectors:
                u_block = uk[bs * ls: (bs + 1) * ls  # ! 5d change
                             , bx * lx: (bx + 1) * lx, by * ly: (by + 1) * ly, bz * lz: (bz + 1) * lz, bt * lt: (bt + 1) * lt]
                if ui_blocked[bs][bx][by][bz][bt] is None:  # ! 5d change
                    ui_blocked[bs][bx][by][bz][bt] = []  # ! 5d change
                ui_blocked[bs][bx][by][bz][bt].append(u_block)  # ! 5d change

            # Orthogonalize over block
            ui_blocked[bs][bx][by][bz][bt] = orthonormalize(ui_blocked[bs][bx][by][bz][bt])  # ! 5d change

        return cls(block_size, ui_blocked, n_basis, L_coarse, L_fine)

    @classmethod
    def gen_from_fine_vectors(cls, fine_vectors, block_size, solver, verbose=False):
        """
        Used to generate a multigrid setup using fine vectors, a block size and a solver.

        solver should be 
            ``(x, info) = solver(b, x0)``
        which solves
            :math:`D x = b`

        we will choose
            ``b = torch.zeros_like(x0)``

        """
        # length of basis
        n_basis = len(fine_vectors)
        # normalize
        bv = [bi / norm(bi) for bi in fine_vectors]
        # compute zero point vectors
        zero = torch.zeros_like(bv[0])
        ui = []
        for i, b in enumerate(bv):
            uk, ret = solver(zero, b)
            if verbose:
                print(f"[{i:2d}]: {ret['converged']} ({ret['k']:5d}) <{ret['res']:.4e}>")
            ui.append(uk)

        # size of fine lattice
        L_fine = list(uk.shape[:5])  # ! 5d change
        # size of coarse lattice
        L_coarse = [lf // bs for lf, bs in zip(L_fine, block_size)]

        # Perform blocking
        ls, lx, ly, lz, lt = block_size  # ! 5d change
        ui_blocked = list(np.empty(L_coarse, dtype=object))

        for bs, bx, by, bz, bt in itertools.product(*(range(li) for li in L_coarse)):  # ! 5d change
            for uk in ui:
                u_block = uk[bs * ls: (bs + 1) * ls  # ! 5d change
                             , bx * lx: (bx + 1) * lx, by * ly: (by + 1) * ly, bz * lz: (bz + 1) * lz, bt * lt: (bt + 1) * lt]
                if ui_blocked[bs][bx][by][bz][bt] is None:  # ! 5d change
                    ui_blocked[bs][bx][by][bz][bt] = []  # ! 5d change
                ui_blocked[bs][bx][by][bz][bt].append(u_block)  # ! 5d change

            # Orthogonalize over block
            ui_blocked[bs][bx][by][bz][bt] = orthonormalize(ui_blocked[bs][bx][by][bz][bt])  # ! 5d change

        return cls(block_size, ui_blocked, n_basis, L_coarse, L_fine)

    def v_project(self, v):
        """
        project fine vector ``v`` to coarse grid.
        """
        projected = torch.zeros(self.L_coarse + [self.n_basis], dtype=torch.cdouble)
        ls, lx, ly, lz, lt = self.block_size  # ! 5d change

        for bs, bx, by, bz, bt in itertools.product(*(range(li) for li in self.L_coarse)):  # ! 5d change
            for k, uk in enumerate(self.ui_blocked[bs][bx][by][bz][bt]):  # ! 5d change
                projected[bs, bx, by, bz, bt, k] = innerproduct(uk, v[bs * ls: (bs + 1) * ls  # ! 5d change
                                                                , bx * lx: (bx + 1) * lx, by * ly: (by + 1) * ly, bz * lz: (bz + 1) * lz, bt * lt: (bt + 1) * lt])
        return projected

    def v_prolong(self, v):
        """
        prolong coarse vector ``v`` to fine grid.
        """
        ls, lx, ly, lz, lt = self.block_size  # ! 5d change
        prolonged = torch.zeros(self.L_fine + list(self.ui_blocked[0][0][0][0][0][0].shape[5:]), dtype=torch.cdouble)  # ! 5d change
        for bs, bx, by, bz, bt in itertools.product(*(range(li) for li in self.L_coarse)):  # ! 5d change
            for k, uk in enumerate(self.ui_blocked[bs][bx][by][bz][bt]):  # ! 5d change
                prolonged[bs * ls: (bs + 1) * ls  # ! 5d change
                          , bx * lx: (bx + 1) * lx, by * ly: (by + 1) * ly, bz * lz: (bz + 1) * lz, bt * lt: (bt + 1) * lt] += uk * v[bs, bx, by, bz, bt, k]  # ! 5d change
        return prolonged

    def get_coarse_operator(self, fine_operator):
        """
        Given a fine operator ``fine_operator(psi)``, construct a coarse operator.

        In case of a 9-point operator, such as Wilson and Wilson-Clover Dirac operator,
        a significantly faster implementation can be achieved by using ``qcd_ml.qcd.dirac.coarsened.coarse_9point_op_NG``.
        """
        def operator(source_coarse):
            source_fine = self.v_prolong(source_coarse)
            dst_fine = fine_operator(source_fine)
            return self.v_project(dst_fine)
        return operator

    def save(self, filename):
        """
        This is a stupid implementation. Saves all arguments as a list.
        """
        torch.save([self.block_size, self.ui_blocked, self.n_basis, self.L_coarse, self.L_fine], filename)

    @classmethod
    def load(cls, filename):
        """
        This is a stupid implementation. Loads all arguments as a list.
        """
        args = torch.load(filename)
        return cls(*tuple(args))

    # def get_basis_vectors(self):
    #    """
    #    Returns the basis vectors. This function is necessary because the basis vectors are stored
    #    "by-coarse-grid-index" and not on a fine grid.
    #    """
    #    result = torch.zeros(self.n_basis, *self.L_fine, 4, 3, dtype=torch.cdouble)
    #    for bx, by, bz, bt in itertools.product(*(range(li) for li in self.L_coarse)):
    #        for k, uk in enumerate(self.ui_blocked[bx][by][bz][bt]):
    #            result[k, bx * self.block_size[0]: (bx + 1)*self.block_size[0]
    #                  , by * self.block_size[1]: (by + 1)*self.block_size[1]
    #                  , bz * self.block_size[2]: (bz + 1)*self.block_size[2]
    #                  , bt * self.block_size[3]: (bt + 1)*self.block_size[3]] = uk
    #    return result
