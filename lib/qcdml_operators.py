# import required packages
import gpt as g
import numpy as np
import torch
from qcdml_convert import lattice2ndarray_5d, ndarray2lattice_5d


def qcdml_dw(U_gpt, mobius_p):
    """torch wrapper for the möbius domain-wall operator from GPT

    Args:
        U_gpt: gauge field in GPT
        mobius_p: operator parameters

    Returns:
        function: f(x) = D_dw * x
    """
    D_dw = g.qcd.fermion.mobius(U_gpt, mobius_p)
    grid5 = D_dw.F_grid
    
    def apply(v_qcdml):
        v_gpt = ndarray2lattice_5d(v_qcdml.numpy(), grid5, g.vspincolor)
        w_gpt = D_dw(v_gpt)
        w_qcdml = torch.tensor(lattice2ndarray_5d(w_gpt))
        return w_qcdml

    return apply


def qcdml_dw_inv(U_gpt, mobius_p, solver_kwargs):
    """torch wrapper for the inverse of the möbius domain-wall operator from GPT

    Args:
        U_gpt: gauge field in GPT
        mobius_p: operator parameters
        solver_kwargs: solver parameters to calculate inverse

    Returns:
        function: f(x) = D_dw^{-1} * x
    """
    D_dw = g.qcd.fermion.mobius(U_gpt, mobius_p)
    grid5 = D_dw.F_grid
    
    cg = g.algorithms.inverter.cg(**solver_kwargs)
    prec = g.qcd.fermion.preconditioner.eo2_ne(parity=g.odd)
    slv = g.algorithms.inverter.preconditioned(prec, cg)
    D_dw_inv = slv(D_dw)
    
    def apply(v_qcdml):
        v_gpt = ndarray2lattice_5d(v_qcdml.numpy(), grid5, g.vspincolor)
        w_gpt = D_dw_inv(v_gpt)
        w_qcdml = torch.tensor(lattice2ndarray_5d(w_gpt))
        return w_qcdml

    return apply


def qcdml_pv(U_gpt, mobius_p):
    """torch wrapper for the daggered pauli-villars operator from GPT

    Args:
        U_gpt: gauge field in GPT
        mobius_p: operator parameters
        solver_kwargs: solver parameters to calculate inverse

    Returns:
        function: f(x) = D_pv^{\dag} * x = D_dw(m=1)^{\dag} * x
    """
    pv_p = mobius_p.copy()
    pv_p["mass"] = 1.0
    D_pv_dag = g.adj(g.qcd.fermion.mobius(U_gpt, pv_p))
    grid5 = D_pv_dag.F_grid
    
    def apply(v_qcdml):
        v_gpt = ndarray2lattice_5d(v_qcdml.numpy(), grid5, g.vspincolor)
        w_gpt = D_pv_dag(v_gpt)
        w_qcdml = torch.tensor(lattice2ndarray_5d(w_gpt))
        return w_qcdml

    return apply


def qcdml_pv_dw(U_gpt, mobius_p):
    """torch wrapper for the product of the daggered pauli-villars and domain-wall operators from GPT

    Args:
        U_gpt: gauge field in GPT
        mobius_p: operator parameters

    Returns:
        function: f(x) = D_pv^{\dag} * D_dw * x
    """
    D_dw = g.qcd.fermion.mobius(U_gpt, mobius_p)
    grid5 = D_dw.F_grid
    pv_p = mobius_p.copy()
    pv_p["mass"] = 1.0
    D_pv = g.qcd.fermion.mobius(U_gpt, pv_p)
    D_op = g.adj(D_pv) * D_dw
    
    def apply(v_qcdml):
        v_gpt = ndarray2lattice_5d(v_qcdml.numpy(), grid5, g.vspincolor)
        w_gpt = D_op(v_gpt)
        w_qcdml = torch.tensor(lattice2ndarray_5d(w_gpt))
        return w_qcdml

    return apply


def qcdml_pv_dw_inv(U_gpt, mobius_p, solver_kwargs):
    """torch wrapper for the inverse of the product of the daggered pauli-villars and domain-wall operators from GPT

    Args:
        U_gpt: gauge field in GPT
        mobius_p: operator parameters
        solver_kwargs: solver parameters to calculate inverse

    Returns:
        function: f(x) = (D_pv^{\dag} * D_dw)^{-1} * x
    """
    D_dw = g.qcd.fermion.mobius(U_gpt, mobius_p)
    grid5 = D_dw.F_grid
    pv_p = mobius_p.copy()
    pv_p["mass"] = 1.0
    D_pv_adj = g.adj(g.qcd.fermion.mobius(U_gpt, pv_p))
    
    cg = g.algorithms.inverter.cg(**solver_kwargs)
    prec = g.qcd.fermion.preconditioner.eo2_ne(parity=g.odd)
    slv = g.algorithms.inverter.preconditioned(prec, cg)
    D_dw_inv = slv(D_dw)
    slv_pv = g.algorithms.inverter.preconditioned(prec, cg)
    D_pv_adj_inv = slv_pv(D_pv_adj)
    
    def apply(v_qcdml):
        v_gpt = ndarray2lattice_5d(v_qcdml.numpy(), grid5, g.vspincolor)
        w_gpt = D_dw_inv(D_pv_adj_inv(v_gpt))
        w_qcdml = torch.tensor(lattice2ndarray_5d(w_gpt))
        return w_qcdml

    return apply
