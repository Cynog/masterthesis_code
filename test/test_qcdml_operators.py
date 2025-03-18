# import required libraries
import gpt as g
import torch
import json


# import local librariers
import sys
sys.path.append("../lib")
from qcdml_operators import qcdml_dw, qcdml_dw_inv, qcdml_pv_dw, qcdml_pv_dw_inv


def test_qcdml_dw_inv():
    """test the function qcdml_operators.qcdml_dw_inv"""
    
    # load domain-wall parameters
    config = json.load(open("../src/dw/config.json"))
    mobius_p = config['mobius_p']
    mobius_p['Ls'] = 8
    mobius_p['mass'] = 0.08
    
    # load gauge field
    U = g.load("../gconfigs/8c16_1200.cfg")
    L5 = [mobius_p['Ls']] + U[0].grid.fdimensions
    
    # define domain-wall operator and its inverse
    D_dw = qcdml_dw(U, mobius_p)
    solver_kwargs = {'eps': 1e-10, 'maxiter': 1000}
    D_dw_inv = qcdml_dw_inv(U, mobius_p, solver_kwargs)
    
    # initialize random number generator
    torch.manual_seed(0)
    
    # random normal source vector
    src = torch.randn(*L5, 4, 3, dtype=torch.cdouble)
    
    # apply operators
    dst = D_dw_inv(D_dw(src))
    
    # assert
    assert torch.allclose(src, dst, atol=1e-5, rtol=1e-5)
    

def test_qcdml_pv_dw_inv():
    """test the function qcdml_operators.qcdml_pv_dw_inv"""
    
    # load domain-wall parameters
    config = json.load(open("../src/dw/config.json"))
    mobius_p = config['mobius_p']
    mobius_p['Ls'] = 8
    mobius_p['mass'] = 0.08
    
    # load gauge field
    U = g.load("../gconfigs/8c16_1200.cfg")
    L5 = [mobius_p['Ls']] + U[0].grid.fdimensions
    
    # define pauli-villars preconditioned domain-wall operator and its inverse
    D_pv_dw = qcdml_pv_dw(U, mobius_p)
    solver_kwargs = {'eps': 1e-10, 'maxiter': 1000}
    D_pv_dw_inv = qcdml_pv_dw_inv(U, mobius_p, solver_kwargs)
    
    # initialize random number generator
    torch.manual_seed(0)
    
    # random normal source vector
    src = torch.randn(*L5, 4, 3, dtype=torch.cdouble)
    
    # apply operators
    dst = D_pv_dw_inv(D_pv_dw(src))
    
    # assert
    assert torch.allclose(src, dst, atol=1e-5, rtol=1e-5)