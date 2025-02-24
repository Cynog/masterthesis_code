# import required packages
import gpt as g
import numpy as np
import torch


def lattice2ndarray_5d(lattice):
    """convert a 5d GPT lattice to a numpy ndarray"""
    shape = lattice.grid.fdimensions
    shape = list(reversed(shape))
    if lattice[:].shape[1:] != (1,):
        shape.extend(lattice[:].shape[1:])

    result = lattice[:].reshape(shape)
    result = np.swapaxes(result, 0, 4)
    result = np.swapaxes(result, 1, 3)

    return result


def ndarray2lattice_5d(ndarray, grid, lat_constructor):
    """convert a 5d numpy ndarray to a 5d lattice"""
    lat = lat_constructor(grid)
    data = np.swapaxes(ndarray, 0, 4)
    data = np.swapaxes(data, 1, 3)
    lat[:] = data.reshape([data.shape[0] * data.shape[1] * data.shape[2] * data.shape[3] * data.shape[4]] + list(data.shape[5:]))

    return lat


def get_U5(U_4d, grid5):
    """get GPT 5d domain-wall gauge field from the basic GPT 4d gauge field"""
    # domain wall compatible U
    U5 = g.qcd.gauge.unit(grid5)
    for i in range(4):
        for s in range(grid5.fdimensions[0]):
            U5[1 + i][s, :, :, :, :] = U_4d[i][:, :, :, :]

    return U5


def gptU4d_2_qcdmlU5d(U_gpt_4d, grid5):
    """get qcd_ml 5d domain-wall gauge field from the GPT 4d gauge field"""
    U_gpt_5d = get_U5(U_gpt_4d, grid5)
    U_qcdml_5d = [torch.tensor(lattice2ndarray_5d(u_gpt)) for u_gpt in U_gpt_5d]
    return U_qcdml_5d
