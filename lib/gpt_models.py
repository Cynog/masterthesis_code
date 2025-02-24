# import required packages
import gpt as g


def get_matrix_operator_from_solver(solver):
    """get matrix operator from a GPT solver"""
    def _mat(dst, src):
        dst @= solver(src)
    return g.matrix_operator(mat=_mat)


def get_layer_ptc(U, paths=None):
    """get a parallel transport convolution layer"""
    # fine grid layer
    grid = U[0].grid
    n_dim = len(U)
    if paths is None:
        paths = [g.path().forward(i) for i in range(n_dim)] + [g.path().backward(i) for i in range(n_dim)]
    ot_i = g.ot_vector_spin_color(4, 3)
    ot_w = g.ot_matrix_spin(4)

    def fine_ptc(n_in, n_out):
        return g.ml.layer.parallel_transport_convolution(grid, U, paths, ot_i, ot_w, n_in, n_out)
    return fine_ptc


def get_ptc1hxl(U, n_layers=1):
    """get PTC one hop multi layer model"""
    # get fine ptc
    layer_fine_ptc = get_layer_ptc(U)

    # ptc_1h1l model
    model = g.ml.model.sequence(*[layer_fine_ptc(1, 1) for _ in range(n_layers)])
    return model


def get_smoother(U):
    """get smoother model"""
    # get fine ptc model
    layer_fine_ptc = get_layer_ptc(U)

    # smoother model
    model = g.ml.model.sequence(layer_fine_ptc(2, 2),
                                layer_fine_ptc(2, 2),
                                layer_fine_ptc(2, 2),
                                layer_fine_ptc(2, 1)
                                )
    return model


def get_matrix_operator_smoother(model, weights, grid, prec=None):
    """get matrix operator from smoother model"""
    null = g.vspincolor(grid)
    null @= null * 0

    def _mat(dst, src):
        if prec is None:
            initial_guess = null
        else:
            initial_guess = prec(src)
        dst @= model(weights, [src, initial_guess])

    return g.matrix_operator(mat=_mat)


def get_wc_coarse(wc, mg_setup_2lvl):
    """get coarse wilson-clover operator"""
    coarse_grid = mg_setup_2lvl[0][0]
    u_bar = mg_setup_2lvl[0][1]
    block_map = g.block.map(coarse_grid, u_bar)
    wc_coarse = block_map.coarse_operator(wc)
    return wc_coarse, coarse_grid


def get_mg_2lvl_vcycle(mg_setup_2lvl, inner_solver_kwargs={"eps": 5e-2, "maxiter": 50, "restartlen": 25, "checkres": False}, smoother_kwargs={"eps": 1e-14, "maxiter": 8, "restartlen": 4, "checkres": False}):
    """get two-level multigrid vcycle solver"""
    # mg inner solvers
    inner_solver = g.algorithms.inverter.fgmres(**inner_solver_kwargs)
    smoother_solver = g.algorithms.inverter.fgmres(**smoother_kwargs)

    # multigrid vcycle
    mg = g.algorithms.inverter.sequence(g.algorithms.inverter.coarse_grid(inner_solver, *mg_setup_2lvl[0]),
                                           # inverter.calculate_residual("before smoother"),  # optional since it costs time but helps to tune MG solver
                                           smoother_solver,
                                           # inverter.calculate_residual("after smoother"),  # optional
                                           )
    return mg


def get_layer_coarse_lptc(mg_setup_2lvl):
    """get coarse local parallel transport convolution layer"""
    # use multigrid setup
    coarse_grid = mg_setup_2lvl[0][0]
    u_bar = mg_setup_2lvl[0][1]

    # identity on coarse grid
    one = g.complex(coarse_grid)
    one[:] = 1

    # local parallel transport convolution
    I = [g.copy(one) for _ in range(4)]
    paths = [g.path().forward(i) for i in range(4)] + [g.path().backward(3)]
    cot_i = g.ot_vector_complex_additive_group(len(u_bar))
    cot_w = g.ot_matrix_complex_additive_group(len(u_bar))

    def coarse_lptc(n_in, n_out):
        return g.ml.layer.local_parallel_transport_convolution(coarse_grid, I, paths, cot_i, cot_w, n_in, n_out)
    return coarse_lptc


def get_coarse_lptc(mg_setup_2lvl):
    """get coarse local parallel transport convolution model"""
    layer_coarse_lptc = get_layer_coarse_lptc(mg_setup_2lvl)

    model = g.ml.model.sequence(layer_coarse_lptc(1, 1))
    return model


def get_model1(U, mg_setup_2lvl):
    """get full model based on the multigrid idea"""
    # fine ptc layer
    layer_fine_ptc = get_layer_ptc(U)

    # coarse lptc layer
    layer_coarse_lptc = get_layer_coarse_lptc(mg_setup_2lvl)

    # restriction and prolongation layers
    coarse_grid = mg_setup_2lvl[0][0]
    u_bar = mg_setup_2lvl[0][1]
    block_map = g.block.map(coarse_grid, u_bar)
    layer_restrict = g.ml.layer.block.project(block_map)
    layer_prolong = g.ml.layer.block.promote(block_map)

    # define model1
    model = g.ml.model.sequence(
        g.ml.layer.parallel(
            g.ml.layer.sequence(),
            g.ml.layer.sequence(
                layer_restrict,
                layer_coarse_lptc(1, 1),
                layer_prolong
            )
        ),
        layer_fine_ptc(2, 2),
        layer_fine_ptc(2, 2),
        layer_fine_ptc(2, 2),
        layer_fine_ptc(2, 1),
    )
    return model
