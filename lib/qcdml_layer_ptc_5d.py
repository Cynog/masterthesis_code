"""PTC and SPTC layers copied from qcd_ml.nn.ptc and qcd_ml.nn.sptc and modified for 5d"""


import torch
from qcd_ml.base.paths.simple_paths import v_ng_evaluate_path


def _es_SU3_group_compose_5d(A, B):
    return torch.einsum("sabcdij,sabcdjk->sabcdik", A, B)  # ! 5d change


def _es_v_gauge_transform_5d(Umu, v):
    return torch.einsum("sabcdij,sabcdSj->sabcdSi", Umu, v)  # ! 5d change


def compile_path_5d(path):
    """
    Compiles a path, such that few rolls are necessary to
    v_ng_evaluate_path. 

    XXX: Do not use when a gauge field is present!
    """
    shifts = [0] * 5  # ! 5d change

    for mu, nhops in path:
        shifts[mu] += nhops

    return [(mu, nhops) for mu, nhops in enumerate(shifts) if nhops != 0]


class PathBuffer_5d:
    """
    This class brings the same functionality as v_evaluate_path and
    v_reverse_evaluate_path but pre-computes the costly gauge transport matrix
    multiplications.
    """

    def __init__(self, U, path, gauge_group_compose=_es_SU3_group_compose_5d, gauge_transform=_es_v_gauge_transform_5d, adjoin=lambda x: x.adjoint(), gauge_identity=torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.cdouble)):
        if isinstance(U, list):
            # required by torch.roll below.
            U = torch.stack(U)
        self.path = path

        self.gauge_group_compose = gauge_group_compose
        self.gauge_transform = gauge_transform
        self.adjoin = adjoin

        if len(self.path) == 0:
            # save computational cost and memory.
            self._is_identity = True
        else:
            self._is_identity = False

            self.accumulated_U = torch.zeros_like(U[0])
            self.accumulated_U[:, :, :, :, :] = torch.clone(gauge_identity)  # ! 5d change

            for mu, nhops in self.path:
                if nhops < 0:
                    direction = -1
                    nhops *= -1
                else:
                    direction = 1

                for _ in range(nhops):
                    if direction == -1:
                        U = torch.roll(U, 1, mu + 1)  # mu + 1 because U is (mu, x, y, z, t)
                        self.accumulated_U = self.gauge_group_compose(U[mu], self.accumulated_U)
                    else:
                        self.accumulated_U = self.gauge_group_compose(self.adjoin(U[mu]), self.accumulated_U)
                        U = torch.roll(U, -1, mu + 1)

            self.path = compile_path_5d(self.path)  # ! 5d change

    def v_transport(self, v):
        """
        Gauge-equivariantly transport the vector-like field ``v`` along the path.
        """
        if not self._is_identity:
            v = self.gauge_transform(self.accumulated_U, v)
            v = v_ng_evaluate_path(self.path, v)
        return v


def v_spin_const_transform_5d(M, v):
    """
    Applies a spin matrix to a vector field.
    """
    return torch.einsum("ij,sabcdjG->sabcdiG", M, v)  # ! 5d change


class v_PTC_5d(torch.nn.Module):
    """
    Parallel Transport Convolution for objects that 
    transform vector-like.

    Weights are stored as [feature_in, feature_out, path].

    paths is a list of paths. Every path is a list [(direction, nhops)].
    An empty list is the path that does not perform any hops.

    For a 1-hop 1-layer model, construct the layer as such::

        U = torch.tensor(np.load("path/to/gauge/config.npy"))

        paths = [[]] + [[(mu, 1)] for mu in range(4)] + [[(mu, -1)] for mu in range(4)]
        layer = v_PTC(1, 1, paths, U)

    """

    def __init__(self, n_feature_in, n_feature_out, paths, U, Wscale=1, **path_buffer_kwargs):
        super().__init__()
        weights = Wscale * torch.randn(n_feature_in, n_feature_out, len(paths), 4, 4, dtype=torch.cdouble)
        weights[:, :, 0] += torch.eye(4, dtype=torch.cdouble)  # ! init to identity
        self.weights = torch.nn.Parameter(weights)

        self.n_feature_in = n_feature_in
        self.n_feature_out = n_feature_out
        self.path_buffer_kwargs = path_buffer_kwargs
        # FIXME: This is more memory intensive compared to the
        # implementation using v_evaluate_path, because instead of one
        # copy of U, all gauge transport matrices are stored.
        # On the other hand this may not be a big deal in most cases,
        # because, for 1h, the number of gauge fields is identical.
        self.path_buffers = [PathBuffer_5d(U, pi, **path_buffer_kwargs) for pi in paths]  # ! 5d change

    def forward(self, features_in):
        if features_in.shape[0] != self.n_feature_in:
            raise ValueError(f"shape mismatch: got {features_in.shape[0]} but expected {self.n_feature_in}")

        features_out = [torch.zeros_like(features_in[0]) for _ in range(self.n_feature_out)]

        for fi, wfi in zip(features_in, self.weights):
            for io, wfo in enumerate(wfi):
                for pi, wi in zip(self.path_buffers, wfo):
                    features_out[io] = features_out[io] + v_spin_const_transform_5d(wi, pi.v_transport(fi))  # ! 5d change

        return torch.stack(features_out)


def _es_v_spin_transform_5d(M, v):
    return torch.einsum("sij,sabcdjG->sabcdiG", M, v)  # ! 5d change and change


class v_sPTC_5d(torch.nn.Module):
    """
    Local Parallel Transport Convolution for objects that 
    transform vector-like.

    Weights are stored as [feature_in, feature_out, path].

    paths is a list of paths. Every path is a list [(direction, nhops)].
    An empty list is the path that does not perform any hops.
    """

    def __init__(self, n_feature_in, n_feature_out, paths, U, Wscale=1, **path_buffer_kwargs):
        super().__init__()
        weights = Wscale * torch.randn(n_feature_in, n_feature_out, len(paths), U[0].shape[0], 4, 4, dtype=torch.cdouble)  # ! change
        weights[:, :, 0, :] += torch.eye(4, dtype=torch.cdouble)  # ! init to identity
        self.weights = torch.nn.Parameter(weights)

        self.n_feature_in = n_feature_in
        self.n_feature_out = n_feature_out
        self.path_buffer_kwargs = path_buffer_kwargs
        self.path_buffers = [PathBuffer_5d(U, pi, **path_buffer_kwargs) for pi in paths]  # ! 5d change

    def forward(self, features_in):
        if features_in.shape[0] != self.n_feature_in:
            raise ValueError(f"shape mismatch: got {features_in.shape[0]} but expected {self.n_feature_in}")

        features_out = [torch.zeros_like(features_in[0]) for _ in range(self.n_feature_out)]

        for fi, wfi in zip(features_in, self.weights):
            for io, wfo in enumerate(wfi):
                for pi, wi in zip(self.path_buffers, wfo):
                    features_out[io] = features_out[io] + _es_v_spin_transform_5d(wi, pi.v_transport(fi))  # ! 5d change and change

        return torch.stack(features_out)
