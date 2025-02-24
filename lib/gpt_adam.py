import gpt as g


class ADAM_State:
    def __init__(self, m, v):
        self.m = m
        self.v = v


def adam(model, Winit, gpt_dW, state=None, eps=1e-8, alpha=1e-3, beta1=0.9, beta2=0.999, maxiter=10_000, eps_regulator=.1):
    r"""
    Daniel Knüttel 2024

    ``model`` must have the method ``gradient(W, gpt_dW)``::

            model.gradient(W, W_which_weights_to_differentate)

    which computes

    .. math::

        \partial_W model(W, vin, b)

    @misc{kingma2017adam,
          title={Adam: A Method for Stochastic Optimization}, 
          author={Diederik P. Kingma and Jimmy Ba},
          year={2017},
          eprint={1412.6980},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }
    """
    if state is None:
        m = [g(0 * g.copy(w)) for w in Winit]
        v = [g(0 * g.copy(w)) for w in Winit]
    else:
        m = state.m
        v = state.v

    eps_field = [g.copy(w) for w in Winit]
    for efi in eps_field:
        efi[:] = eps_regulator
    W = Winit

    for t in range(1, maxiter + 1):
        grad = model.gradient(W, gpt_dW)
        grad2 = []
        for gi in grad:
            gir = g(g.component.real(gi))
            gii = g(g.component.imag(gi))
            grad2.append(g(g.component.multiply(gir, gir) + 1j * g.component.multiply(gii, gii)))

        gradnorm = sum(g.norm2(dw) for dw in grad)
        # if t % 5 == 0:
        g.message(f"ADAM: {t:5d}: gnorm: {gradnorm:.3e}")
        if (gradnorm < eps):
            return W, (True, t, ADAM_State(m, v))

        m = [g(g(beta1 * mi) + g((1 - beta1) * gi)) for mi, gi in zip(m, grad)]
        v = [g(g(beta2 * vi) + g((1 - beta2) * gi2)) for vi, gi2 in zip(v, grad2)]
        mhat = [g(mi / (1 - beta1**t)) for mi in m]
        vhat = [g(vi / (1 - beta2**t)) for vi in v]

        for (w, e), (mi, vi) in zip(zip(gpt_dW, eps_field), zip(mhat, vhat)):
            w @= w - g.component.multiply(g(alpha * mi), g.component.inv(g(g.component.sqrt(vi) + e)))

    return W, (False, t, ADAM_State(m, v))
