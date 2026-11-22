"""Template for a newton.linear_solve implementation: dense Cholesky, one thread per world.

Correct and simple, not fast. Replace the kernel with your own algorithm; keep the interface.

Contract (see phost.stages.newton_linear): for every active world solve H x = g, where H is rebuilt from the UPPER
triangle of ``h`` (the lower triangle is undefined), write search = -x, search_dot = |x|^2, newton_decrement = g.x,
leave inactive worlds untouched, and do not modify ``h`` or ``grad``.
"""

import warp as wp


@wp.kernel
def cholesky_solve(h: wp.array3d[float], grad: wp.array2d[float], active: wp.array[int], nv: int,
                   factor: wp.array3d[float], y: wp.array2d[float],
                   search: wp.array2d[float], search_dot: wp.array[float], newton_decrement: wp.array[float]):
    w = wp.tid()
    if active[w] == 0:
        return
    # factor H = L L^T. Only the upper triangle of h is defined, so H[i, j] with i > j is read as h[j, i].
    for j in range(nv):
        s = h[w, j, j]
        for k in range(j):
            s -= factor[w, j, k] * factor[w, j, k]
        d = wp.sqrt(s)
        factor[w, j, j] = d
        for i in range(j + 1, nv):
            t = h[w, j, i]
            for k in range(j):
                t -= factor[w, i, k] * factor[w, j, k]
            factor[w, i, j] = t / d
    # forward substitution L y = g
    for i in range(nv):
        t = grad[w, i]
        for k in range(i):
            t -= factor[w, i, k] * y[w, k]
        y[w, i] = t / factor[w, i, i]
    # back substitution L^T x = y, in place: y becomes x
    for r in range(nv):
        i = nv - 1 - r
        t = y[w, i]
        for k in range(i + 1, nv):
            t -= factor[w, k, i] * y[w, k]
        y[w, i] = t / factor[w, i, i]
    dot = float(0.0)
    dec = float(0.0)
    for i in range(nv):
        x = y[w, i]
        search[w, i] = -x
        dot += x * x
        dec += grad[w, i] * x
    search_dot[w] = dot
    newton_decrement[w] = dec


class Runtime:
    def __init__(self, host, params):
        self.nv = int(host.model.nv)
        worlds = int(host.data.nworld)
        device = host.device
        self.factor = wp.zeros((worlds, self.nv, self.nv), dtype=float, device=device)
        self.y = wp.zeros((worlds, self.nv), dtype=float, device=device)

    def solve(self, system):
        wp.launch(cholesky_solve, dim=system.active.shape[0],
                  inputs=[system.h, system.grad, system.active, self.nv, self.factor, self.y],
                  outputs=[system.search, system.search_dot, system.newton_decrement], device=system.h.device)

    def metrics(self):
        return {"status": 0, "fallbacks": 0}
