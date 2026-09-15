"""
Reference solutions for every exercise in graph_pc_walkthrough.ipynb.

Graph-structured predictive coding, following Millidge, Tschantz & Buckley (2020),
"Predictive Coding Approximates Backprop along Arbitrary Computation Graphs".

Use it two ways:

    python graph_pc_solutions.py            # run every check, print every number
    python graph_pc_solutions.py --fast     # skip the ~2 min training sweep

    from graph_pc_solutions import infer, pc_weight_grads   # or import one piece

Every printed value below was produced by running this file. If your notebook cell
disagrees with the number here, the notebook cell is the thing to fix.
"""

import time
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

jax.config.update("jax_platform_name", "cpu")


# =============================================================================
# Exercise 1 - the children map
# =============================================================================
# A graph is a list of nodes; nodes[j] = (parents, fn), and thetas[j] is node j's
# own parameter (or None). Node 0 is the input, node N the output.
#
# Equation 2 sums over the CHILDREN of each node, but the graph is stored by
# parents. So invert it -- and record which argument slot the parent occupies,
# because a merge node's two parents are not interchangeable.

def children_of(nodes):
    """children_of(nodes)[i] = [(j, slot), ...]: node j has node i as parent #slot."""
    ch = [[] for _ in nodes]
    for j, (parents, _) in enumerate(nodes):
        for slot, p in enumerate(parents):
            ch[p].append((j, slot))
    return ch


# =============================================================================
# Exercise 2 - the forward pass
# =============================================================================
# Algorithm 1's first loop. Nodes are in topological order, so a single sweep
# upward is enough: every parent is computed before its children.

def forward(nodes, thetas, x):
    """Feedforward pass. Returns v, which at this point equals mu (the predictions)."""
    v = [x] + [None] * (len(nodes) - 1)
    for i in range(1, len(nodes)):
        parents, fn = nodes[i]
        v[i] = fn([v[p] for p in parents], thetas[i])
    return v


def loss_fn(nodes, thetas, x, target):
    v = forward(nodes, thetas, x)
    return 0.5 * jnp.sum((target - v[-1]) ** 2)


# ---- given in the notebook: the autodiff reference and the VJP helper --------

def true_node_grads(nodes, thetas, x, target):
    """dL/dv_i by autodiff, for every node. This is what PC has to reproduce."""
    v_ff = forward(nodes, thetas, x)
    grads = []
    for i in range(len(nodes)):
        def L_from_i(vi, i=i):
            v = list(v_ff)
            v[i] = vi
            for j in range(i + 1, len(nodes)):
                parents, fn = nodes[j]
                v[j] = fn([v[p] for p in parents], thetas[j])
            return 0.5 * jnp.sum((target - v[-1]) ** 2)
        grads.append(jax.grad(L_from_i)(v_ff[i]))
    return grads


def jT_factory(nodes, thetas):
    """jT(j, slot, parent_vals, e) = (d fn_j / d parent[slot])^T @ e."""
    def jT(j, slot, parent_vals, e):
        f = lambda vp: nodes[j][1](
            [vp if k == slot else parent_vals[k] for k in range(len(parent_vals))],
            thetas[j])
        _, vjp = jax.vjp(f, parent_vals[slot])
        return vjp(e)[0]
    return jT


# =============================================================================
# Exercise 3 - Equation 2 for an arbitrary DAG
# =============================================================================
# The three things that make this an exercise rather than a transcription:
#
# 1. THE SUM. One jT term per child. On a chain that loop runs once; on a branch
#    point it runs twice. Accumulate (child_term = child_term + ...), never assign.
#
# 2. THE SIGN. Appendix D (p. 24):
#        dF/dv_i = eps_i - sum_{j in C(i)} J_j^T eps_j
#    and the dynamics are the NEGATIVE of that. Put dF/dv itself in the variable
#    and then subtract it. Storing -dF/dv and also subtracting negates twice and
#    silently gives gradient ASCENT -- no error, plausible numbers for hundreds
#    of steps, then divergence (F reaches ~1e14 by t=800 instead of settling at
#    1.8588). Exercise 3.5 in the notebook exists to catch exactly this.
#
# 3. THE FIXED-PREDICTION FLAG (Sec. 2, p. 5). The paper freezes mu at the
#    feedforward values throughout the relaxation, which is what makes each
#    node's problem local. The flag controls where the Jacobians are evaluated:
#    at the frozen feedforward parents (True) or the current relaxed ones (False).
#    With False you must also recompute mu at the end of each step.

def infer(nodes, thetas, x, target, n_steps, lr, fixed_prediction=True):
    """Relax the free vertices under Eq. 2.

    Returns (v, mu, eps_hist) where eps_hist[t][i] is epsilon_i BEFORE step t,
    so entry 0 is the feedforward state and entry n_steps the final state.
    """
    N  = len(nodes) - 1
    ch = children_of(nodes)
    jT = jT_factory(nodes, thetas)

    v_ff = forward(nodes, thetas, x)
    mu   = list(v_ff)
    v    = list(v_ff)
    v[N] = target                       # clamp the output to the target

    eps_hist = []
    for t in range(n_steps):
        eps = [jnp.zeros_like(v[0])] + [v[i] - mu[i] for i in range(1, N + 1)]
        eps_hist.append([jnp.asarray(e) for e in eps])

        base  = v_ff if fixed_prediction else v
        new_v = list(v)

        for i in range(1, N):           # node 0 and node N are clamped
            child_term = jnp.zeros_like(v[i])
            for (j, slot) in ch[i]:                     # <- one term PER CHILD
                pv = [base[p] for p in nodes[j][0]]
                child_term = child_term + jT(j, slot, pv, eps[j])

            dF_dvi   = eps[i] - child_term              # <- dF/dv itself
            new_v[i] = v[i] - lr * dF_dvi               # <- forward Euler descent

        v = new_v

        if not fixed_prediction:
            for i in range(1, N + 1):
                parents, fn = nodes[i]
                mu[i] = fn([v[p] for p in parents], thetas[i])

    eps = [jnp.zeros_like(v[0])] + [v[i] - mu[i] for i in range(1, N + 1)]
    eps_hist.append([jnp.asarray(e) for e in eps])
    return v, mu, eps_hist


def energy(eps):
    """F = 1/2 sum_i ||eps_i||^2. Must settle; if it grows without bound you are ascending."""
    return 0.5 * sum(float(jnp.sum(e ** 2)) for e in eps[1:])


# =============================================================================
# Exercise 4 - the gradient at the input node
# =============================================================================
# The input node is clamped, so it never acquires an eps of its own. Its gradient
# has to be read off its children: pull each child's eps back through that child's
# Jacobian and sum. The leading minus matches the -eps_i = dL/dv_i convention used
# everywhere else.

def pc_input_grad(nodes, thetas, x, eps, v_ff):
    jT  = jT_factory(nodes, thetas)
    tot = jnp.zeros_like(x)
    for (j, slot) in children_of(nodes)[0]:
        pv  = [v_ff[p] for p in nodes[j][0]]
        tot = tot + jT(j, slot, pv, eps[j])
    return -tot


# =============================================================================
# Exercise 5 - three topologies
# =============================================================================
# A node holds ONE value with ONE formula, so v_i = v_{i-1} + tanh(W_i v_{i-1})
# -- where v_{i-1} appears twice -- cannot be one node if you want the reuse to
# show up in the graph. Splitting it in two is what gives the block input two
# children, and that is the whole point of the exercise.

D = 4


def mk_chain(L, key):
    """v_i = tanh(W_i v_{i-1}). The control."""
    ks = jr.split(key, L)
    nodes, thetas = [([], None)], [None]
    for i in range(L):
        nodes.append(([i], lambda p, th: jnp.tanh(th @ p[0])))
        thetas.append(jr.normal(ks[i], (D, D)) * 0.5)
    return nodes, thetas


def mk_resid_folded(L, key):
    """v_i = tanh(W_i v_{i-1}) + v_{i-1}, skip folded INSIDE the edge function.

    Computes the same function as mk_resid_explicit, but the graph is still a
    chain: the identity path lives inside one Jacobian, so no node gains a
    second child.
    """
    ks = jr.split(key, L)
    nodes, thetas = [([], None)], [None]
    for i in range(L):
        nodes.append(([i], lambda p, th: jnp.tanh(th @ p[0]) + p[0]))
        thetas.append(jr.normal(ks[i], (D, D)) * 0.5)
    return nodes, thetas


def mk_resid_explicit(L, key):
    """a_i = tanh(W_i v_{i-1}); v_i = v_{i-1} + a_i. The ViT form.

    Three bookkeeping decisions, each with a wrong version that still runs:

      * `a = len(nodes) - 1` is read AFTER appending the sublayer, so it names
        that node. Read it before and the merge gets parents [prev, prev] --
        it computes v0 + v0 and the sublayer feeds nothing.

      * `prev` advances to the MERGE, not the sublayer. Point it at the sublayer
        and the graph still reports 3 multi-child nodes, so the check cell passes,
        but the residual stream is never carried forward.

      * the merge's thetas entry is None. `p[0] + p[1]` has no weight in it, and
        pc_weight_grads skips None entries. Give it a matrix and you compute a
        gradient for a parameter the forward pass never uses. It is also what
        makes the node counts match the chain, so the comparison tests topology
        rather than capacity.

    Note that no lambda here mentions the loop variable `i`: closures capture by
    reference, so every lambda would see the final `i`. Only `ks[i]`, evaluated
    immediately, may use it.
    """
    ks = jr.split(key, L)
    nodes, thetas = [([], None)], [None]
    prev = 0                                    # index of this block's input
    for i in range(L):
        nodes.append(([prev], lambda p, th: jnp.tanh(th @ p[0])))
        thetas.append(jr.normal(ks[i], (D, D)) * 0.5)
        a = len(nodes) - 1                      # AFTER the append

        nodes.append(([prev, a], lambda p, th: p[0] + p[1]))
        thetas.append(None)                     # merge has no parameters

        prev = len(nodes) - 1                   # next block hangs off the MERGE
    return nodes, thetas


# =============================================================================
# Exercise 6 - the inference budget
# =============================================================================
# Error starts only at the clamped output and moves one edge per relaxation step.
# So the first step at which node i's eps becomes nonzero is exactly its shortest
# path to the output -- which makes that distance the MINIMUM inference budget T.
# Necessary, not sufficient: reaching a node is not the same as converging there.

def graph_dist_to_output(nodes):
    """Shortest number of edges from each node to the output node."""
    ch, N = children_of(nodes), len(nodes) - 1
    INF = 10 ** 9
    d = [INF] * len(nodes)
    d[N] = 0
    changed = True
    while changed:                          # relax until nothing improves
        changed = False
        for i in range(len(nodes) - 1, -1, -1):
            for (j, _) in ch[i]:
                if d[j] + 1 < d[i]:
                    d[i] = d[j] + 1
                    changed = True
    return d


def front(nodes, thetas, xv, tgt, n_steps=14, lr=0.05, tol=1e-14):
    """First inference step at which each node's error becomes nonzero."""
    _, _, hist = infer(nodes, thetas, xv, tgt, n_steps=n_steps, lr=lr)
    N   = len(nodes) - 1
    out = []
    for i in range(1, N + 1):
        first = None
        for t, eps in enumerate(hist):
            if float(jnp.max(jnp.abs(eps[i]))) > tol:
                first = t
                break
        out.append(first)
    return out


# =============================================================================
# Exercise 7 - weight updates (Equation 3)
# =============================================================================
# Equation 3 is the same VJP as Equation 2, taken with respect to theta_i instead
# of the parent value. Parents are read from the RELAXED v, not from v_ff -- the
# relaxation is what turns local errors into backprop-equivalent gradients.
# Unparameterised nodes (the merges) are skipped.

def pc_weight_grads(nodes, thetas, v, mu, N):
    out = [None] * len(nodes)
    for i in range(1, N + 1):
        if thetas[i] is None:
            continue
        eps_i   = v[i] - mu[i]
        parents, fn = nodes[i]
        pv      = [v[p] for p in parents]
        _, vjp  = jax.vjp(lambda th: fn(pv, th), thetas[i])
        out[i]  = -vjp(eps_i)[0]
    return out


def train(mode, xv, tgt, key, epochs=25, T=100, lr_w=0.05, lr_v=0.05):
    """Train mk_resid_explicit(3) by backprop or by PC. Deliberately un-jitted."""
    nodes, thetas = mk_resid_explicit(3, key)
    losses = []
    for ep in range(epochs):
        losses.append(float(loss_fn(nodes, thetas, xv, tgt)))
        if mode == "bp":
            wg = jax.grad(lambda th: loss_fn(nodes, th, xv, tgt))(thetas)
        else:
            v, mu, _ = infer(nodes, thetas, xv, tgt, n_steps=T, lr=lr_v)
            wg = pc_weight_grads(nodes, thetas, v, mu, len(nodes) - 1)
        thetas = [None if t is None else t - lr_w * wg[i]
                  for i, t in enumerate(thetas)]
    return losses


# =============================================================================
# Checks - every number the notebook asks you to reproduce
# =============================================================================

def _fig2():
    theta_val = jnp.array([2.0])
    nodes = [
        ([],     None),                                          # v0 input
        ([0],    lambda p, th: p[0] * th),                       # v1 = theta*v0
        ([1],    lambda p, th: jnp.sqrt(p[0])),                  # v2 = sqrt(v1)
        ([0],    lambda p, th: p[0] ** 2),                       # v3 = v0^2  <- reuse
        ([2, 3], lambda p, th: jnp.tan(p[0]) + jnp.sin(p[1])),   # v4
    ]
    thetas = [None, theta_val, None, None, None]
    return nodes, thetas, jnp.array([5.0]), jnp.array([1.0])


def main(fast=False):
    nodes, thetas, x, target = _fig2()

    print("=" * 72)
    print("Exercises 1-2  |  children map and forward pass")
    print("=" * 72)
    for i, c in enumerate(children_of(nodes)):
        print(f"  children of node {i}: {c}")
    v_ff = forward(nodes, thetas, x)
    print("  v0..v4:", [round(float(z.ravel()[0]), 6) for z in v_ff])
    print("  loss  :", round(float(loss_fn(nodes, thetas, x, target)), 6))

    tg = true_node_grads(nodes, thetas, x, target)
    print("  backprop dL/dv_i:", [round(float(g.ravel()[0]), 6) for g in tg])

    print()
    print("=" * 72)
    print("Exercise 3.5  |  does the relaxation descend?")
    print("=" * 72)
    _, _, h = infer(nodes, thetas, x, target, n_steps=800, lr=0.02)
    for t in [0, 5, 20, 100, 400, 800]:
        print(f"   t={t:<5} F = {energy(h[t]):.6e}")
    print("  eps_2 settling:", [round(float(h[t][2].ravel()[0]), 6) for t in range(0, 61, 10)])

    print()
    print("=" * 72)
    print("Exercises 3-4  |  PC errors against backprop gradients")
    print("=" * 72)
    for label, fp in [("fixed-prediction (the paper's assumption)", True),
                      ("predictions recomputed each step",          False)]:
        v, mu, hist = infer(nodes, thetas, x, target, n_steps=800, lr=0.02,
                            fixed_prediction=fp)
        eps = hist[-1]
        g0  = pc_input_grad(nodes, thetas, x, eps, v_ff)
        print(f"\n  {label}")
        print("    -eps_i at equilibrium :", [round(-float(e.ravel()[0]), 6) for e in eps[1:]])
        print("    backprop dL/dv_i      :", [round(float(g.ravel()[0]), 6) for g in tg[1:]])
        print(f"    PC grad at branch v0  : {float(g0.ravel()[0]):.6f}"
              f"   backprop: {float(tg[0].ravel()[0]):.6f}")
        print(f"    abs error at v0       : {abs(float(g0.ravel()[0]) - float(tg[0].ravel()[0])):.3e}")

    # ---- vector-valued graphs -------------------------------------------
    key = jr.PRNGKey(0)
    xv  = jr.normal(jr.PRNGKey(1), (D,))
    tgt = jr.normal(jr.PRNGKey(2), (D,))
    GRAPHS = {"chain":          mk_chain(6, key),
              "resid_folded":   mk_resid_folded(6, key),
              "resid_explicit": mk_resid_explicit(3, key)}

    print()
    print("=" * 72)
    print("Exercise 5  |  three topologies")
    print("=" * 72)
    for name, (nd, th) in GRAPHS.items():
        n_multi = sum(1 for cc in children_of(nd) if len(cc) > 1)
        print(f"  {name:16s} {len(nd)-1} non-input nodes, {n_multi} node(s) with >1 child")

    def cos(a, b):
        a, b = a.ravel(), b.ravel()
        return float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-30))

    print("\n  equivalence check (this is your regression test):")
    for name, (nd, th) in GRAPHS.items():
        tgv = true_node_grads(nd, th, xv, tgt)
        _, _, hist = infer(nd, th, xv, tgt, n_steps=400, lr=0.05)
        eps  = hist[-1]
        errs = [float(jnp.max(jnp.abs(-eps[i] - tgv[i]))) for i in range(1, len(nd))]
        coss = [cos(-eps[i], tgv[i]) for i in range(1, len(nd))]
        print(f"  {name:16s} max|-eps - dL/dv| = {max(errs):.2e}   min cosine = {min(coss):.6f}")

    print()
    print("=" * 72)
    print("Exercise 6  |  the front IS the graph distance")
    print("=" * 72)
    for name, (nd, th) in GRAPHS.items():
        d = graph_dist_to_output(nd)[1:]
        f = front(nd, th, xv, tgt)
        print(f"  {name:16s} graph distance to output : {d}")
        print(f"  {'':16s} first-nonzero-eps step   : {f}")
        print(f"  {'':16s} identical: {d == f}")

    print("\n  the floor is necessary, not sufficient:")
    Ts = [1, 2, 3, 5, 10, 20, 50, 100, 200, 400]
    for name in ["chain", "resid_explicit"]:
        nd, th = GRAPHS[name]
        tgv = true_node_grads(nd, th, xv, tgt)
        errs = []
        for T in Ts:
            _, _, hist = infer(nd, th, xv, tgt, n_steps=T, lr=0.05)
            eps = hist[-1]
            errs.append(max(float(jnp.max(jnp.abs(-eps[i] - tgv[i])))
                            for i in range(1, len(nd))))
        print(f"  {name:16s}", ["%.2e" % e for e in errs])

    if fast:
        print("\n(--fast: skipping the training sweep)")
        return

    print()
    print("=" * 72)
    print("Exercise 7  |  weight updates, PC against backprop")
    print("=" * 72)
    t0 = time.time(); bp = train("bp", xv, tgt, key); bp_t = time.time() - t0
    print(f"  backprop     {bp_t:5.1f}s  final loss = {bp[-1]:.6f}")
    for T in [30, 100, 200]:
        t0 = time.time(); pc = train("pc", xv, tgt, key, T=T); dt = time.time() - t0
        gap = max(abs(a - b) for a, b in zip(bp, pc))
        print(f"  PC  T={T:<4}   {dt:5.1f}s  final loss = {pc[-1]:.6f}   "
              f"max|bp - pc| over epochs = {gap:.3e}")


if __name__ == "__main__":
    import sys
    main(fast="--fast" in sys.argv)
