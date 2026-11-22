# phost-author

Write [phost](https://github.com/cdevingo-bd/phost) implementations without a simulator integration. Type an equation and get
a Warp kernel with a float64 reference generated from the same expression tree, or submit a kernel to be checked against the
stage's contract in a sandbox.

## Typed equations

A stage declares how a law for it can be typed (a form, in its contract: `phost.equation`), and `expression_impls.register`
adds a `<stage>/expression` implementation for every stage that does. There is no table of stages in this package.

| Stage | Form | You type |
|---|---|---|
| `dof.passive_force` | one law per DOF | `-a*q - b*v` (names `q v time`) |
| `control.law` | one law per actuator | `a*(b - length) - c*velocity` |
| `body.wrench` | one law per body, six outputs | `fx = -c*vx` ... `tz = ...`; names `x y z wx wy wz vx vy vz time`; a line left out is 0 |
| `probe.state` | one line per measured column | `energy = 0.5*sum(qvel*qvel)`; arrays `qpos qvel qacc`, reductions, `qvel[3]` |
| `effective.solve`, `newton.linear_solve` | a solver step | `init` and `step` lines, `iterations` passes; `rhs`, `diag`, `A @ name`, `dot`, `k`, `n` |

```
phost-lab --plugin phost_author.expression_impls validate examples/typed_spring.json
phost-lab run examples/typed_spring.json --scene trampoline12:impact
```

```json
{"stages": {"dof.passive_force": {"impl": "dof.passive_force/expression",
  "params": {"expr": "-a*q - b*q**3 + c*sin(d*time)", "a": 2.0, "b": 5.0, "c": 0.2, "d": 3.0}}}}
```

The grammar is small on purpose, so an equation is data, not code: numbers, the stage's names, free constants `a`..`h`, `pi`,
`+ - * / **`, and `sin cos tan tanh atan exp log sqrt abs min max clamp where` (`where(q > 0, x, y)` takes the comparison).
Names that are arrays have a type (an axis): entry-by-entry operations need the same axis, `sum mean amax amin maxabs norm dot`
collapse one to a number, `name[3]` picks an entry, and `A @ name` applies a solver's matrix. Anything else is refused with a
message, before any simulation. Lines are separated by newlines or `;`.

How a checked tree becomes code is a **backend** (`phost_author.backends`): the NumPy backend evaluates it in float64 and is
also the reference; the Warp backend generates a kernel from the same tree (compiled once per equation: changing a constant does
not recompile; written for how the host stores each port, so a `(world, body)` array of 6-vectors works as well as a plain
float array); the JAX backend builds a function over `jax.numpy` from it, for a host whose ports are `phost.ports.Ref` cells
(it needs `jax`, and its tests run only where it is installed). A Torch or CUDA-C backend is a class with `pointwise`, `channels` and `iteration` and one
`register_backend` call; it is chosen by the host's `backend` name and touches no stage and no grammar. The stage's contract
check judges the kernel against the text, and ground-truth tests check the motion against MuJoCo (C).

A force law is also differentiated symbolically with respect to `v`, and the result is reported as the law's `damping`
so that a host whose integrator treats joint damping implicitly (MuJoCo's Euler does) treats the typed law the same
way: `-a*v` then lands as close to MuJoCo's own joint damper as MJWarp's native damper does, where a plain force is
about 3e-4 away at 2 ms. The derivative is checked by the contract against central differences of the float64
evaluation. `implicit: false` applies the law as a plain explicit force. At a kink (`abs`, `min`, `max`, `clamp`,
`where`) the derivative is that of the branch taken.

### A typed solver step

`effective.solve` and `newton.linear_solve` take a short program instead of an equation. `init` runs once, `step` runs
`iterations` times, every name keeps its value from one pass to the next, and `x` at the end is the answer:

```
init:  x = 0*rhs ; r = rhs ; z = r/diag ; p = z ; rz = dot(r, z)          # conjugate gradients, Jacobi preconditioner
step:  Ap = A @ p
       alpha = rz / max(dot(p, Ap), 1e-30)
       x = x + alpha*p
       r = r - alpha*Ap
       z = r/diag
       rz_new = dot(r, z)
       p = z + (rz_new / max(rz, 1e-30))*p
       rz = rz_new
```

A damped Jacobi step is the one line `x = x + a*(rhs - A@x)/diag`. The program is **approximate** by declaration: it claims a
relative residual (`tolerance`, default 1e-2) and the check holds it to that claim instead of to the stage's exact gate; the
lab ranks the run on speed and accuracy. The check also reports `max_program_error`, the generated float32 code against the
same program in float64, so a method that is merely slow to converge is told apart from code that does not compute what was
typed. The Warp form runs the whole program in one thread per world: right for any program, fast only for small systems.
It is for trying methods; the lab's opportunity report says what a stage is worth before anyone tunes it.

`expression_impls.preview(registry, stage, text, ...)` checks what was typed and shows what it does without a simulation: for a
pointwise law a curve per output (and a force law's symbolic derivative), for a probe each channel's value on a sample state,
for a solver the relative residual after every pass on a random symmetric positive definite system of a chosen size and
condition number. `describe_form(registry, stage)` says what a person typing for a stage can use. A bad text raises
`ExpressionError` with the message to show.

## Submitting a kernel

`submit.check_submission(Submission(source))` runs the Python source of a user's runtime (see
`src/phost_author/templates/newton_solver_template.py`, a working Newton solver) through the stage's contract check on a
real MJWarp host, inside a sandbox, and returns a report. A wrong, crashing, hanging, input-modifying or
network-seeking submission produces `ok: False` and a reason; it does not raise.

## The sandbox, exactly

Enforced with `bwrap` (refuses to run untrusted code without it): read-only filesystem; empty private home and
`/tmp`; no network; own pid and ipc namespaces; the process tree is killed at the wall-clock limit; CPU-time,
core-dump limits; a limit on the size of any one file and one on the total growth of the scratch directory (polled,
so a fast writer overshoots by a fraction of a second of writing); a scrubbed environment; one writable scratch
directory, seeded with a copy of the trusted Warp kernel cache. The per-file limit cannot be set below
`MIN_COMPILE_FILE_MB` (128): Warp's compiler needs to create files that large, and under it no kernel compiles. Measured
on Warp 1.16: 64 MB fails a one-line kernel, 128 MB passes it and the template Newton solver. Larger kernels were not
measured, so the default is 1 GB.

Source that defines kernels is loaded with `phost_author.modules.load_source`: Warp reads a kernel's source from a
file, so source exec'd from a string cannot define one. The source is written to a private temporary file and imported
under a name made from its hash.

**Not enforced:** the GPU is shared (a submission can hold GPU memory or fault the device); memory is not limited;
and the verdict is computed in the same process as the submission, so a submission written to tamper with the
checker can forge a pass. The sandbox protects the machine. A competition needs the judge in a separate process from
the code it judges; that belongs to the scoring layer, not here. Reports say so (`integrity`).

MIT licensed.
