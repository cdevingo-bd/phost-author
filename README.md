# phost-author

Author [phost](../phost) stages without writing a simulator integration: type an equation, or submit a kernel.

## Typed equations

`dof.passive_force/expression` and `control.law/expression` take the law as text.

```
phost-lab --plugin phost_author.expression_impls validate examples/typed_spring.json
phost-lab run examples/typed_spring.json --scene trampoline12:impact
```

```json
{"stages": {"dof.passive_force": {"impl": "dof.passive_force/expression",
  "params": {"expr": "-a*q - b*q**3 + c*sin(d*time)", "a": 2.0, "b": 5.0, "c": 0.2, "d": 3.0}}}}
```

The grammar is small on purpose, so an equation is data, not code: numbers, the stage's names (`q`, `v`, `time` for
the force law; `length`, `velocity`, `time` for the controller), free constants `a`..`h`, `pi`, `+ - * / **`, and
`sin cos tan tanh atan exp log sqrt abs min max clamp where` (`where(q > 0, x, y)` takes the comparison). Anything
else is refused with a message, before any simulation. The float64 reference evaluates the checked tree; the Warp
kernel is generated from the same tree and compiled once per equation (changing a constant does not recompile).
The stage's contract check judges the kernel against the text, and a ground-truth test checks the motion against
MuJoCo (C) integrating the same equation.

## Submitting a kernel

`submit.check_submission(Submission(source))` runs the Python source of a user's runtime (see
`src/phost_author/templates/newton_solver_template.py`, a working Newton solver) through the stage's contract check on a
real MJWarp host, inside a sandbox, and returns a report. A wrong, crashing, hanging, input-modifying or
network-seeking submission produces `ok: False` and a reason; it does not raise.

## The sandbox, exactly

Enforced with `bwrap` (refuses to run untrusted code without it): read-only filesystem; empty private home and
`/tmp`; no network; own pid and ipc namespaces; the process tree is killed at the wall-clock limit; CPU-time,
file-size (1 GB: NVRTC maps large files when compiling) and core-dump limits; a scrubbed environment; one writable
scratch directory, seeded with a copy of the trusted Warp kernel cache.

**Not enforced:** the GPU is shared (a submission can hold GPU memory or fault the device); memory is not limited;
and the verdict is computed in the same process as the submission, so a submission written to tamper with the
checker can forge a pass. The sandbox protects the machine. A competition needs the judge in a separate process from
the code it judges; that belongs to the scoring layer, not here. Reports say so (`integrity`).

MIT licensed.
