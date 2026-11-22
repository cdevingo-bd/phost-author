"""Sandboxed worker: load a user's runtime source, build a host, run the stage's contract check, print the verdict.

Reads one JSON request on stdin and prints one result line. Runs inside ``sandbox.run``.
"""

from __future__ import annotations

import json
import sys
import traceback

PREFIX = "PHOST_AUTHOR_RESULT "


def build_host(scenes, scene_id: str, worlds: int):
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    from phost.testing.mjwarp import DrivableMJWarpHost

    scene = scenes.get(scene_id)
    cpu_model = mujoco.MjModel.from_xml_string(scene.xml())
    cpu = mujoco.MjData(cpu_model)
    for _ in range(scene.settle_steps):
        mujoco.mj_step(cpu_model, cpu)
    mujoco.mj_forward(cpu_model, cpu)
    with wp.ScopedDevice("cuda:0"):
        model = mjw.put_model(cpu_model)
        data = mjw.put_data(cpu_model, cpu, nworld=worlds, nconmax=16 * worlds + 64, njmax=2048)
    return DrivableMJWarpHost(model, data, cpu_model=cpu_model, cpu_data=cpu)


def main() -> int:
    request = json.loads(sys.stdin.read())
    try:
        from phost import Impl, ImplKind, Registry, Selection
        from phost.stages import register_all
        from phost.testing.contract import check_impl

        from .modules import load_source
        module = load_source(request["source"], "submission")
        runtime_class = getattr(module, request["entry"], None)
        if runtime_class is None:
            raise LookupError(f"the source defines no {request['entry']!r}")
        registry = Registry()
        register_all(registry)
        stage = request["stage"]
        registry.add_impl(Impl(
            id=f"{stage}/user", stage=stage, stage_version=1, version="user", title="User submission",
            kind=ImplKind.KERNEL, doc=(module.__doc__ or "").strip(),
            factory=lambda host, params: runtime_class(host, params), reference=getattr(module, "reference", None)))
        from phost_lab import default_scenes
        from phost_lab.opportunity import scene_for
        scenes = default_scenes()
        scene_id = request.get("scene") or scene_for(stage, scenes, registry)
        if scene_id is None:
            raise LookupError(f"no scene accepts stage {stage!r} on this host")
        host = build_host(scenes, scene_id, request["worlds"])
        verdict = check_impl(host, Selection(stage, f"{stage}/user", request.get("params", {})), registry)
        print(PREFIX + json.dumps(dict(scene=scene_id, verdict=dict(ok=bool(verdict.ok), stage=verdict.stage, impl=verdict.impl,
                                                      metrics={k: float(v) for k, v in verdict.metrics.items()},
                                                      failures=list(verdict.failures)))), flush=True)
        return 0
    except BaseException as error:
        print(PREFIX + json.dumps(dict(error=f"{type(error).__name__}: {error}",
                                       traceback=traceback.format_exc()[-3000:])), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
