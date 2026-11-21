"""A small actuated model, a way to run it on MJWarp with stages installed, and MuJoCo (C) as the truth."""

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from phost import Session
from phost.testing.mjwarp import DrivableMJWarpHost

XML = """<mujoco><option timestep="0.002" integrator="Euler" solver="Newton"/>
<worldbody>
 <geom type="plane" size="3 3 .1"/>
 <body pos="0 0 2"><freejoint/><geom type="box" size=".05 .05 .05" mass=".5"/></body>
 <body pos="0.5 0 1.0"><joint name="h1" type="hinge" axis="0 1 0"/>
   <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.3"/>
   <body pos="0.3 0 0"><joint name="h2" type="hinge" axis="0 1 0"/>
     <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.2"/></body></body>
 <body pos="-0.5 0 0.5"><joint name="s1" type="slide" axis="0 0 1"/><geom type="sphere" size=".05" mass=".2"/></body>
</worldbody>
<actuator><motor joint="h1" gear="2"/><position joint="h2" kp="5"/></actuator></mujoco>"""
QPOS = [7, 8, 9]                         # h1, h2, s1 (the free joint holds 0..6)
DOFS = [6, 7, 8]


def make_host(nworld=3):
    cm = mujoco.MjModel.from_xml_string(XML)
    cd = mujoco.MjData(cm)
    mujoco.mj_forward(cm, cd)
    with wp.ScopedDevice("cuda:0"):
        model = mjw.put_model(cm)
        data = mjw.put_data(cm, cd, nworld=nworld, nconmax=16 * nworld, njmax=128)
    return DrivableMJWarpHost(model, data, cpu_model=cm, cpu_data=cd), cm, cd


def run_gpu(nsteps, selections=(), registry=None, nworld=2):
    host, cm, cd = make_host(nworld)
    with wp.ScopedDevice("cuda:0"):
        if selections:
            with Session(host, selections, registry):
                for _ in range(nsteps):
                    mjw.step(host.model, host.data)
        else:
            for _ in range(nsteps):
                mjw.step(host.model, host.data)
        wp.synchronize()
        return host.data.qpos.numpy()[0].astype(np.float64)


def cpu_truth(nsteps, force=None):
    """MuJoCo (C, float64). ``force(qpos, qvel, time) -> array over DOFS`` is applied explicitly before each step."""
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    for _ in range(nsteps):
        if force is not None:
            data.qfrc_applied[DOFS] = force(data.qpos[QPOS], data.qvel[DOFS], data.time)
        mujoco.mj_step(model, data)
    return data.qpos.copy()


def error(a, b):
    return float(np.abs(a[QPOS] - b[QPOS]).max())
