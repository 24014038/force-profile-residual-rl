"""
Peg-in-hole 환경 (robosuite 1.5.2 / MuJoCo 3.x).

- Franka Panda 그리퍼에 정사각 막대(peg)를 고정한다 (eef 바디의 자식 바디).
  손목 F/T 센서(ft_frame)는 그리퍼 루트에 있으므로 막대 접촉력이 그대로 측정된다.
- 테이블 위가 아니라 공중에 고정된 구멍 블록(hole_block)을 둔다.
  구멍은 박스 벽 + 45도 모따기(삼각기둥 메쉬)로 만든다.
- 막대와 구멍은 서로만 충돌한다 (contype = conaffinity = 2).
- reset 때마다 명목 구멍 = 막대 끝 바로 아래(start_height 위), 실제 구멍 = 명목 + offset.

길이 단위: 생성자 인자는 mm, 내부 계산은 m.
"""

from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string, new_body, new_geom

import xml.etree.ElementTree as ET

import mujoco

MM = 1e-3


class PegHoleEnv(ManipulationEnv):
    def __init__(
        self,
        robots="Panda",
        controller_configs=None,
        clearance_mm=2.0,        # 한쪽 틈 δ
        chamfer_mm=2.0,          # 모따기 폭 c (45도)
        peg_half_mm=10.0,        # 막대 반폭 (20 x 20 mm)
        peg_len_mm=60.0,
        hole_depth_mm=30.0,
        start_height_mm=10.0,    # 시작 시 막대 끝과 구멍 윗면 사이 거리
        friction=0.4,
        success_depth_mm=20.0,
        offset_mm=(0.0, 0.0),    # 구멍 위치 오차 (x, y)
        k_trans=1000.0,          # 병진 강성 (N/m), x·y·z 동일
        k_rot=50.0,              # 회전 강성 (N·m/rad)
        damping_ratio=1.0,
        resist="none",           # 구멍 안 저항: "none" | "const" | "snap"
        insert_resist_N=0.0,     # const: 일정 저항 (N)
        resist_from_mm=2.0,      # 저항 시작 깊이
        base_resist_N=3.0,       # snap: 기본 마찰 (N, 움직일 때)
        snap_peak_N=12.0,        # snap: 걸쇠 반발력 정점 (N)
        snap_mm=(15.0, 17.5),    # snap: (반발 시작, 정점) 깊이. 정점을 넘으면 반발력 0
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera="frontview",
        control_freq=20,
        horizon=200,
        ignore_done=True,
        **kwargs,
    ):
        self.clearance = clearance_mm * MM
        if resist == "none" and insert_resist_N > 0:
            resist = "const"
        self.resist = resist
        self.insert_resist = insert_resist_N
        self.resist_from = resist_from_mm * MM
        self.base_resist = base_resist_N
        self.snap_peak = snap_peak_N
        self.snap_mm = tuple(snap_mm)
        self.chamfer = chamfer_mm * MM
        self.peg_half = peg_half_mm * MM
        self.peg_len = peg_len_mm * MM
        self.hole_depth = hole_depth_mm * MM
        self.start_height = start_height_mm * MM
        self.friction = friction
        self.success_depth = success_depth_mm * MM
        self.offset = np.array([offset_mm[0], offset_mm[1]], dtype=float) * MM

        # 카테시안 임피던스 제어기 (막대 끝 기준). 세 조건 모두 같은 값 사용.
        self.k_trans = k_trans
        self.k_rot = k_rot
        self.d_trans = 2.0 * damping_ratio * np.sqrt(k_trans * 2.0)   # 유효 질량 2 kg 가정
        self.d_rot = 2.0 * damping_ratio * np.sqrt(k_rot * 0.02)      # 유효 관성 0.02 kg·m² 가정
        self.k_null, self.d_null = 10.0, 2.0                          # 자세 유지(널스페이스)
        self.ff_force = np.zeros(3)      # 힘 피드포워드 (world, N). P+C/P+F에서 z만 사용
        self.tip_target = None
        self._last_torque = None

        # 막대: eef 프레임 z축(=그리퍼 진행 방향, 아래쪽)으로 [-20, +40] mm 구간
        self.peg_center_z = 0.010
        self.peg_tip_local = np.array([0.0, 0.0, self.peg_center_z + self.peg_len / 2])

        self.table_full_size = (0.8, 0.8, 0.05)
        self.table_offset = np.array((0, 0, 0.8))

        self._force_bias = np.zeros(3)
        self._torque_bias = np.zeros(3)
        self.nominal_hole_top = np.zeros(3)
        self.hole_top = np.zeros(3)

        super().__init__(
            robots=robots,
            env_configuration="default",
            controller_configs=controller_configs,
            base_types="default",
            gripper_types="default",
            initialization_noise=None,
            use_camera_obs=False,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=False,
            **kwargs,
        )

    # ------------------------------------------------------------------ model
    def _contact_kwargs(self):
        return dict(
            contype="2",
            conaffinity="2",
            condim="3",
            friction=f"{self.friction} 0.005 0.0001",
            solref="0.004 1",
            solimp="0.95 0.99 0.001",
        )

    def _build_hole(self, model):
        a = self.peg_half + self.clearance       # 구멍 반폭
        c = self.chamfer
        D = self.hole_depth
        W = 0.03                                 # 벽 두께
        T = 0.01                                 # 바닥판 두께
        outer = a + c + W
        ck = self._contact_kwargs()
        rgba = "0.55 0.6 0.7 1"
        rgba_front = "0.55 0.6 0.7 0.25"   # 앞(-y) 벽은 반투명: side 카메라로 안이 보이게

        body = new_body(name="hole_block", pos=(0.0, 0.0, 1.0))  # reset에서 위치 재설정

        # 바닥판
        body.append(new_geom(name="hole_floor", type="box", size=(outer, outer, T / 2),
                             pos=(0, 0, -D - T / 2), rgba=rgba, group=1, **ck))

        # 하부 벽 (z: -D ~ -c, 안쪽 면 = a), 상부 벽 (z: -c ~ 0, 안쪽 면 = a + c)
        for name, inner, z0, z1 in (("low", a, -D, -c), ("up", a + c, -c, 0.0)):
            h = (z1 - z0) / 2
            zc = (z1 + z0) / 2
            th = (outer - inner) / 2
            mid = inner + th
            if h <= 0:
                continue
            body.append(new_geom(name=f"hole_{name}_px", type="box", size=(th, outer, h),
                                 pos=(mid, 0, zc), rgba=rgba, group=1, **ck))
            body.append(new_geom(name=f"hole_{name}_nx", type="box", size=(th, outer, h),
                                 pos=(-mid, 0, zc), rgba=rgba, group=1, **ck))
            body.append(new_geom(name=f"hole_{name}_py", type="box", size=(outer, th, h),
                                 pos=(0, mid, zc), rgba=rgba, group=1, **ck))
            body.append(new_geom(name=f"hole_{name}_ny", type="box", size=(outer, th, h),
                                 pos=(0, -mid, zc), rgba=rgba_front, group=1, **ck))

        # 모따기: 단면 삼각형 (a, -c), (a+c, -c), (a+c, 0) 을 갖는 삼각기둥 (볼록 → 메쉬 가능)
        if c > 0:
            L = a + c
            asset = model.asset
            for name, verts in self._chamfer_meshes(a, c, L).items():
                mesh = ET.Element("mesh", name=name,
                                  vertex=" ".join(array_to_string(v) for v in verts))
                asset.append(mesh)
                g = ET.Element("geom", name=f"{name}_g", type="mesh", mesh=name,
                               rgba="0.85 0.55 0.3 0.25" if name.endswith("ny") else "0.85 0.55 0.3 1",
                               group="1", **ck)
                body.append(g)

        # 걸쇠 위치 표시 (보이기만 하고 충돌 없음, 물리에 영향 없음). 걸쇠 힘은 _pre_action에서 따로 가함
        if self.resist == "snap":
            z0, z1 = -self.snap_mm[1] * MM, -self.snap_mm[0] * MM
            tab = 0.2 * MM
            # 앞(-y)·뒤(+y) 벽 안쪽 면에 주황 띠. 앞쪽은 반투명이라 막대 위에 겹쳐 보인다
            for sgn, nm, alpha in ((1, "py", "1"), (-1, "ny", "0.45")):
                body.append(new_geom(name=f"snap_mark_{nm}", type="box",
                                     size=(a, tab / 2, (z1 - z0) / 2),
                                     pos=(0, sgn * (a - tab / 2), (z0 + z1) / 2),
                                     rgba=f"0.95 0.45 0.1 {alpha}", group=1,
                                     contype="0", conaffinity="0"))

        # 구멍을 비스듬히 내려다보는 카메라 (closeup), 단면처럼 옆에서 보는 카메라 (side)
        body.append(ET.Element("camera", name="closeup", mode="fixed",
                               pos="0.09 -0.09 0.07", xyaxes="0.707 0.707 0 -0.38 0.38 0.84"))
        body.append(ET.Element("camera", name="side", mode="fixed",
                               pos="0 -0.16 0.0", xyaxes="1 0 0 0 0 1"))
        model.worldbody.append(body)
        self._outer = outer

    @staticmethod
    def _chamfer_meshes(a, c, L):
        tri = [(a, -c), (a + c, -c), (a + c, 0.0)]       # (안쪽 방향 좌표, z)
        meshes = {}
        for side in ("px", "nx", "py", "ny"):
            verts = []
            for s in (-L, L):
                for (u, z) in tri:
                    if side == "px":
                        verts.append((u, s, z))
                    elif side == "nx":
                        verts.append((-u, s, z))
                    elif side == "py":
                        verts.append((s, u, z))
                    else:
                        verts.append((s, -u, z))
            meshes[f"chamfer_{side}"] = verts
        return meshes

    def _attach_peg(self, model):
        # 이름으로 eef 바디 찾기 (예: gripper0_right_eef)
        eef = None
        for b in model.worldbody.iter("body"):
            n = b.get("name", "")
            if n.startswith("gripper0") and n.endswith("eef"):
                eef = b
                break
        if eef is None:
            raise RuntimeError("gripper eef body not found")
        self._eef_body_name = eef.get("name")

        peg = new_body(name="peg", pos=(0, 0, self.peg_center_z))
        inertial = ET.Element("inertial", pos="0 0 0", mass="0.05",
                              diaginertia="1e-5 1e-5 1e-5")
        peg.append(inertial)
        peg.append(new_geom(name="peg_g", type="box",
                            size=(self.peg_half, self.peg_half, self.peg_len / 2),
                            rgba="0.2 0.5 0.9 1", group=1, **self._contact_kwargs()))
        peg.append(ET.Element("site", name="peg_tip", pos=f"0 0 {self.peg_len / 2}",
                              size="0.002", rgba="1 0 0 1", type="sphere", group="1"))
        eef.append(peg)

    def _load_model(self):
        super()._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        arena = TableArena(table_full_size=self.table_full_size, table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[r.robot_model for r in self.robots],
        )
        self._attach_peg(self.model)
        self._build_hole(self.model)

    # --------------------------------------------------------- references
    def _setup_references(self):
        super()._setup_references()
        m = self.sim.model
        self.hole_body_id = m.body_name2id("hole_block")
        self.peg_tip_site_id = m.site_name2id("peg_tip")
        self.peg_geom_id = m.geom_name2id("peg_g")
        self.peg_body_id = m.body_name2id("peg")
        self.ft_site_id = m.site_name2id("gripper0_right_ft_frame")
        names = [m.actuator_id2name(i) for i in range(m.nu)]
        self._grip_act = [i for i, n in enumerate(names) if "finger_joint" in n]
        self._grip_ctrl = np.array([0.01 if "joint1" in names[i] else -0.01 for i in self._grip_act])

    def _setup_observables(self):
        return OrderedDict(super()._setup_observables())

    # -------------------------------------------------------------- reset
    def _reset_internal(self):
        super()._reset_internal()
        self.sim.forward()
        tip = self.peg_tip_pos()
        self.nominal_hole_top = tip - np.array([0.0, 0.0, self.start_height])
        # 막대 끝 ↔ 컨트롤러 기준점(eef site) 사이 오프셋, 초기 자세 (절대 목표 계산용)
        self._settle_vertical()
        tip = self.peg_tip_pos()
        self.nominal_hole_top = tip - np.array([0.0, 0.0, self.start_height])
        self.tip_target = tip.copy()
        self.ff_force = np.zeros(3)
        self.set_offset(self.offset)

    def _settle_vertical(self):
        """막대를 수직으로 세운 초기 자세. 처음 한 번만 제어기로 맞추고, 이후엔 그 관절값을 바로 넣는다."""
        robot = self.robots[0]
        qi, vi = robot._ref_arm_joint_pos_indexes, robot._ref_arm_joint_vel_indexes
        d = self.sim.data._data
        if getattr(self, "_q_settled", None) is None:
            R = d.site_xmat[self.peg_tip_site_id].reshape(3, 3)
            z = np.array([0.0, 0.0, -1.0])                      # 막대 축 = 아래 방향
            x = R[:, 0] - np.dot(R[:, 0], z) * z
            x /= np.linalg.norm(x)
            self._rot_target = np.column_stack([x, np.cross(z, x), z])
            self._q0 = d.qpos[qi].copy()
            self.tip_target = self.peg_tip_pos()
            self.ff_force = np.zeros(3)
            for _ in range(1500):                                # 3 s (0.002 s 스텝)
                self.sim.forward()
                self._pre_action(self.tip_target, policy_step=False)
                self.sim.step()
            self._q_settled = d.qpos[qi].copy()
            self._q0 = self._q_settled.copy()
            # 정지 상태(제어기가 자세를 잡고 있는 상태)의 손목 힘 = 그리퍼 + 막대 무게 → 바이어스
            self._force_bias = self.robots[0].ee_force["right"].copy()
            self._torque_bias = self.robots[0].ee_torque["right"].copy()
        d.qpos[qi] = self._q_settled
        d.qvel[vi] = 0.0
        self.sim.forward()

    def set_offset(self, offset_m):
        """구멍 실제 위치 = 명목 + offset (m). reset 직후 호출."""
        self.offset = np.asarray(offset_m, dtype=float)
        self.hole_top = self.nominal_hole_top + np.array([self.offset[0], self.offset[1], 0.0])
        self.sim.model.body_pos[self.hole_body_id] = self.hole_top
        self.sim.forward()

    # ------------------------------------------------------------ control
    # 행동 = 막대 끝 목표 위치 (world, m, 3차원). 자세 목표는 초기 자세로 고정.
    # 제어 법칙 (시뮬레이션 매 스텝, 500 Hz):
    #   F = K (p* - p) - D v + F_ff,   M = K_r e_R - D_r w
    #   tau = Jp^T F + Jr^T M + N^T (k_q (q0 - q) - d_q qd) + 중력·코리올리 보상
    # 강성 행렬이 대각(world 기준)이라 축 사이 결합이 없다.

    @property
    def action_dim(self):
        return 3

    @property
    def action_spec(self):
        return -np.ones(3) * 10, np.ones(3) * 10

    def tip_target_action(self, tip_target):
        return np.asarray(tip_target, dtype=float).copy()

    def _pre_action(self, action, policy_step=False):
        if policy_step or self.tip_target is None:
            self.tip_target = np.asarray(action[:3], dtype=float).copy()
        m, d = self.sim.model._model, self.sim.data._data
        robot = self.robots[0]
        dofs = robot._ref_arm_joint_vel_indexes
        qpos_idx = robot._ref_arm_joint_pos_indexes

        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, self.peg_tip_site_id)
        jp, jr = jp[:, dofs], jr[:, dofs]

        p = d.site_xpos[self.peg_tip_site_id]
        R = d.site_xmat[self.peg_tip_site_id].reshape(3, 3)
        qd = d.qvel[dofs]
        v, w = jp @ qd, jr @ qd

        F = self.k_trans * (self.tip_target - p) - self.d_trans * v + self.ff_force
        Rd = self._rot_target
        e_r = 0.5 * (np.cross(R[:, 0], Rd[:, 0]) + np.cross(R[:, 1], Rd[:, 1]) + np.cross(R[:, 2], Rd[:, 2]))
        M = self.k_rot * e_r - self.d_rot * w

        J = np.vstack([jp, jr])
        Jpinv = np.linalg.pinv(J)
        N = np.eye(len(dofs)) - Jpinv @ J
        tau_null = N.T @ (self.k_null * (self._q0 - d.qpos[qpos_idx]) - self.d_null * qd)

        tau = jp.T @ F + jr.T @ M + tau_null + d.qfrc_bias[dofs]
        act = robot._ref_arm_joint_actuator_indexes
        lo, hi = m.actuator_ctrlrange[act, 0], m.actuator_ctrlrange[act, 1]
        tau = np.clip(tau, lo, hi)
        d.ctrl[act] = tau
        d.ctrl[self._grip_act] = self._grip_ctrl
        self._last_torque = tau

        # 구멍 안 저항 (xfrc_applied로 막대에 직접 가함)
        #  - 마찰: 움직이는 반대 방향, 5 mm/s 이하에서는 속도에 비례 (수치 안정용)
        #  - 걸쇠: 깊이에 따른 반발 스프링 (위쪽). 정점을 넘으면 0 → 힘이 모자라면 확실히 멈춘다
        f = np.zeros(6)
        depth = self.insertion_depth()
        R = self.friction_force(depth)
        if R > 0:
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, self.peg_tip_site_id, vel, 0)
            f[2] -= R * np.clip(vel[5] / 0.005, -1.0, 1.0)
        f[2] += self.snap_force(depth)
        d.xfrc_applied[self.peg_body_id] = f

    def friction_force(self, depth):
        """깊이(m)에 따른 마찰 크기(N). 이동 반대 방향."""
        if self.resist == "none" or depth <= self.resist_from:
            return 0.0
        if self.resist == "const":
            return self.insert_resist
        return self.base_resist

    def snap_force(self, depth):
        """걸쇠 반발력(N, 위쪽 +). 갈고리가 휘며 깊이에 비례해 커지다가 정점을 넘으면 사라진다."""
        if self.resist != "snap":
            return 0.0
        z = depth / MM
        a, b = self.snap_mm
        if a <= z < b:
            return self.snap_peak * (z - a) / (b - a)
        return 0.0

    def resist_profile(self, depth):
        """움직이는 중 막대가 받는 총 저항(N). 그림·분석용."""
        return self.friction_force(depth) + self.snap_force(depth)

    # ------------------------------------------------------------ helpers

    def peg_tip_pos(self):
        return self.sim.data.site_xpos[self.peg_tip_site_id].copy()

    def wrist_force(self):
        """손목 힘 (센서 좌표계, 바이어스 제거, N)."""
        return self.robots[0].ee_force["right"] - self._force_bias

    def wrist_force_world(self):
        """손목 힘 (world 좌표, 바이어스 제거, N). 아래로 누르면 z가 음수."""
        R = self.sim.data.site_xmat[self.ft_site_id].reshape(3, 3)
        return R @ self.wrist_force()

    def wrist_torque(self):
        return self.robots[0].ee_torque["right"] - self._torque_bias

    def insertion_depth(self):
        """구멍 윗면 기준 막대 끝 삽입 깊이 (m, 양수 = 안쪽)."""
        return self.hole_top[2] - self.peg_tip_pos()[2]

    def lateral_error(self):
        """막대 끝과 실제 구멍 중심의 수평 거리 (m)."""
        d = self.peg_tip_pos()[:2] - self.hole_top[:2]
        return float(np.linalg.norm(d))

    def contact_force_norm(self):
        """막대-구멍 접촉력 합력의 크기 (N, 시뮬레이터 직접 계산)."""
        total = np.zeros(3)
        f6 = np.zeros(6)
        for i in range(self.sim.data.ncon):
            con = self.sim.data.contact[i]
            if self.peg_geom_id in (con.geom1, con.geom2):
                mujoco.mj_contactForce(self.sim.model._model, self.sim.data._data, i, f6)
                total += con.frame.reshape(3, 3).T @ f6[:3]   # 접촉 좌표 → world, 벡터 합
        return float(np.linalg.norm(total))

    # ------------------------------------------------------- task logic
    def _check_success(self):
        return self.insertion_depth() >= self.success_depth

    def reward(self, action=None):
        return 1.0 if self._check_success() else 0.0
