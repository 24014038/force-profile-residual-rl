"""
generate_demo.py
----------------
NutAssemblyRound(robosuite 1.5.x)에서 '오차 0' nominal 시범을 스크립트로 생성하고
위치 / 자세 / 힘·토크 / 행동 / 단계 라벨을 저장한다.

  python generate_demo.py                 # 화면 없이 실행, ./nominal_data 에 저장
  python generate_demo.py --render        # 시뮬레이션 창 띄우기
  python generate_demo.py --peg-offset-mm 2 0   # 구멍(peg) 위치 오차 2 mm 주고 같은 nominal 재생 (실험용)

검증한 API: robosuite 1.5.2 소스 (load_composite_controller_config, robot.ee_force 등)
필요 버전: robosuite >= 1.5.0, mujoco >= 3.2.3 (1.5.2 는 >= 3.3.0, < 3.10)
"""

import argparse
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# 0. 버전 점검 — 에러의 가장 흔한 원인이라 맨 먼저 확인
# ---------------------------------------------------------------------------
def check_versions():
    import mujoco
    import robosuite

    mj = tuple(int(x) for x in mujoco.__version__.split(".")[:3])
    rs = tuple(int(x) for x in robosuite.__version__.split(".")[:2])
    print(f"[check] python {sys.version.split()[0]} | mujoco {mujoco.__version__} | robosuite {robosuite.__version__}")
    if rs < (1, 5):
        sys.exit(
            "[error] robosuite 1.4 이하입니다. 이 스크립트는 1.5 API(load_composite_controller_config, BASIC)를 씁니다.\n"
            "        pip install 'robosuite>=1.5.1'"
        )
    if mj < (3, 2, 3):
        sys.exit(
            f"[error] mujoco {mujoco.__version__} 는 robosuite 1.5 와 맞지 않습니다 (>= 3.2.3 필요).\n"
            "        pip install 'mujoco>=3.3,<3.10'"
        )
    if mj >= (3, 10, 0):
        sys.exit("[error] mujoco 3.10 이상은 robosuite 1.5.2 컨트롤러와 충돌합니다. pip install 'mujoco>=3.3,<3.10'")


# ---------------------------------------------------------------------------
# 1. 환경 생성
# ---------------------------------------------------------------------------
def make_env(render, seed, horizon):
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config

    # Panda 기본 컨트롤러 = BASIC(composite) 안에 OSC_POSE(right arm) + GRIP
    # controller="BASIC" 을 직접 넣으면 torso/head/legs 설정까지 같이 들어오므로 robot 기준으로 불러온다.
    ctrl_cfg = load_composite_controller_config(controller=None, robot="Panda")

    env = suite.make(
        "NutAssemblyRound",
        robots="Panda",
        controller_configs=ctrl_cfg,
        has_renderer=render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
        horizon=horizon,
        ignore_done=True,  # horizon 도달로 중간에 끊기지 않게, 종료는 스크립트가 판단
        seed=seed,
    )
    return env


# ---------------------------------------------------------------------------
# 2. 유틸
# ---------------------------------------------------------------------------
def nut_handles(env):
    nut = env.nuts[env.nut_id]  # RoundNut
    body_id = env.obj_body_id[nut.name]
    handle_id = env.sim.model.site_name2id(nut.important_sites["handle"])
    return nut, body_id, handle_id


def place_nut_fixed(env, xy=(-0.11, -0.17), yaw=0.0):
    """nominal 을 매번 같게 하려고 너트를 고정 위치/자세로 놓는다 (yaw=0 → 손잡이가 +x 방향)."""
    nut, _, _ = nut_handles(env)
    z = env.table_offset[2] + 0.012  # 너트 두께 2 cm → 중심이 테이블 위 1 cm, 살짝만 띄움
    quat_wxyz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    env.sim.data.set_joint_qpos(nut.joints[0], np.concatenate([[xy[0], xy[1], z], quat_wxyz]))
    env.sim.forward()


def shift_peg(env, dx, dy):
    """구멍 위치 오차 주입: 실제 peg 만 옮기고, nominal 은 원래 위치를 목표로 한다."""
    pid = env.sim.model.body_name2id("peg2")
    env.sim.model.body_pos[pid][0] += dx
    env.sim.model.body_pos[pid][1] += dy
    env.sim.forward()


def ori_error_axisangle(target_quat_xyzw, cur_quat_xyzw):
    import robosuite.utils.transform_utils as T

    q = T.quat_multiply(target_quat_xyzw, T.quat_inverse(cur_quat_xyzw))
    if q[3] < 0:  # 최단 회전
        q = -q
    return T.quat2axisangle(q)


def make_action(env, target_pos, target_quat, grip, pos_gain=1.0, max_step=0.05):
    """OSC_POSE delta 입력: [-1,1] 이 ±0.05 m, ±0.5 rad 에 대응."""
    eef = np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]])
    dpos = np.clip(pos_gain * (target_pos - eef) / max_step, -1, 1)

    import robosuite.utils.transform_utils as T

    cur_quat = T.mat2quat(np.array(env.sim.data.site_xmat[env.robots[0].eef_site_id["right"]]).reshape(3, 3))
    drot = np.clip(ori_error_axisangle(target_quat, cur_quat) / 0.5, -1, 1)
    return np.concatenate([dpos, drot, [grip]])


# ---------------------------------------------------------------------------
# 3. 스크립트 시범 (waypoint state machine)
# ---------------------------------------------------------------------------
PHASES = ["approach", "descend", "grasp", "lift", "transport", "insert", "release", "retreat"]


def run_demo(env, render, peg_offset_m=(0.0, 0.0)):
    import robosuite.utils.transform_utils as T

    env.reset()
    place_nut_fixed(env)

    # nominal 목표는 '원래' peg 위치 (오차 주입 전에 읽어 둔다)
    peg_id = env.sim.model.body_name2id("peg2")
    peg_nominal = np.array(env.sim.data.body_xpos[peg_id]).copy()
    if any(peg_offset_m):
        shift_peg(env, *peg_offset_m)

    robot = env.robots[0]
    site = robot.eef_site_id["right"]
    eef = lambda: np.array(env.sim.data.site_xpos[site])
    hold_quat = T.mat2quat(np.array(env.sim.data.site_xmat[site]).reshape(3, 3))  # 시작 자세 유지

    nut, nut_body, handle_site = nut_handles(env)
    nut_pos = lambda: np.array(env.sim.data.body_xpos[nut_body])
    handle_pos = lambda: np.array(env.sim.data.site_xpos[handle_site])

    table_z = env.table_offset[2]
    peg_top_z = peg_nominal[2] + 0.1
    safe_z = peg_top_z + 0.08

    log = {k: [] for k in ["pos", "quat", "force", "torque", "action", "phase", "nut_pos"]}
    warns = {}  # 단계별 목표 미도달 거리(mm) 기록

    def step(action, phase):
        env.step(action)
        if render:
            env.render()
        log["pos"].append(eef())
        log["quat"].append(T.mat2quat(np.array(env.sim.data.site_xmat[site]).reshape(3, 3)))  # xyzw
        log["force"].append(np.array(robot.ee_force["right"]))
        log["torque"].append(np.array(robot.ee_torque["right"]))
        log["action"].append(action)
        log["phase"].append(PHASES.index(phase))
        log["nut_pos"].append(nut_pos())

    def go_to(target, grip, phase, tol=0.004, max_steps=250, gain=1.0, ki=0.05, int_axes=(1, 1, 1)):
        # 정상상태 오차(너트 무게, OSC 강성 한계) 보정용 적분항: 남은 오차를 누적해 목표를 살짝 앞당긴다
        integ = np.zeros(3)
        mask = np.array(int_axes, dtype=float)
        for _ in range(max_steps):
            err = target - eef()
            integ = np.clip(integ + err * mask, -0.5, 0.5)  # 최대 보정 0.5*ki = 25 mm
            a = make_action(env, target + ki * integ, hold_quat, grip, pos_gain=gain)
            step(a, phase)
            if np.linalg.norm(target - eef()) < tol:
                return True
        miss = np.linalg.norm(target - eef()) * 1000
        warns[phase] = round(float(miss), 1)
        print(f"  [warn] {phase}: 목표 도달 실패 (남은 거리 {miss:.1f} mm)")
        return False

    def hold(grip, phase, n):
        tgt = eef()
        for _ in range(n):
            step(make_action(env, tgt, hold_quat, grip), phase)

    OPEN, CLOSE = -1.0, 1.0

    # 너트가 테이블에 완전히 내려앉을 때까지 대기 (기록하지 않음)
    still = np.zeros(7); still[-1] = OPEN
    for _ in range(20):
        env.step(still)
        if render:
            env.render()
    h = handle_pos()
    print(f"  [info] handle z = {h[2]:.4f} m (table {table_z:.3f})")

    # 1) 손잡이 위로 접근 → 2) 내려가기 → 3) 잡기 → 4) 들기
    go_to(np.array([h[0], h[1], h[2] + 0.10]), OPEN, "approach")
    h = handle_pos()  # 접근 후 한 번 더 읽기 (밀렸을 수 있음)
    go_to(np.array([h[0], h[1], h[2] - 0.003]), OPEN, "descend", tol=0.003)
    hold(CLOSE, "grasp", 25)
    # 잡았는지 확인: 손가락이 닫힌 뒤 들어올렸을 때 너트가 따라오는지로 판단

    go_to(np.array([eef()[0], eef()[1], safe_z]), CLOSE, "lift")
    lifted = nut_pos()[2] - (table_z + 0.01)
    print(f"  [info] lift: 너트가 {lifted*100:.1f} cm 들림" + ("" if lifted > 0.05 else "  ← 잡기 실패"))

    # 잡은 뒤 '손끝 → 너트 중심' 오프셋을 측정해서, 너트 중심이 peg 위로 가도록 손끝 목표를 계산
    grasp_offset = eef() - nut_pos()
    above_peg = np.array([peg_nominal[0], peg_nominal[1], safe_z]) + np.array([grasp_offset[0], grasp_offset[1], 0])

    # 5) peg 위로 이동 → 6) 끼우기(천천히) → 7) 놓기 → 8) 빠지기
    go_to(above_peg, CLOSE, "transport", tol=0.002)
    insert_target = above_peg.copy()
    insert_target[2] = table_z + 0.03 + grasp_offset[2]  # 너트가 테이블 근처까지 내려가도록
    # 삽입은 z 적분을 끈다 (걸렸을 때 계속 세게 누르지 않도록, xy 정렬만 보정)
    go_to(insert_target, CLOSE, "insert", tol=0.004, max_steps=250, gain=0.4, int_axes=(1, 1, 0))
    hold(OPEN, "release", 15)
    go_to(eef() + np.array([0, 0, 0.10]), OPEN, "retreat")

    success = bool(env._check_success())
    data = {k: np.asarray(v) for k, v in log.items()}
    return data, success, warns


# ---------------------------------------------------------------------------
def main():
    import csv
    import datetime
    import json
    import subprocess

    p = argparse.ArgumentParser()
    p.add_argument("--render", action="store_true", help="시뮬레이션 창 표시")
    p.add_argument("--out", default="data/nominal", help="실행마다 이 아래에 run_날짜_시각 폴더가 생김")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--peg-offset-mm", type=float, nargs=2, default=[0.0, 0.0], metavar=("DX", "DY"))
    p.add_argument("--note", default="", help="이번 실행 메모 (예: '적분항 추가')")
    args = p.parse_args()

    check_versions()
    env = make_env(args.render, args.seed, horizon=2000)
    offset_m = tuple(np.array(args.peg_offset_mm) / 1000.0)

    data, success, warns = run_demo(env, args.render, offset_m)
    env.close()

    # --- 실행마다 새 폴더에 저장 (덮어쓰기 없음) ---
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{stamp}_off{args.peg_offset_mm[0]:g}_{args.peg_offset_mm[1]:g}mm"
    run_dir = os.path.join(args.out, run_name)
    os.makedirs(run_dir, exist_ok=True)
    for k in ["pos", "quat", "force", "torque", "action", "phase"]:
        np.save(os.path.join(run_dir, f"nominal_{k}.npy"), data[k])
    np.savez(os.path.join(run_dir, "nominal_all.npz"), success=success, phases=np.array(PHASES), **data)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    f = np.linalg.norm(data["force"], axis=1)
    meta = dict(
        run=run_name, time=stamp, git_commit=commit, seed=args.seed, peg_offset_mm=args.peg_offset_mm,
        success=success, steps=int(len(data["pos"])), max_force_N=round(float(f.max()), 2),
        phase_miss_mm=warns, note=args.note,
    )
    with open(os.path.join(run_dir, "meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2, ensure_ascii=False)

    # 가장 최근 실행을 data/nominal/latest 로 연결 (그래프 그릴 때 편하게)
    latest = os.path.join(args.out, "latest")
    if os.path.islink(latest):
        os.remove(latest)
    if not os.path.exists(latest):
        os.symlink(run_name, latest)

    # 실행 기록 한 줄을 notes/run_log.csv 에 추가 (이 파일은 Git 에 올라감)
    os.makedirs("notes", exist_ok=True)
    log_path = "notes/run_log.csv"
    new = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as fp:
        w = csv.writer(fp)
        if new:
            w.writerow(["time", "run", "git_commit", "seed", "peg_offset_mm", "success", "steps", "max_force_N", "phase_miss_mm", "note"])
        w.writerow([stamp, run_name, commit, args.seed, args.peg_offset_mm, success, meta["steps"], meta["max_force_N"], warns, args.note])

    print(f"[done] steps={meta['steps']}  success={success}  max|F|={meta['max_force_N']} N  → {run_dir}/")
    if not success:
        print("  성공 판정 실패: --render 로 어느 단계에서 어긋나는지 확인하세요 ([warn] 줄 참고).")


if __name__ == "__main__":
    main()
