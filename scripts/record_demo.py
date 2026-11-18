"""
3단계: 힘 조절 교시 시범 기록.

교시자(스크립트)는 아래 순서로 막대를 끼운다.
  FREE    : 구멍 위 10 mm에서 10 mm/s로 하강, 힘 0
  PRESS   : 접촉 감지(위쪽 반력 > 2 N) → 최대 10 N까지 누름
  INSERT  : 진입 감지(접촉 높이보다 2 mm 하강) → 최대 15 N까지 누르며 10 mm/s로 25 mm까지 삽입
  RELEASE : 힘 0, 1 s 유지

교시자 제어 = 힘 제한 위치 제어: 목표를 10 mm/s로 내리되 목표와 막대 끝 차이를 F_lim/K 이하로 묶는다.
막히면 누르는 힘이 F_lim까지만 쌓이고, 안 막히면 10 mm/s로 내려간다(빈 구멍에서 가속해 바닥을 치지 않음).
기록되는 힘 프로파일은 명령값이 아니라 손목 센서에 실제로 잡힌 값이다.

오차 0이면 틈(0.5 mm) 안으로 아무 접촉 없이 들어가 힘이 기록되지 않으므로,
교시자의 정렬 오차를 포착 범위 안쪽(기본 1.0 mm)으로 두고 시범한다.
교시자의 수평 명령은 시범 내내 명목 구멍 중심이고, 1 mm 차이는 모따기와 임피던스 순응으로 흡수된다.

저장 (data/demo/demo_<시각>/):
  demo.npz  : 매 스텝(20 Hz) t, 막대 끝 위치, 명령 위치, 손목 힘·토크(world), 단계, 삽입 깊이, 접촉력, F_ff 명령
  meta.json : 설정, 성공 여부, 단계별 시작 스텝, 단계별 평균 F_z
  profile.png : 위치·힘 프로파일 그래프

사용 예:
  python scripts/record_demo.py
  python scripts/record_demo.py --render
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs import PegHoleEnv  # noqa: E402

PHASES = ["FREE", "PRESS", "INSERT", "RELEASE"]


def record(env, press_N=10.0, insert_N=15.0, contact_N=2.0, entry_mm=2.0,
           depth_mm=25.0, speed_mm_s=10.0, max_steps=400, render=False):
    env.reset()
    dt = 1.0 / env.control_freq
    step = speed_mm_s * 1e-3 * dt
    nom = env.nominal_hole_top.copy()
    tgt = env.peg_tip_pos().copy()
    tgt[:2] = nom[:2]                       # 교시자의 수평 명령 = 명목 중심
    phase, z_contact, release_count = 0, None, 0
    log = {k: [] for k in ["t", "tip", "target", "force", "torque", "phase",
                           "depth", "contact", "ff_cmd"]}

    for k in range(max_steps):
        tip = env.peg_tip_pos()
        fz_up = -env.wrist_force_world()[2]          # 위쪽 반력 (누르면 양수)

        # ---- 단계 전환
        if phase == 0 and fz_up > contact_N:
            phase, z_contact = 1, tip[2]
        elif phase == 1 and tip[2] < z_contact - entry_mm * 1e-3:
            phase = 2
        elif phase == 2 and env.insertion_depth() >= depth_mm * 1e-3:
            phase = 3

        # ---- 명령: 힘 제한 위치 제어 (교시자 = "10 mm/s로 내리되 설정 힘 이상은 안 민다")
        # 목표를 10 mm/s로 내리고, 목표가 막대보다 F/K 이상 앞서지 않게 묶는다.
        # 막히면 스프링 힘이 설정 힘까지만 쌓이고, 안 막히면 그냥 10 mm/s로 따라 내려간다.
        if phase == 0:
            f_lim = press_N
        elif phase == 1:
            f_lim = press_N
        elif phase == 2:
            f_lim = insert_N
        else:
            f_lim = 0.0
        if phase < 3:
            tgt[2] = max(tgt[2] - step, tip[2] - f_lim / env.k_trans)
        else:
            tgt[2] = tip[2]
            release_count += 1
        ff = f_lim if phase in (1, 2) else 0.0   # 기록용: 교시자가 의도한 최대 누름 힘

        env.ff_force = np.zeros(3)
        env.step(env.tip_target_action(tgt))
        if render:
            env.render()

        log["t"].append(k * dt)
        log["tip"].append(env.peg_tip_pos())
        log["target"].append(tgt.copy())
        log["force"].append(env.wrist_force_world())
        R = env.sim.data.site_xmat[env.ft_site_id].reshape(3, 3)
        log["torque"].append(R @ env.wrist_torque())
        log["phase"].append(phase)
        log["depth"].append(env.insertion_depth())
        log["contact"].append(env.contact_force_norm())
        log["ff_cmd"].append(ff)

        if phase == 3 and release_count >= int(1.0 / dt):
            break

    out = {k: np.array(v) for k, v in log.items()}
    out["nominal_hole_top"] = nom
    out["hole_top"] = env.hole_top.copy()
    return out


def summarize(d):
    ph = d["phase"]
    starts = {PHASES[p]: int(np.argmax(ph == p)) if np.any(ph == p) else None for p in range(4)}
    fz_up = -d["force"][:, 2]
    mean_fz = {PHASES[p]: float(fz_up[ph == p].mean()) if np.any(ph == p) else None for p in range(4)}
    return starts, mean_fz


def plot(d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = d["t"]
    fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    ax[0].plot(t, d["depth"] * 1e3, label="insertion depth")
    ax[0].set_ylabel("depth (mm)")
    ax[1].plot(t, -d["force"][:, 2], label="measured F_z (up)")
    ax[1].plot(t, d["ff_cmd"], "--", label="push limit")
    ax[1].set_ylabel("force (N)")
    ax[1].set_xlabel("time (s)")
    ph = d["phase"]
    for a in ax:
        for p in range(1, 4):
            if np.any(ph == p):
                a.axvline(t[np.argmax(ph == p)], color="gray", ls=":", lw=1)
        a.grid(alpha=0.3)
    ax[1].legend(loc="upper right", fontsize=8)
    ax[0].set_title("demo: FREE | PRESS | INSERT | RELEASE")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clearance-mm", type=float, default=0.5)
    ap.add_argument("--chamfer-mm", type=float, default=1.0)
    ap.add_argument("--demo-offset-mm", type=float, nargs=2, default=[1.0, 0.0])
    ap.add_argument("--press-N", type=float, default=10.0)
    ap.add_argument("--insert-N", type=float, default=15.0)
    ap.add_argument("--resist-N", type=float, default=0.0, help="끼움 저항 (0이면 없음)")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--camera", default="side")
    ap.add_argument("--out", default="data/demo")
    a = ap.parse_args()

    env = PegHoleEnv(clearance_mm=a.clearance_mm, chamfer_mm=a.chamfer_mm,
                     offset_mm=tuple(a.demo_offset_mm), insert_resist_N=a.resist_N, has_renderer=a.render,
                     render_camera=a.camera)
    d = record(env, press_N=a.press_N, insert_N=a.insert_N, render=a.render)
    starts, mean_fz = summarize(d)
    success = bool(d["depth"][-1] >= env.success_depth)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(a.out, f"demo_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "demo.npz"), **d)
    meta = dict(
        clearance_mm=a.clearance_mm, chamfer_mm=a.chamfer_mm,
        demo_offset_mm=a.demo_offset_mm, press_N=a.press_N, insert_N=a.insert_N,
        insert_resist_N=a.resist_N, k_trans=env.k_trans, success=success, steps=int(len(d["t"])),
        final_depth_mm=float(d["depth"][-1] * 1e3),
        max_contact_N=float(d["contact"].max()),
        phase_start_step=starts, mean_measured_Fz_up_N=mean_fz,
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    plot(d, os.path.join(out_dir, "profile.png"))
    latest = os.path.join(a.out, "latest")
    if os.path.islink(latest) or os.path.exists(latest):
        os.remove(latest)
    os.symlink(os.path.abspath(out_dir), latest)

    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print("저장:", out_dir)


if __name__ == "__main__":
    main()
