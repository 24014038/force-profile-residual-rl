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
  python scripts/record_demo.py --render            # 화면으로 보기 (실제 속도, 끝나고 5 s 유지)
  python scripts/record_demo.py --gif                # demo.gif 저장 (화면 없이, 노션 캡처용)
  python scripts/record_demo.py --gif --camera closeup
"""

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs import PegHoleEnv  # noqa: E402

PHASES = ["FREE", "PRESS", "INSERT", "RELEASE"]


def record(env, press_N=10.0, insert_N=15.0, contact_N=2.0, entry_mm=2.0,
           depth_mm=25.0, speed_mm_s=10.0, max_steps=400, render=False,
           frames=None, camera="side"):
    env.reset()
    dt = 1.0 / env.control_freq
    step = speed_mm_s * 1e-3 * dt
    nom = env.nominal_hole_top.copy()
    tgt = env.peg_tip_pos().copy()
    tgt[:2] = nom[:2]                       # 교시자의 수평 명령 = 명목 중심
    phase, z_contact, release_count = 0, None, 0
    log = {k: [] for k in ["t", "tip", "target", "force", "torque", "phase",
                           "depth", "contact", "ff_cmd"]}

    prev_tip, push = None, 0.0
    for k in range(max_steps):
        tip = env.peg_tip_pos()
        fz_up = -env.wrist_force_world()[2]          # 위쪽 반력 (누르면 양수)

        # ---- 단계 전환
        if phase == 0 and fz_up > contact_N:
            phase, z_contact = 1, tip[2]
        elif phase == 1 and tip[2] < z_contact - entry_mm * 1e-3:
            phase = 2
        elif phase == 2 and tip[2] <= z_contact - (depth_mm - 1.0) * 1e-3:
            phase = 3

        # ---- 명령
        # FREE/PRESS: 힘 제한 위치 제어. 목표를 10 mm/s로 내리되 막대보다 F_lim/K 이상 앞서지 않게 묶는다.
        # INSERT    : 사람처럼 민다. 10 mm/s보다 느리면(막히면) 미는 힘을 올리고(최대 insert_N),
        #             빠르면(걸쇠를 넘어 튀면) 뺀다. 위치 목표는 막대 끝에 붙여 스프링 힘 0.
        # RELEASE   : 힘 0, 제자리.
        v_down = 0.0 if prev_tip is None else (prev_tip[2] - tip[2]) / dt
        if phase in (0, 1):
            tgt[2] = max(tgt[2] - step, tip[2] - press_N / env.k_trans)
            push = 0.0
        elif phase == 2:
            # 속도 오차에 비례해 미는 힘을 조절. 느리면 천천히 올리고(10 mm/s 오차당 1 s에 20 N),
            # 빠르면(딸깍 후 튀어 나감) 세 배 빠르게 뺀다 — 사람이 걸쇠를 넘긴 직후 힘을 푸는 것과 같음
            err = speed_mm_s * 1e-3 - v_down
            gain = 2000.0 if err > 0 else 6000.0
            push = float(np.clip(push + gain * err * dt, 0.0, insert_N))
            if v_down > 2 * speed_mm_s * 1e-3:      # 딸깍: 기준 속도의 두 배를 넘으면 바로 힘을 뺀다
                push = 0.0
            tgt[2] = tip[2]
        else:
            tgt[2] = tip[2]
            push = 0.0
            release_count += 1
        ff = push

        env.ff_force = np.array([0.0, 0.0, -push])
        prev_tip = tip.copy()
        env.step(env.tip_target_action(tgt))
        if render:
            env.render()
            time.sleep(dt)                           # 실제 속도로 보이게
        if frames is not None:
            frames.append(grab_frame(env, camera, PHASES[phase], -env.wrist_force_world()[2],
                                     env.insertion_depth(), k * dt))

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


def grab_frame(env, camera, phase_name, fz_up, depth, t, w=640, h=480):
    """오프스크린 렌더 한 장 + 단계·깊이·힘 글자."""
    from PIL import Image, ImageDraw
    img = env.sim.render(width=w, height=h, camera_name=camera)[::-1]
    im = Image.fromarray(np.ascontiguousarray(img))
    dr = ImageDraw.Draw(im)
    txt = f"t={t:4.2f}s  {phase_name:<7s}  depth={depth*1e3:5.1f} mm  Fz={fz_up:5.1f} N"
    dr.rectangle([0, 0, w, 22], fill=(0, 0, 0))
    dr.text((8, 5), txt, fill=(255, 255, 255))
    return im


def save_gif(frames, path, dt):
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(dt * 1000), loop=0)


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
    ax[1].plot(t, d["ff_cmd"], "--", label="teacher push (command)")
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
    ap.add_argument("--insert-N", type=float, default=20.0)
    ap.add_argument("--resist", default="snap", choices=["none", "const", "snap"])
    ap.add_argument("--resist-N", type=float, default=8.0, help="const일 때 일정 저항")
    ap.add_argument("--snap-peak-N", type=float, default=14.0)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--camera", default="side", help="side(단면) / closeup / frontview")
    ap.add_argument("--gif", action="store_true", help="demo.gif 저장 (화면 없이)")
    ap.add_argument("--out", default="data/demo")
    a = ap.parse_args()

    env = PegHoleEnv(clearance_mm=a.clearance_mm, chamfer_mm=a.chamfer_mm,
                     offset_mm=tuple(a.demo_offset_mm), resist=a.resist, insert_resist_N=a.resist_N,
                     snap_peak_N=a.snap_peak_N, has_renderer=a.render,
                     has_offscreen_renderer=a.gif, render_camera=a.camera)
    frames = [] if a.gif else None
    d = record(env, press_N=a.press_N, insert_N=a.insert_N, render=a.render,
               frames=frames, camera=a.camera)
    if a.render:
        for _ in range(int(5.0 * env.control_freq)):     # 끝난 모습 5 s 유지
            env.render()
            time.sleep(1.0 / env.control_freq)
    starts, mean_fz = summarize(d)
    success = bool(d["depth"][-1] >= env.success_depth)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(a.out, f"demo_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "demo.npz"), **d)
    meta = dict(
        clearance_mm=a.clearance_mm, chamfer_mm=a.chamfer_mm,
        demo_offset_mm=a.demo_offset_mm, press_N=a.press_N, insert_N=a.insert_N,
        resist=a.resist, insert_resist_N=a.resist_N, snap_peak_N=a.snap_peak_N, k_trans=env.k_trans, success=success, steps=int(len(d["t"])),
        final_depth_mm=float(d["depth"][-1] * 1e3),
        max_contact_N=float(d["contact"].max()),
        phase_start_step=starts, mean_measured_Fz_up_N=mean_fz,
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    plot(d, os.path.join(out_dir, "profile.png"))
    if frames:
        save_gif(frames, os.path.join(out_dir, "demo.gif"), 1.0 / env.control_freq)
        frames[0].save(os.path.join(out_dir, "frame_start.png"))
        i_peak = int(np.argmax(-d["force"][:, 2] * (d["phase"] == 2) * (d["contact"] < 1.0)))
        frames[i_peak].save(os.path.join(out_dir, "frame_snap.png"))
        frames[-1].save(os.path.join(out_dir, "frame_end.png"))
    latest = os.path.join(a.out, "latest")
    if os.path.islink(latest) or os.path.exists(latest):
        os.remove(latest)
    os.symlink(os.path.abspath(out_dir), latest)

    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print("저장:", out_dir)


if __name__ == "__main__":
    main()
