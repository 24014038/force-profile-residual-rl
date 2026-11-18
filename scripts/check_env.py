"""
1단계 환경 점검.

막대를 명목 구멍 중심(오차 모름) 위에서 일정 속도로 내려 보낸다.
- 오차 0이면 들어가야 하고 (success True)
- 오차가 포착 범위(c + δ)보다 크면 윗면에 걸려야 한다 (success False)

사용 예:
  python scripts/check_env.py                       # 오차 0
  python scripts/check_env.py --offset-mm 3 0       # c+δ=4 mm 안: 모따기 타고 들어가길 기대
  python scripts/check_env.py --offset-mm 6 0       # c+δ=4 mm 밖: 막힘 기대
  python scripts/check_env.py --render              # 화면으로 보기 (기본: 구멍 근접 카메라)
  python scripts/check_env.py --render --camera side --hold 10   # 옆에서 (앞벽 반투명), 끝나고 10초 유지
  python scripts/check_env.py --bench               # 속도 측정 (steps/s)
  python scripts/check_env.py --sweep               # 오차 0~6 mm 훑기
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs import PegHoleEnv  # noqa: E402

STEP_MM = 0.5      # 한 스텝당 목표 하강량 (20 Hz → 10 mm/s)


def descend(env, steps=200, render=False):
    env.reset()
    hole_nom = env.nominal_hole_top.copy()
    target = env.peg_tip_pos().copy()
    target[:2] = hole_nom[:2]                                 # 명목 중심 (실제 오차는 모름)
    final_z = hole_nom[2] - 0.025                             # 목표 삽입 깊이 25 mm
    max_wrist, max_contact = 0.0, 0.0
    for t in range(steps):
        target[2] = max(final_z, target[2] - STEP_MM * 1e-3)  # 목표점을 10 mm/s로 내림
        env.step(env.tip_target_action(target))
        if render:
            env.render()
        max_wrist = max(max_wrist, float(np.linalg.norm(env.wrist_force())))
        max_contact = max(max_contact, env.contact_force_norm())
        if env._check_success() and env.insertion_depth() >= 0.025 - 1e-3:
            break
    return dict(
        success=bool(env._check_success()),
        depth_mm=env.insertion_depth() * 1e3,
        lateral_mm=env.lateral_error() * 1e3,
        max_wrist_N=max_wrist,
        max_contact_N=max_contact,
        steps=t + 1,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset-mm", type=float, nargs=2, default=[0.0, 0.0])
    ap.add_argument("--clearance-mm", type=float, default=2.0)
    ap.add_argument("--chamfer-mm", type=float, default=2.0)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--camera", default="closeup", help="closeup / side / frontview / agentview")
    ap.add_argument("--hold", type=float, default=5.0, help="렌더 시 끝난 뒤 화면 유지 시간(초)")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--sweep-mm", type=float, nargs="+", default=[0, 1, 2, 3, 3.5, 4, 4.5, 5, 6])
    a = ap.parse_args()

    kw = dict(clearance_mm=a.clearance_mm, chamfer_mm=a.chamfer_mm,
              has_renderer=a.render, render_camera=a.camera, offset_mm=tuple(a.offset_mm))
    env = PegHoleEnv(**kw)
    print(f"틈 δ = {a.clearance_mm} mm, 모따기 c = {a.chamfer_mm} mm → 포착 범위 c+δ = "
          f"{a.clearance_mm + a.chamfer_mm} mm")

    if a.bench:
        env.reset()
        n = 1000
        t0 = time.time()
        for _ in range(n):
            env.step(env.tip_target_action(env.peg_tip_pos()))
        sps = n / (time.time() - t0)
        print(f"속도: {sps:.0f} steps/s (렌더 없음)")
        print(f"  100,000 step 환경 시간만: {100000 / sps / 60:.1f} 분 (SAC 업데이트 시간 별도)")
        return

    if a.sweep:
        print(" 오차(mm)  성공   깊이(mm)  수평오차(mm)  최대손목힘(N)  최대접촉력(N)")
        for r in a.sweep_mm:
            env.offset = np.array([r, 0.0]) * 1e-3
            res = descend(env)
            print(f" {r:7.1f}  {str(res['success']):5}  {res['depth_mm']:8.1f}  "
                  f"{res['lateral_mm']:11.2f}  {res['max_wrist_N']:13.1f}  {res['max_contact_N']:13.1f}")
        return

    res = descend(env, render=a.render)
    if a.render:
        t_end = time.time() + a.hold
        while time.time() < t_end:
            env.render()
            time.sleep(0.03)
    for k, v in res.items():
        print(f"{k:14s} {v:.2f}" if isinstance(v, float) else f"{k:14s} {v}")


if __name__ == "__main__":
    main()
