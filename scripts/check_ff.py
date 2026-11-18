"""
2단계 확인: 힘 피드포워드(F_ff)를 주면 그만큼 실제로 누르는가?

구멍을 5 mm 옆으로 옮겨 막대가 윗면에 막히게 한 뒤,
z 목표를 막대 끝 위치에 고정(스프링 힘 0)하고 F_ff만 바꿔가며 1 s씩 누른다.
마지막 0.5 s 평균 접촉력과 손목 힘이 F_ff와 같으면 통과.

비교로, F_ff 없이 목표만 25 mm 아래로 둔 P 방식의 힘(스프링 힘)도 잰다.

사용 예:
  python scripts/check_ff.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs import PegHoleEnv  # noqa: E402


def touch_down(env):
    """윗면에 닿을 때까지 천천히 내린다."""
    env.reset()
    tgt = env.peg_tip_pos().copy()
    tgt[:2] = env.nominal_hole_top[:2]
    for _ in range(200):
        tgt[2] -= 0.0005
        env.step(env.tip_target_action(tgt))
        if -env.wrist_force_world()[2] > 2.0:
            break
    return tgt


def press(env, ff_N, n=20):
    tgt = touch_down(env)
    cs, ws = [], []
    for k in range(n):
        tgt[2] = env.peg_tip_pos()[2]          # 스프링 힘 0
        env.ff_force = np.array([0.0, 0.0, -ff_N])
        env.step(env.tip_target_action(tgt))
        if k >= n // 2:
            cs.append(env.contact_force_norm())
            ws.append(env.wrist_force_world()[2])
    env.ff_force = np.zeros(3)
    return float(np.mean(cs)), float(np.mean(ws))


def spring_only(env, below_mm=25.0, n=40):
    tgt = touch_down(env)
    tgt[2] = env.peg_tip_pos()[2] - below_mm * 1e-3
    for _ in range(n):
        env.step(env.tip_target_action(tgt))
    return env.contact_force_norm()


def main():
    env = PegHoleEnv(clearance_mm=0.5, chamfer_mm=1.0, offset_mm=(5.0, 0.0))
    print("F_ff 명령(N) | 접촉력(N) | 손목 F_z world(N)")
    for f in [0, 5, 10, 15]:
        c, w = press(env, f)
        print(f"   {f:5.1f}     |  {c:6.1f}   |  {w:7.1f}")
    print(f"\nP 방식(F_ff 없음, 목표 25 mm 아래) 스프링 힘: {spring_only(env):.1f} N")


if __name__ == "__main__":
    main()
