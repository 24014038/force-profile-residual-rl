"""
4·6단계: RL 없이 nominal(P / P+C / P+F)만으로 평가.

반경별 고정 오차 세트(configs/eval_offsets.json, 반경당 20개, 모든 조건·시드 공통)에서
에피소드 200 step(10 s)씩 돌려 성공률·성공 시각·최종 깊이·최대 접촉력을 잰다.

--z-extra-mm 를 주면 위치 목표를 그만큼 더 아래로 둔다 (RL이 z 보정을 최대로 줬을 때 통과 가능한지 확인용).

사용 예:
  python scripts/eval_nominal.py                        # 기본: demos/snap14, 3조건 × 반경 0/1/2/3
  python scripts/eval_nominal.py --modes P P+C P+Cmax P+F   # 최대 힘 일정 조건 포함
  python scripts/eval_nominal.py --z-extra-mm 0 3 10 --modes P
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs import PegHoleEnv  # noqa: E402
from envs.nominal import ALL_MODES, MODES, Demo, NominalPolicy, load_eval_offsets, make_eval_offsets  # noqa: E402

MM = 1e-3


def run_episode(env, policy, offset_mm, z_extra_mm=0.0, horizon=200):
    env.offset = np.asarray(offset_mm, dtype=float) * MM
    env.reset()
    policy.reset(env)
    t_succ, max_fz, max_phase = None, 0.0, 0
    inter = []
    for k in range(horizon):
        target, F = policy(env)
        target[2] -= z_extra_mm * MM
        env.ff_force = np.array([0.0, 0.0, -F])
        env.step(env.tip_target_action(target))
        inter.append(env.interaction_force_norm())
        max_fz = max(max_fz, -env.wrist_force_world()[2])
        max_phase = max(max_phase, policy.phase)
        if t_succ is None and env.insertion_depth() >= env.success_depth:
            t_succ = (k + 1) / env.control_freq
    env.ff_force = np.zeros(3)
    inter = np.array(inter)
    med3 = np.median(np.stack([inter[:-2], inter[1:-1], inter[2:]]), axis=0)
    return dict(success=bool(env.insertion_depth() >= env.success_depth), t_success=t_succ,
                depth_mm=env.insertion_depth() / MM, lateral_mm=env.lateral_error() / MM,
                max_force_N=float(med3.max()),          # 지속되는 최대 힘 (1 스텝 충격 제외)
                peak_impact_N=float(inter.max()),       # 순간 최대 (충격 포함)
                max_fz_N=max_fz, max_phase=max_phase)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default="demos/snap14", help="본 실험 교시 (Git에 고정). 새로 기록한 걸 쓰려면 data/demo/latest")
    ap.add_argument("--offsets", default="configs/eval_offsets.json")
    ap.add_argument("--modes", nargs="+", default=MODES)
    ap.add_argument("--radii", type=float, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--z-extra-mm", type=float, nargs="+", default=[0.0])
    ap.add_argument("--clearance-mm", type=float, default=0.5)
    ap.add_argument("--chamfer-mm", type=float, default=1.0)
    ap.add_argument("--resist", default="snap")
    ap.add_argument("--snap-peak-N", type=float, default=14.0)
    ap.add_argument("--out", default="results/nominal_eval")
    a = ap.parse_args()

    if not os.path.exists(a.offsets):
        os.makedirs(os.path.dirname(a.offsets), exist_ok=True)
        with open(a.offsets, "w") as f:
            json.dump(make_eval_offsets(), f, indent=1)
        print("평가 오차 세트 생성:", a.offsets)
    offsets = load_eval_offsets(a.offsets)

    demo = Demo(a.demo)
    print("교시:", json.dumps(demo.summary(), ensure_ascii=False))
    env = PegHoleEnv(clearance_mm=a.clearance_mm, chamfer_mm=a.chamfer_mm, resist=a.resist,
                     snap_peak_N=a.snap_peak_N)

    rows = []
    for z_extra in a.z_extra_mm:
        for mode in a.modes:
            pol = NominalPolicy(demo, mode)
            for r in a.radii:
                offs = offsets[float(r)]
                if r == 0:
                    offs = offs[:1]                  # 오차 0은 20개가 모두 같음
                for i, o in enumerate(offs):
                    res = run_episode(env, pol, o, z_extra)
                    rows.append(dict(mode=mode, z_extra_mm=z_extra, radius_mm=r, idx=i,
                                     dx_mm=o[0], dy_mm=o[1], **res))

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n z보정  조건    반경  성공률   성공시각(s)  최종깊이(mm)  최대힘(N) 중앙/최대   순간충격(N) 중앙")
    summary = []
    for z_extra in a.z_extra_mm:
        for mode in a.modes:
            for r in a.radii:
                g = [x for x in rows if x["mode"] == mode and x["radius_mm"] == r and x["z_extra_mm"] == z_extra]
                sr = np.mean([x["success"] for x in g])
                ts = [x["t_success"] for x in g if x["t_success"] is not None]
                mc = [x["max_force_N"] for x in g]
                pk = [x["peak_impact_N"] for x in g]
                s = dict(z_extra_mm=z_extra, mode=mode, radius_mm=r, n=len(g), success_rate=float(sr),
                         t_success_mean=float(np.mean(ts)) if ts else None,
                         depth_mean_mm=float(np.mean([x["depth_mm"] for x in g])),
                         max_force_median=float(np.median(mc)), max_force_max=float(np.max(mc)),
                         impact_median=float(np.median(pk)))
                summary.append(s)
                tstr = f"{s['t_success_mean']:5.2f}" if ts else "  -  "
                print(f" {z_extra:4.0f}  {mode:5s}  {r:4.1f}  {sr*100:5.0f}%     {tstr}       "
                      f"{s['depth_mean_mm']:6.1f}       {s['max_force_median']:5.1f} / {s['max_force_max']:5.1f}      {s['impact_median']:6.1f}")
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(dict(demo=demo.summary(), rows=summary), f, indent=1, ensure_ascii=False)
    print("\n저장:", a.out)


if __name__ == "__main__":
    main()
