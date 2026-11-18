"""
4단계: 교시 재생기 (nominal).

세 조건 모두 같은 제어기, 같은 위치 궤적, 같은 이벤트 감지기를 쓰고, 미는 힘(F_ff)만 다르다.

  위치 궤적  : 시간 기준. 교시 때 막대 끝 높이(명목 구멍 윗면 기준)를 20 Hz 그대로 재생.
               수평 목표는 명목 구멍 중심 (교시자의 수평 명령과 같음). 시범이 끝나면 마지막 높이 유지.
               → 걸려도 목표는 계속 내려가므로 P도 스프링 힘으로 누른다.
  이벤트     : 접촉 = 위쪽 반력 > 2 N, 진입 = 접촉 높이보다 2 mm 하강,
               해제 = 삽입 깊이가 교시의 해제 깊이에 도달.
  미는 힘    : P    = 0
               P+C  = 접촉 ~ 해제 동안 교시 F_z 평균(일정)
               P+F  = 접촉 ~ 해제 동안 교시 F_z 프로파일을 '삽입 깊이'로 읽는다.
                      (이벤트로 켜고 끄고, 켜져 있는 동안은 깊이로 맞춤 → 걸쇠 위치에서 세게 민다)
  삽입 깊이  : 명목 구멍 윗면 높이 − 막대 끝 높이. 구멍 오차는 수평 방향뿐이라 윗면 높이는 모든 조건이 아는 값이고,
               위치 궤적도 같은 기준을 쓴다.
               (처음엔 '접촉한 높이부터 들어간 거리'로 읽었는데, 접촉 높이가 상황마다 0~2 mm 달라
                폭 2.5 mm짜리 걸쇠 정점 직전에 힘이 빠지는 문제가 있어 바꿈)

교시 F_z는 1 샘플짜리 튐(딸깍 충격)을 없애려고 3점 중앙값 필터를 거친다.
"""

import json
import os

import numpy as np

PHASES = ["FREE", "PRESS", "INSERT", "RELEASE"]
MODES = ["P", "P+C", "P+F"]          # 본 실험 조건
ALL_MODES = MODES + ["P+Cmax"]        # P+Cmax: 교시 최대 힘을 일정하게 (검토용)


def _median3(x):
    y = x.copy()
    y[1:-1] = np.median(np.stack([x[:-2], x[1:-1], x[2:]]), axis=0)
    return y


class Demo:
    """기록된 교시(demo.npz)에서 위치 궤적과 힘 프로파일을 뽑는다."""

    def __init__(self, path, bin_mm=0.25):
        f = os.path.join(path, "demo.npz") if os.path.isdir(path) else path
        d = np.load(f)
        self.path = f
        self.dt = float(d["t"][1] - d["t"][0])
        nom = d["nominal_hole_top"]
        ph = d["phase"]
        tip = d["tip"]

        # 위치 궤적: 명목 구멍 윗면 기준 막대 끝 높이 (m)
        self.z_rel = tip[:, 2] - nom[2]

        s = nom[2] - tip[:, 2]                      # 삽입 깊이 (명목 윗면 기준, m)
        fz = _median3(-d["force"][:, 2])            # 위쪽 반력 (N), 1 샘플 튐 제거

        i_r = int(np.argmax(ph == 3)) if np.any(ph == 3) else len(ph)
        self.s_release = float(s[i_r - 1])          # 이 깊이에 도달하면 해제
        m = (ph >= 1) & (ph <= 2)

        # 깊이별 프로파일 (bin 중앙값, 빈 bin은 보간)
        edges = np.arange(np.floor(s[m].min() / (bin_mm * 1e-3)) * bin_mm * 1e-3,
                          self.s_release + bin_mm * 1e-3, bin_mm * 1e-3)
        centers, vals = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = m & (s >= lo) & (s < hi)
            if np.any(sel):
                centers.append((lo + hi) / 2)
                vals.append(float(np.median(fz[sel])))
        self.prof_s = np.array(centers)
        self.prof_F = np.clip(np.array(vals), 0.0, None)
        self.F_const = float(np.clip(fz[m].mean(), 0.0, None))
        self.raw_fz = fz
        self.raw_s = s
        self.phase = ph

    def force_at(self, depth):
        """삽입 깊이(m)에서 교시가 민 힘(N). 범위 밖은 양 끝 값."""
        return float(np.interp(depth, self.prof_s, self.prof_F))

    def summary(self):
        return dict(
            demo=self.path, steps=int(len(self.z_rel)), dt=self.dt,
            release_depth_mm=self.s_release * 1e3, F_const_N=self.F_const,
            F_peak_N=float(self.prof_F.max()),
            depth_at_peak_mm=float(self.prof_s[int(np.argmax(self.prof_F))] * 1e3),
        )


class NominalPolicy:
    """매 스텝 (막대 끝 위치 목표, 아래로 미는 힘 N)을 돌려준다."""

    def __init__(self, demo, mode, contact_N=2.0, entry_mm=2.0):
        assert mode in ALL_MODES, mode
        self.demo, self.mode = demo, mode
        self.contact_N, self.entry = contact_N, entry_mm * 1e-3

    def reset(self, env):
        self.k = 0
        self.phase = 0
        self.z_contact = None
        self.nom = env.nominal_hole_top.copy()

    def depth(self, tip_z):
        return self.nom[2] - tip_z

    def __call__(self, env):
        tip = env.peg_tip_pos()
        fz_up = -env.wrist_force_world()[2]

        # 이벤트 (세 조건 공통)
        if self.phase == 0 and fz_up > self.contact_N:
            self.phase, self.z_contact = 1, tip[2]
        elif self.phase == 1 and tip[2] < self.z_contact - self.entry:
            self.phase = 2
        if self.phase in (1, 2) and self.depth(tip[2]) >= self.demo.s_release:
            self.phase = 3

        # 위치 목표 (시간 기준)
        i = min(self.k, len(self.demo.z_rel) - 1)
        target = np.array([self.nom[0], self.nom[1], self.nom[2] + self.demo.z_rel[i]])

        # 미는 힘
        if self.mode == "P" or self.phase in (0, 3):
            F = 0.0
        elif self.mode == "P+C":
            F = self.demo.F_const
        elif self.mode == "P+Cmax":
            F = float(self.demo.prof_F.max())
        else:
            F = self.demo.force_at(self.depth(tip[2]))

        self.k += 1
        return target, F


# ---------------------------------------------------------------- 평가용 오차 세트

def sample_offset(rng, r_mm):
    """반경 r 원판 안에서 균일하게 (θ ~ U[0, 2π), ρ = r √u). mm 단위."""
    th = rng.uniform(0, 2 * np.pi)
    rho = r_mm * np.sqrt(rng.uniform())
    return np.array([rho * np.cos(th), rho * np.sin(th)])


def make_eval_offsets(radii=(0, 1, 2, 3), n=20, seed=2026):
    rng = np.random.default_rng(seed)
    return {str(r): [sample_offset(rng, r).round(4).tolist() for _ in range(n)] for r in radii}


def load_eval_offsets(path):
    with open(path) as f:
        return {float(k): np.array(v) for k, v in json.load(f).items()}
