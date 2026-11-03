# 힘 프로파일 교시가 잔차 강화학습의 학습 부담에 미치는 영향

**연구 질문.** 교시된 기본 동작(nominal)에 위치 궤적뿐 아니라 **힘 프로파일**까지 넣으면, 위치만 넣은 경우보다 RL이 배워야 할 잔차가 얼마나 줄고 학습이 얼마나 빨라지는가?

세종대학교 Medical Robotics Lab 학부연구생 개인연구

## 가설

위치만 교시된 nominal과 비교해, 위치 + 힘 nominal(임피던스 제어로 걸리면 양보)은

- 잔차(보정량) 크기가 **더 작고**
- 수렴이 **더 빠르고**
- 최대 접촉력이 **더 낮을** 것이다.

단, 구멍 위치 오차가 커지면 임피던스만으로는 부족해 이점이 줄어들 것으로 예상하며, 그 전환 지점도 함께 본다.

## 실험 설계

| 항목 | 설정 |
|---|---|
| 시뮬레이터 | MuJoCo + robosuite 1.5.2 |
| 로봇 | Franka Panda + 손목 힘/토크 센서 |
| 과제 | Peg-in-hole (목표 틈 약 1 mm) |
| 비교 조건 | 위치만 nominal + SAC vs. 위치 + 힘 nominal + SAC |
| 구멍 위치 오차 | 0 / 1 / 2 / 3 mm |
| 측정 지표 | 성공률, 수렴 스텝 수, 잔차 크기, 최대 접촉력 |
| 실험 규모 | nominal 2종 × 오차 4단계 × 시드 3개 = 24회 |

## 진행 단계

1. **교시 데이터 생성** (진행 중): 사람 시범 대신 스크립트가 오차 0 상황에서 시범을 보이고, 손끝 위치·자세·손목 힘/토크·단계 라벨을 기록한다.
2. **nominal 제어기**: (A) 위치 궤적 재생, (B) 위치 + 힘 프로파일 추종(임피던스 제어).
3. **잔차 SAC 학습**: 구멍 오차가 있는 상황에서 각 nominal 위에 보정값 Δa를 학습한다.
4. **평가**: 위 지표로 두 조건을 비교한다.

## 환경 설치

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 사용법

```bash
# nominal 시범 기록 (실행마다 data/nominal/run_<시각>_off<dx>_<dy>mm/ 에 저장)
python scripts/generate_demo.py --render --note "이번에 바꾼 점"

# 구멍 위치 오차를 주고 재생 (실험용)
python scripts/generate_demo.py --peg-offset-mm 2 0
```

실행할 때마다 결과 요약 한 줄이 `notes/run_log.csv`에 쌓인다.

## 진행 상황 (2026-11-03)

- [x] 실험 환경 구축 (가상환경, robosuite 1.5.2, MuJoCo 3.9)
- [x] 스크립트 시범 기록 파이프라인 (위치 / 힘 / 단계)
- [x] 구멍 위치 오차 주입 옵션
- [ ] NutAssemblyRound에서 시범 안정적 성공
- [ ] 틈 1 mm peg-in-hole 환경 제작 (NutAssemblyRound는 틈이 약 11 mm라 0–3 mm 오차 실험에 부적합)
- [ ] nominal A / B 제어기 구현
- [ ] 잔차 SAC 학습 및 평가

## 폴더 구조

```
scripts/    실행 코드
envs/       직접 만들 환경 (예정)
notes/      실험 기록
data/       생성 데이터 (Git 제외)
results/    그래프·지표 (Git 제외)
```
