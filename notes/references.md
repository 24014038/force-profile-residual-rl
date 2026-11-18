# 참고문헌 후보 (11/18 정리)

- ★ = 2쪽 KRoC 논문 기본 세트 (약 9개). 나머지는 11/20 보고서·논문지 버전에서 추가.
- 확인 = 서지 정보를 출판처·DB에서 확인함 / 기억 = 기억으로 적음, 넣기 전에 확인 필요.

## 1. 잔차 강화학습 (방법의 뿌리)

| | 서지 | 인용 이유 | 확인 |
|---|---|---|---|
| ★ | T. Johannink, S. Bahl, A. Nair, J. Luo, A. Kumar, M. Loskyll, J. A. Ojea, E. Solowjow, S. Levine, "Residual Reinforcement Learning for Robot Control," *ICRA*, pp. 6023–6029, 2019. | 고정 제어기 + RL 보정 구조의 원조. 우리 a = a_nominal + Δa의 근거 | 확인 |
| | T. Silver, K. Allen, J. Tenenbaum, L. Kaelbling, "Residual Policy Learning," arXiv:1812.06298, 2018. | 같은 시기 제안된 잔차 정책 학습. Johannink와 함께 묶어 인용 가능 | 기억 |

## 2. 교시 + 잔차 (nominal을 교시로 만드는 근거)

| | 서지 | 인용 이유 | 확인 |
|---|---|---|---|
| ★ | T. Davchev, K. S. Luck, M. Burke, F. Meier, S. Schaal, S. Ramamoorthy, "Residual Learning From Demonstration: Adapting DMPs for Contact-Rich Manipulation," *IEEE RA-L*, vol. 7, no. 2, pp. 4488–4495, 2022. | 교시 궤적(DMP) 위에 잔차 학습. 단, 교시는 위치만 사용 → 우리는 힘 정보를 추가해 비교 | 확인 |
| | M. Alakuijala, G. Dulac-Arnold, J. Mairal, J. Ponce, C. Schmid, "Residual Reinforcement Learning from Demonstrations," arXiv:2106.08050, 2021. | 교시 기반 잔차 RL의 다른 예 | 기억 |

## 3. 힘이 들어간 nominal + 잔차 RL (가장 가까운 연구, 차이를 꼭 써야 함)

| | 서지 | 인용 이유 / 우리와 차이 | 확인 |
|---|---|---|---|
| ★ | K. Abuibaid, A. Sidorenko, A. Wagner, M. Ruskowski, "Learning-Based Strategy for Composite Robot Assembly Skill Adaptation," arXiv:2604.06949, 2026. | MuJoCo + SAC + 잔차 RL + peg-in-hole, nominal = 힘/위치 하이브리드 제어기. **차이:** nominal을 손으로 설계했고, 위치만 쓴 nominal과 비교하지 않음 | 확인 |
| | A. Ranjbar, N. A. Vien, H. Ziesche, J. Boedecker, G. Neumann, "Residual Feedback Learning for Contact-Rich Manipulation Tasks with Uncertainty," *IROS*, 2021. | 임피던스 기반 상태기계(일정 힘으로 누름) + 잔차. **차이:** 힘은 사람이 정한 상수, 교시 없음, 잔차 방식끼리만 비교 | 확인 (쪽수 미확인) |

## 4. 힘 교시가 필요한 이유 (동기)

| | 서지 | 인용 이유 | 확인 |
|---|---|---|---|
| ★ | P. Kormushev, S. Calinon, D. G. Caldwell, "Imitation Learning of Positional and Force Skills Demonstrated via Kinesthetic Teaching and Haptic Input," *Advanced Robotics*, vol. 25, no. 5, pp. 581–603, 2011. | 교시로 위치와 힘을 함께 학습하는 고전 연구 | 확인 |
| | S. Scherzinger, A. Roennau, R. Dillmann, "Learning Human-Inspired Force Strategies for Robotic Assembly," *IEEE CASE*, 2023 (arXiv:2303.12440). | 교시 힘 프로파일 학습. "위치만으로는 힘 전략을 설명할 수 없다" → 우리 동기 뒷받침. RL 없음 | 학회는 dblp 기준, 쪽수 미확인 |

## 5. 최근 유사 연구 (논문지 버전 관련 연구)

| | 서지 | 우리와 차이 | 확인 |
|---|---|---|---|
| | C. Jiang, Y. Cai, S. Li, W. Sun, J. Carvalho, K. Daniilidis, O. Arenz, C. de Farias, J. Peters, "Residual Compliant Policy: Learning Compliant Policies from Few Demonstrations," arXiv:2610.14035, 2026. | 교시 기반 nominal + PPO 잔차(위치 3축만). 교시에 힘 프로파일 없음, nominal 종류별 비교 없음 | 확인 (게재처 미표기) |
| | Y. Huang, N. Ma, W. Zhao, Z. Liu, J. Sun, Q. Wang, Y. Chen, "Force-Aware Residual DAgger via Trajectory Editing for Precision Insertion with Impedance Control," arXiv:2603.04038, 2026. | nominal이 위치+힘 예측, 힘 유무 성공률 비교(27.6→32.4%). 모방학습이고 학습 부담·축별 분석 없음 | 확인 (게재처 미표기) |

## 6. 과제 모델링 (모따기, 걸쇠)

| | 서지 | 인용 이유 | 확인 |
|---|---|---|---|
| ★ | D. E. Whitney, "Quasi-Static Assembly of Compliantly Supported Rigid Parts," *J. Dyn. Syst. Meas. Control*, vol. 104, no. 1, pp. 65–77, 1982. (DOI 10.1115/1.3149634) | 모따기·순응 지지 삽입의 고전 해석. 포착 범위 c+δ, 마찰 조건 μ < tan α의 근거 | 저자·연도·DOI 확인, 권·쪽은 기억 |
| ★ | A. Lämmle, P. Tenbrock, B. A. Bálint, D. Traunecker, F. Nägele, J. Váncza, M. F. Huber, "Simulation-Based Learning of Electrical Cabinet Assembly Using Robot Skills," arXiv:2602.14561, 2026. | MuJoCo 스냅핏 + SAC. 강체 시뮬레이션만으로는 스냅 변형력이 안 나와 외부 힘 모델을 넣음 → 우리 걸쇠 저항 모델의 근거 | 확인 |
| | DS.04.CERTH.SnapFitForceProfiles, Zenodo, https://zenodo.org/records/1209203 | 실제 스냅핏 삽입력 측정 데이터. 걸쇠 정점 값 근거 후보 (내용 확인 예정) | 미확인 |

## 7. 도구

| | 서지 | 확인 |
|---|---|---|
| ★ | T. Haarnoja, A. Zhou, P. Abbeel, S. Levine, "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor," *ICML*, pp. 1861–1870, 2018. | 기억 |
| ★ | E. Todorov, T. Erez, Y. Tassa, "MuJoCo: A Physics Engine for Model-Based Control," *IROS*, pp. 5026–5033, 2012. | 기억 |
| | N. Hogan, "Impedance Control: An Approach to Manipulation: Part I—Theory," *J. Dyn. Syst. Meas. Control*, vol. 107, no. 1, pp. 1–7, 1985. | 기억 |
| | Y. Zhu et al., "robosuite: A Modular Simulation Framework and Benchmark for Robot Learning," arXiv:2009.12293, 2020. | 기억 |
| | A. Raffin et al., "Stable-Baselines3: Reliable Reinforcement Learning Implementations," *JMLR*, vol. 22, no. 268, pp. 1–8, 2021. | 기억 |

## 관련 연구 단락 초안 (2쪽 논문용, 3문장)

잔차 강화학습은 고정된 기본 제어기 위에 RL이 보정량만 학습하는 방식으로 [Johannink 2019], 교시 궤적을 기본 제어기로 쓰는 연구 [Davchev 2022]와 힘 제어가 포함된 기본 제어기를 쓰는 연구 [Abuibaid 2026, Ranjbar 2021]로 확장되었다.
한편 교시에 담긴 힘 정보가 위치만으로는 표현되지 않는 조립 전략을 담는다는 점은 모방학습 분야에서 보고되었다 [Kormushev 2011, Scherzinger 2023].
그러나 교시 기반 기본 제어기에 힘 정보를 얼마나 넣느냐에 따라 잔차 RL이 배워야 할 양이 어떻게, 어느 축에서 달라지는지를 분리해 측정한 연구는 찾지 못했으며, 본 연구는 이를 위치만(P), 일정 힘(P+C), 힘 프로파일(P+F)의 세 조건으로 비교한다.

## 검색 범위 메모

- 11/18 arXiv·영문 웹 검색 기준. 국내(KCI, RISS, 이전 KRoC) 미확인 → "잔차 강화학습 힘 교시", "residual reinforcement learning force demonstration" 로 확인 필요.
