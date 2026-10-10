# toy: 30큐비트 블록 HVA 학습과 spectral norm 인증

10×10 Heisenberg 모델의 HHKL 블록 하나(5×6 = 30큐비트, ℓ=3)에서 $e^{-iH_\square t^\star}$를 근사하는
palindromic HVA(P1)를 statevector 손실 + adjoint 미분 + L-BFGS로 학습하고,
$\epsilon_\text{spec}=\min_\phi\|U-e^{i\phi}V\|$를 Lanczos로 인증한다. 설계와 수치의 근거는 [`PLAN.md`](PLAN.md)에 있다.

이 문서는 **사용자가 데스크탑과 H200에서 직접 실행하기 위한 가이드**다(PLAN.md §5).
모든 스크립트는 마지막 줄에 `PASS`/`FAIL`과 핵심 수치를 출력하고, 결과를 `results/`에 JSON으로 남긴다.

---

## 1. 설치

```bash
pip install -r requirements.txt          # numpy, scipy, torch, pytest
# 그림(선택): pip install matplotlib
```

torch는 CUDA 빌드여야 한다. 확인:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import torch; x = torch.randn(1 << 20, dtype=torch.complex128, device='cuda'); print(torch.vdot(x, x))"
```

두 번째 줄이 오류 없이 실수에 가까운 복소수(약 2.1e6)를 출력하면 CUDA complex128이 동작한다.
`pip install torch`가 CPU 빌드를 설치했다면 <https://pytorch.org/get-started/locally/>의 CUDA 휠 명령을 쓴다.

## 2. 코드 구조

| 파일 | 내용 | Stage |
|---|---|---|
| `hhkl_toy/lattice.py` | 직사각 패치, 본드, 4색(A/B/C/D), $h_j$, dense $H$(작은 n 참조용) | 1 |
| `hhkl_toy/sv.py` | torch 상태벡터 커널: 닫힌 형태 본드 게이트, 색·장 슬롯, 생성자 overlap, $H$ 적용, Haar 상태 | 2 |
| `hhkl_toy/target.py` | Lanczos(공용), 스펙트럼 양 끝, Chebyshev $e^{\mp iHt}$, $K$ 자동 결정 | 3 |
| `hhkl_toy/ansatz.py` | P1 슬롯 열(병합), 파라미터 맵, S2^r·S4 초기값, CNOT 수·깊이 | 4 |
| `hhkl_toy/adjoint.py` | 차이벡터 손실, adjoint 기울기, drift 진단 | 5 |
| `hhkl_toy/certify.py` | $S=(W-W^\dagger)/2i$ Lanczos, $2\sin(w/4)$, 1·2 GPU 분담 | 6 |
| `hhkl_toy/train.py` | L-BFGS 구동, $J/J_0$ 스케일, 종료 규칙, 체크포인트, 2-GPU 상태 분담 | 7 |
| `hhkl_toy/block.py` | (patch, seed, t, L) 하나의 학습 → held-out → 인증 → JSON | 7 |
| `scripts/validate.py` | Stage 7 검증 (a)~(d) | 7 |
| `scripts/run_block.py` | 본 실험 구동 | 8, 10 |
| `scripts/bench.py` | 커널 시간·메모리 측정과 외삽 | 9 |
| `scripts/report.py` | `results/*.json` → 표 1(markdown), 그림 1 | 11 |

규칙: 큐비트 $i=r\cdot s_x+c$, 상태벡터 인덱스의 비트 $i$(little-endian). `5x6`은 5열 6행이다.
장치와 dtype: 기본 complex128, `--devices`로 `cpu`, `cuda`, `cuda:0 cuda:1`을 고른다(기본 `auto` = GPU 최대 2장, 없으면 CPU).

## 3. 단위 테스트 (Stage 1~7, 클라우드에서 통과 확인됨)

```bash
python -m pytest -q tests          # CPU, 약 7분 (n ≤ 12 dense 참조)
python tests/test_sv.py            # 파일 하나만: 마지막 줄 PASS
```

| 파일 | 검증 |
|---|---|
| `test_lattice.py` | 본드 49개, 색별 12/12/15/10, 색 안 서로소, $h_j$ 재현, dense $H$ = 독립 Pauli 합 (2×3, 3×2, 3×3) |
| `test_sv.py` | 본드 게이트·슬롯·$H$ 적용·overlap이 dense와 ≤1e-14 (n≤12), 역슬롯 복귀, 노름 보존, little-endian, chunk 경로 |
| `test_target.py` | 스펙트럼 양 끝 = `eigh`, Chebyshev = `eigh`/`expm` ≤1e-13, $K$ vs $K+5$ ≤1e-15 |
| `test_ansatz.py` | P1@S2·S4 = 색 그룹으로 직접 만든 S2·S4 ≤1e-14, 오차 차수 $t^3$·$t^5$, CNOT 표 (294/24/17 … 1326/108/77) |
| `test_adjoint.py` | adjoint vs 중앙차분 상대오차 ≤1e-6 (실측 ~1e-9), drift ≤1e-13, $J\approx5\times10^{-21}$에서 유효 숫자 유지 |
| `test_certify.py` | 2×3·3×3·3×4에서 정확 고유위상 $2\sin(w/4)$와 Lanczos ≤1e-6 (실측 ~1e-11), dense φ 탐색과 일치, 시작 벡터 2개 일치 |
| `test_train.py` | 7(a) 학습 ≤ 초기값, 7(b) S4 7.574e-6 → HVA 3.44e-6 (2.2배), 체크포인트 재시작, 2-장치 분담 = 1-장치 |

## 4. Stage 7: 검증 (데스크탑)

```bash
python scripts/validate.py --device cuda
```

| 항목 | 내용 | 통과 기준 |
|---|---|---|
| (a) | 2×3, 3×3, $t^\star$, $L$=2·4 학습과 인증 | 학습 후 손실·$\epsilon_\text{spec}$ 모두 ≤ Trotter 초기값 |
| (b) | 2×3, $t$=0.05, $L$=10, S4 초기값 | S4 ≈ 7.6e-6 (±5 %), 개선 배수 1.8~2.6 |
| (c) | 4×4(16), 4×5(20큐비트), $t^\star$, $L$=2, $M$=4 | held-out과 학습 손실 차이 ≤ 1 % |
| (d) | 3×4에서 CPU와 GPU 비교 | 회로·Chebyshev ≤1e-13, 기울기 상대 ≤1e-10 |

- 출력: `results/validate_cuda0.json`, 개별 실행은 `results/validate_cuda0/`.
- 예상 시간: 수 분(클라우드 CPU `--quick`은 55초). 메모리는 1 GB 미만.
- (c)가 실패하면 $M$=4가 부족하다는 뜻이다. `--M 8`로 다시 확인하고, 본 실험도 `--M 8`로 돌린다(메모리 예산 재계산 필요, PLAN.md §11).
- 클라우드 CPU 실행 결과(참고): `results/cloud/validate_cpu.json` — 6/6 PASS, (c) 4×4 차이 0.41 %.

## 5. Stage 8: 리허설 (데스크탑, 20~24큐비트)

```bash
python scripts/run_block.py --patch 4x5 --t 3.41e-3 --L 2 4 6 10
python scripts/bench.py --n 20 24            # 30큐비트 비용 외삽용
# 메모리가 되면: python scripts/run_block.py --patch 4x6 --t 3.41e-3 --L 2 4 6 10
```

- 통과 기준: 끝까지 오류 없이 실행되고 마지막 줄이 `PASS`, `results/4x5_s0_t0.00341_L{2,4,6,10}.json`이 완성됨(`"complete": true`).
- 벡터 하나: 20큐비트 16.8 MB, 24큐비트 268 MB. 데스크탑 GPU에서 메모리는 문제되지 않는다.
- `bench.py`가 출력하는 `train it ~…s`와 `cert 300 its ~…h`로 30큐비트 비용을 외삽한다(벡터 크기에 거의 비례).

## 6. Stage 9: H200 준비

처음 접속하면 확인할 것:

```bash
nvidia-smi                       # H200 ×2, 141 GB, 드라이버/CUDA 버전
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
python -c "import torch; x = torch.randn(1 << 20, dtype=torch.complex128, device='cuda:1'); print(torch.vdot(x, x))"
free -g                          # 호스트 RAM
df -h .                          # 디스크 (결과는 작다. 체크포인트도 MB 단위)
```

벤치마크:

```bash
python scripts/bench.py --n 26 28 30 --devices cuda:0 cuda:1
```

- 출력: `results/bench_{patch}_cuda0x2.json` (26=2x13, 28=4x7, 30=5x6). 마지막 줄에 크기별 학습 반복·인증 시간 추정.
- 통과 기준: 30큐비트에서 오류 없이 끝나고(complex128 CUDA 동작) 왕복 오차·drift ≤1e-12.
- 메모리 예산 (30큐비트, complex128 벡터 하나 17.2 GB, 대각 $D$ 8.6 GB, 코드 기준 계산):

| 작업 | GPU당 최대 벡터 | GPU당 메모리 |
|---|---|---|
| $V\psi$ 계산 (상태 2개째) | 캐시 1 + Chebyshev 4 | 약 95 GB |
| 학습 (GPU당 상태 2개) | $V\psi$ 2 + $\phi$, $\lambda$ | 약 77 GB |
| 인증, GPU 2장 분담 | GPU0: Lanczos 2 + Chebyshev 3 | 약 95 GB |
| 인증, GPU 1장 | Lanczos 2 + 회로/Chebyshev 4 | 약 112 GB |

  `bench.py`의 `*_GB` 값으로 실측해서 교체한다. 부족하면 `--no-diag`로 $D$(8.6 GB)를 끈다(대신 $H$ 적용이 느려짐).

## 7. Stage 10: 본 실험 (H200)

```bash
tmux new -s hva
python scripts/run_block.py --patch 5x6 --seed 0 --t 3.41e-3 --L 2 4 6 10 --devices cuda:0 cuda:1
# 중단되었으면 같은 명령에 --resume
```

- $L$을 2 → 4 → 6 → 10 순서로 처리한다. 각 $L$은 학습 → held-out 평가 → 인증(Trotter 초기값, 학습 후; 각각 시작 벡터 2개)이다.
- 결과: `results/5x6_s0_t0.00341_L{L}.json`(단계마다 다시 씀), 로그 `results/5x6_s0_t0.00341.log`,
  체크포인트 `results/ckpt/*.jsonl`(커밋하지 않음).
- `--resume`: 끝난 단계는 건너뛰고, 학습은 마지막 체크포인트의 $c$에서 새 L-BFGS로 이어간다. 완료된 조합은 `--force` 없이는 다시 하지 않는다.
- 통과 기준: 각 $L$의 JSON에 `"complete": true`, 인증 시작 벡터 일치(`start_vectors_agree`), drift ≤1e-10. 마지막 줄 `PASS`.
- 시간(PLAN.md 추정, Stage 9 실측으로 교체): 반복 1회 ~30초, 학습 수 시간, 인증 1회 ~1시간, 전체 1~3일.
- 인증 시간을 줄이려면 `--cert-seeds 1`(시작 벡터 1개)로 둘 수 있다. 기본은 2개다(PLAN.md §4.6).
- 보조 실험(선택): `--t 1.35e-3 --L <가장 좋은 L>`, 무질서 `--seed 1`, `--seed 2`.

## 8. Stage 11: 정리

```bash
python scripts/report.py --plot results/fig1.png
```

`results/report.md`에 표 1(PLAN.md §8)과 학습·인증 진단 표를 쓴다.

## 9. 결과 회수

- `results/*.json`, `results/*.log`, `results/report.md`를 커밋해서 push하거나, 내용을 세션에 붙여 넣는다.
- `results/ckpt/`와 큰 배열(`*.npy`, `*.pt`)은 `.gitignore`로 제외되어 있다.

## 10. 주의 사항 (클라우드 세션에서 확인한 것)

1. **plateau 종료 규칙이 너무 이르다.** PLAN.md §4.5의 규칙(최근 20회 동안 개선 < 0.1 %)은
   $t^\star$의 $L\ge4$에서 약 23회 만에 멈춘다. 학습 곡선은 5회 안에 $J/J_0\approx0.7$로 내려간 뒤
   약 150회 동안 거의 평평하다가 다시 크게 내려간다.

   | 패치 | $L$ | 20회 규칙으로 멈춘 $J/J_0$ (반복) | 계속했을 때 $J/J_0$ (반복) |
   |---|---|---|---|
   | 3×3 | 4 | 0.709 (23) | 0.199 (317, scipy 수렴) |
   | 3×3 | 6 | 0.709 (24) | 0.170 (500) |
   | 4×4 | 4 | 0.687 (23) | 0.198 (407) |

   기본값은 PLAN.md대로 20이다. `--plateau-window 0`이면 규칙을 끄고 500회 또는 scipy 수렴까지 간다
   (`--plateau-window 100`도 위 세 경우에서 조기 종료하지 않았다). 본 실험 전에 어느 쪽으로 할지 정해야 한다.
2. **$L$=10, $t^\star$는 float64 바닥에 가깝다.** 3×3에서 S4 초기값의 $J$는 약 3e-23($\epsilon_\text{rHS}\approx8\times10^{-12}$)이고
   L-BFGS가 10회 만에 멈췄다. 잔차 벡터의 반올림 오차(~1e-15)가 손실 잡음이 되는 수준이라, 이 점에서의 학습 이득은 정밀도로 제한된다.
   30큐비트에서는 $J$가 더 크므로(블록이 크다) 정도가 덜하겠지만 결과 해석 때 유의한다.
3. **인증의 위상 부호.** PLAN.md §4.6은 $W=e^{-i\hat\phi}U^\dagger V$로 쓰지만, $\hat\phi=\arg\langle V\psi|U\psi\rangle$(손실과 같은 정의)와
   일관된 것은 $W=e^{+i\hat\phi}U^\dagger V$다. 코드는 후자를 쓴다. $w$는 $\hat\phi$와 무관하므로($|\delta|<\pi/2$인 한) 결과는 같다.
4. **Haar 상태는 장치별로 다시 만든다.** 같은 seed라도 CPU와 CUDA의 난수는 다르다. 같은 장치 종류에서는 재현된다.
5. Lanczos의 Ritz 값은 안쪽에서 수렴하므로 $\epsilon_\text{spec}$은 과소평가 방향이다. 두 시작 벡터 중 **큰 값**을 보고한다.
6. 학습 후 인증값이 Trotter 초기값보다 나쁘면 초기값을 보고하고 `result.fallback_to_init = true`로 기록한다.
