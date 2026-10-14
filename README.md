# toy: 30큐비트 블록 HVA 학습과 spectral norm 인증

10×10 Heisenberg 모델의 HHKL 블록 하나(5×6 = 30큐비트, ℓ=3)에서 $e^{-iH_\square t^\star}$를 근사하는
palindromic HVA(P1)를 statevector 손실 + adjoint 미분 + (헤시안 전처리한) L-BFGS로 학습하고,
$\epsilon_\text{spec}=\min_\phi\|U-e^{i\phi}V\|$를 Lanczos로 인증한다. 설계와 수치의 근거는 [`PLAN.md`](PLAN.md)에 있다.

이 문서는 **사용자가 데스크탑과 H200에서 직접 실행하기 위한 가이드**다(PLAN.md §5).
모든 스크립트는 마지막 줄에 `PASS`/`FAIL`과 핵심 수치를 출력하고, 결과를 `results/`에 JSON으로 남긴다.

---

## 1. 설치

```bash
pip install -r requirements.txt          # numpy, scipy, torch, pytest
# 그림(선택): pip install matplotlib
```

torch는 **CUDA 빌드**여야 한다.

- **Linux:** `pip install torch`는 CUDA 13.0 빌드를 설치한다. 드라이버 R580 이상이 필요하다.
- **Windows:** PyPI의 torch는 CPU 빌드다. CUDA 휠을 따로 설치한다.
- **드라이버가 낮으면:** `nvidia-smi` 오른쪽 위의 "CUDA Version" 이하의 휠을 고른다.

```bash
# Windows, 또는 드라이버가 낮을 때 (CUDA 12.8 / 12.6 휠)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

확인:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability(), torch.cuda.get_arch_list())"
python -c "import torch; x = torch.randn(1 << 20, dtype=torch.complex128, device='cuda'); print(torch.vdot(x, x))"
```

- RTX 2060은 compute capability `(7, 5)`다. arch 목록에 `sm_75`가 있어야 한다. CUDA 13.0 PyPI 빌드(2.14)에서 `sm_75`가 들어 있는 것을 확인했다.
- H200은 `(9, 0)`, 즉 `sm_90`이다.
- 두 번째 줄이 오류 없이 약 2.1e6의 복소수를 출력하면 CUDA complex128이 동작한다.

## 2. 데스크탑 (RTX 2060 6 GB, Ryzen 5 5600, RAM 24 GB)으로 가능한 범위

**메모리.** 2026-10-10에 실측한 최대 메모리를 상태벡터 개수로 나타냈다(PLAN.md §4.7). GPU 한 장, $M$=4, 대각 $D$ 포함이다.

| 단계 | 벡터 수 | 20큐비트 (4×5) | 24큐비트 (4×6) | 25큐비트 (5×5) | 26큐비트 |
|---|---|---|---|---|---|
| 학습 ($V\psi$ 4개 캐시 포함) | 7.3 | 0.12 GB | 2.0 GB | 3.9 GB | 7.8 GB ✗ |
| 인증 | 6.2 | 0.10 GB | 1.7 GB | 3.3 GB | 6.7 GB ✗ |

- CUDA context(약 0.3~0.5 GB)와, 화면 출력에 쓰는 메모리를 빼면 쓸 수 있는 양은 약 5~5.5 GB다.
- 따라서 **25큐비트(5×5)까지 가능**하고 26큐비트 이상은 불가능하다.
- 호스트 RAM 24 GB는 어느 단계에서도 제약이 아니다. 단위 테스트의 dense 참조가 가장 많이 쓰고, 그것도 약 1~2 GB다.

**속도.** FP64 연산 속도는 Turing에서 FP32의 1/32(약 0.2 TFLOPS)다. 하지만 이 코드는 메모리 대역폭(336 GB/s)이 병목이라 실사용에는 지장이 없다. 아래 시간은 대역폭 모델(유효 약 250 GB/s)로 어림한 값이다. 실제 값은 `python scripts/bench.py --n 20 24 --devices cuda`로 먼저 확인한다.

| 작업 | 20큐비트 | 24큐비트 | 25큐비트 |
|---|---|---|---|
| `validate.py` 전체 | 수 분 | – | – |
| 학습 기울기 1회 ($L$=10, $M$=4) | 약 2초 | 약 25초 | 약 50초 |
| `run_block.py --L 2 4 6 10` 전체 (학습 + 인증) | 15~30분 | 약 2~3시간 | 약 5~6시간 |

CPU(Ryzen 5 5600)만 쓰면 24큐비트에서 GPU보다 약 8배 느리다. 그래서 CPU는 20큐비트 이하의 확인용으로만 권한다.

## 3. 코드 구조

| 파일 | 내용 | Stage |
|---|---|---|
| `hhkl_toy/lattice.py` | 직사각 패치, 본드, 4색(A/B/C/D), $h_j$, dense $H$(작은 n 참조용) | 1 |
| `hhkl_toy/sv.py` | torch 상태벡터 커널: 닫힌 형태 본드 게이트, 색·장 슬롯, 생성자 overlap, $H$ 적용, Haar 상태 | 2 |
| `hhkl_toy/target.py` | Lanczos(공용), 스펙트럼 양 끝, Chebyshev $e^{\mp iHt}$(Clenshaw 합), $K$ 자동 결정 | 3 |
| `hhkl_toy/ansatz.py` | P1 슬롯 열(병합), 파라미터 맵, S2^r·S4 초기값, CNOT 수·깊이 | 4 |
| `hhkl_toy/adjoint.py` | 차이벡터 손실, adjoint 기울기, drift 진단 | 5 |
| `hhkl_toy/certify.py` | $S=(W-W^\dagger)/2i$ Lanczos, $2\sin(w/4)$, 1·2 GPU 분담 | 6 |
| `hhkl_toy/train.py` | 헤시안 전처리 L-BFGS, $J/J_0$ 스케일, 종료 규칙, 체크포인트, 2-GPU 상태 분담 | 7 |
| `hhkl_toy/block.py` | (patch, seed, t, L) 하나의 학습 → held-out → 인증 → JSON | 7 |
| `scripts/validate.py` | Stage 7 검증 (a)~(d) | 7 |
| `scripts/check_M.py` | 학습 상태 수 $M$이 충분한지 확인 | 7 |
| `scripts/run_block.py` | 본 실험 구동 | 8, 10 |
| `scripts/bench.py` | 커널 시간·메모리 측정과 외삽 | 9 |
| `scripts/report.py` | `results/*.json` → 표 1(markdown), 그림 1 | 11 |

규칙: 큐비트 $i=r\cdot s_x+c$, 상태벡터 인덱스의 비트 $i$(little-endian). `5x6`은 5열 6행이다.
장치와 dtype: 기본 complex128, `--devices`로 `cpu`, `cuda`, `cuda:0 cuda:1`을 고른다(기본 `auto` = GPU 최대 2장, 없으면 CPU).

## 4. 단위 테스트 (클라우드에서 통과 확인됨)

```bash
python -m pytest -q tests          # CPU, 약 4~7분 (n ≤ 12 dense 참조)
python tests/test_sv.py            # 파일 하나만: 마지막 줄 PASS
```

| 파일 | 검증 |
|---|---|
| `test_lattice.py` | 본드 49개, 색별 12/12/15/10, 색 안 서로소, $h_j$ 재현, dense $H$ = 독립 Pauli 합 (2×3, 3×2, 3×3) |
| `test_sv.py` | 본드 게이트·슬롯·$H$ 적용·overlap이 dense와 ≤1e-14 (n≤12), 역슬롯 복귀, 노름 보존, little-endian, chunk 경로 |
| `test_target.py` | 스펙트럼 양 끝 = `eigh`, Chebyshev = `eigh`/`expm` ≤1e-13, $K$ vs $K+5$ ≤1e-15, 노름 보존 |
| `test_ansatz.py` | P1@S2·S4 = 색 그룹으로 직접 만든 S2·S4 ≤1e-14, 오차 차수 $t^3$·$t^5$, CNOT 표 (294/24/17 … 1326/108/77) |
| `test_adjoint.py` | adjoint vs 중앙차분 상대오차 ≤1e-6 (실측 ~1e-9), drift ≤1e-13, $J\approx5\times10^{-21}$에서 유효 숫자 유지 |
| `test_certify.py` | 정확 고유위상 $2\sin(w/4)$와 Lanczos ≤1e-6 (실측 ~1e-11), dense φ 탐색과 일치, 시작 벡터 2개 일치, 위상 부호·중심 이동에 견고 |
| `test_train.py` | 7(a), 7(b), 유한차분 헤시안·전처리, 전처리로 정체 구간이 사라짐, 체크포인트 재시작, 2-장치 분담 = 1-장치 |

## 5. Stage 7: 검증 (데스크탑)

```bash
python scripts/validate.py --device cuda
```

| 항목 | 내용 | 통과 기준 |
|---|---|---|
| (a) | 2×3, 3×3, $t^\star$, $L$=2·4 학습과 인증 | 학습 후 손실·$\epsilon_\text{spec}$ 모두 ≤ Trotter 초기값 |
| (b) | 2×3, $t$=0.05, $L$=10, S4 초기값 | S4 ≈ 7.6e-6 (±5 %), 개선 배수 ≥ 1.8 (이전 세션 2.1배; 전처리 학습은 3.9배) |
| (c) | 4×4(16), 4×5(20큐비트), $t^\star$, $L$=2, $M$=4 | held-out과 학습 손실 차이 ≤ 1 % |
| (d) | 3×4에서 CPU와 GPU 비교 | 회로·Chebyshev ≤1e-13, 기울기 상대 ≤1e-10 |

- 출력: `results/validate_cuda0.json`, 개별 실행은 `results/validate_cuda0/`.
- 메모리는 1 GB 미만이다. 클라우드 CPU의 `--quick`(4×4만)은 약 1.5분 걸렸다.
- 클라우드 결과(참고): `results/cloud/validate_cpu.json`. 6/6 PASS였다.
  - (b): 7.57e-6 → 1.95e-6
  - (c): 4×4 차이 0.41 %
- (선택) $M$ 확인을 데스크탑에서 더 큰 크기로 하려면 다음을 실행한다. 클라우드 결과는 §11.3에 있다.

  ```bash
  python scripts/check_M.py --patch 4x5 --L 4 --M 1 4 8 --sets 2 --heldout 8 --devices cuda
  ```

## 6. Stage 8: 리허설 (데스크탑, 20~25큐비트)

```bash
python scripts/bench.py --n 20 24 --devices cuda           # 먼저: 시간·메모리 실측
python scripts/run_block.py --patch 4x5 --t 3.41e-3 --L 2 4 6 10
python scripts/run_block.py --patch 4x6 --t 3.41e-3 --L 2 4 6 10    # 24큐비트, 약 2~3시간
# 선택: 25큐비트 python scripts/run_block.py --patch 5x5 --t 3.41e-3 --L 2 4 6 10
```

- 통과 기준: 끝까지 오류 없이 실행되고 마지막 줄이 `PASS`, `results/4x5_s0_t0.00341_L{2,4,6,10}.json`이 완성됨(`"complete": true`).
- 기본 학습은 자기 헤시안으로 전처리한다(`--precond own`). 기울기 평가가 $P+1$회 더 든다($L$=2·4·6·10에서 18·33·48·78회).
- 작은 패치의 헤시안을 옮겨 쓸 수 있다(`--precond 3x4`). 4×4에서 3×3·3×4 헤시안으로 같은 결과가 나오는 것을 확인했다.
- `bench.py`의 `train it ~…s`와 `cert 300 its ~…h`로 30큐비트 비용을 외삽한다. 벡터 크기에 거의 비례한다.
- **2026-10-11 결과:** 4×5·4×6 완료. 분석은 [`results/analysis_stage7_8.md`](results/analysis_stage7_8.md), 요약은 PLAN.md §1.4에 있다.
- **2026-10-14부터 $L$=6의 기본 초기값은 Y4(3단 4차)이다.** 그 전의 $L$=6 결과(S2³)는 `..._L6_S23.json`으로 이름을 바꿔 남겼다.
- **2026-10-14 $L$=6(Y4) 재실행 완료:** 학습 후 4×5 1.41e-9, 4×6 1.72e-9로 예측과 1 % 이내였다. `validate.py --device cuda`도 8개 모두 통과했다.
  - 같은 $L$을 다른 초기값으로 다시 돌릴 때는 `--resume`을 쓰지 않는다. 같은 이름에 다른 초기값의 결과 파일이 있으면 실행이 멈추고 알려 준다.
- $\epsilon_\text{blk}$(1.71e-9)은 10×10 분해의 **5×6 블록** 예산이다. 작은 패치 결과는 본드 수로 환산한 5×6 추정치($\epsilon\times49/\text{본드 수}$)로 비교한다. `report.py` 표의 `5x6 est. / eps_blk` 열과 `run_block.py` 마지막 줄이 이 값이다. 그림 1의 $\epsilon_\text{blk}$ 선은 5×6 결과에만 그린다.

## 7. Stage 9: H200 준비

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

- 출력: `results/bench_{patch}_cuda0x2.json` (26=2x13, 28=4x7, 30=5x6). 마지막 줄에 크기별 학습 반복·인증 시간 추정이 나온다.
- 통과 기준: 30큐비트에서 오류 없이 끝나고(complex128 CUDA 동작) 왕복 오차·drift ≤1e-12.

**메모리 예산** (30큐비트, 벡터 하나 17.2 GB. 실측 벡터 수 × 17.2 GB, PLAN.md §4.7)

| 작업 | GPU 2장 (GPU당) | GPU 1장 |
|---|---|---|
| 스펙트럼 Lanczos | 약 69 GB | 약 69 GB |
| 학습 ($M$=4) | 약 90 GB | 약 126 GB |
| 인증 | GPU0 약 77 GB | 약 107 GB |

`bench.py`의 `*_GB` 값으로 실측해서 확인한다. 부족하면 `--no-diag`로 대각 $D$(8.6 GB)를 끈다. 대신 $H$ 적용이 느려진다.

## 8. Stage 10: 본 실험 (H200)

```bash
tmux new -s hva
python scripts/run_block.py --patch 5x6 --seed 0 --t 3.41e-3 --L 2 4 6 10 --precond 5x4 --devices cuda:0 cuda:1
# 중단되었으면 같은 명령에 --resume
```

- `--precond 5x4`: 헤시안을 20큐비트 패치 5×4에서 구해 옮겨 쓴다. 몇 분이면 되고, 자기 헤시안(`--precond own`, 기본값)보다 약 2.6시간 빠르다.
  - 5×4는 5×6과 열 수가 같아서 색별 본드 수 비율(A:B:C:D)이 가장 비슷하다. 4×5도 쓸 수 있다.
  - 옮겨 쓴 헤시안으로도 전처리 후 조건수가 2~3이고(자기 헤시안 1, 전처리 없으면 1e6~1e9), 결과는 같다(PLAN.md §4.5).
- 초기값은 $L$=2·4가 S2$^{L/2}$, $L$=6이 Y4, $L$=10이 S4이다(PLAN.md §4.1). $t$는 $t^\star$로 고정한다(2026-10-14 결정).

- $L$을 2 → 4 → 6 → 10 순서로 처리한다. 각 $L$은 다음 순서로 진행한다.
  1. 헤시안 전처리
  2. 학습
  3. held-out 평가
  4. 인증: Trotter 초기값과 학습 후 회로 각각, 시작 벡터 2개
- 결과 파일:
  - `results/5x6_s0_t0.00341_L{L}.json`(단계마다 다시 씀)
  - 로그 `results/5x6_s0_t0.00341.log`
  - 체크포인트 `results/ckpt/*.jsonl`, 헤시안 캐시 `results/ckpt/*.hess_*.json`(둘 다 커밋하지 않음)
- `--resume`: 끝난 단계는 건너뛴다. 헤시안은 캐시된 열부터 이어서 계산하고, 학습은 마지막 체크포인트의 $c$에서 같은 전처리로 새 L-BFGS를 시작한다. 완료된 조합은 `--force` 없이는 다시 하지 않는다.
- 통과 기준: 각 $L$의 JSON에 `"complete": true`, 인증 시작 벡터 일치(`start_vectors_agree`), drift ≤1e-10, 마지막 줄 `PASS`.
- 종료 규칙(PLAN.md §4.5): 최근 30회 동안 개선 < 0.1 %, 500회, scipy 자체 종료 중 먼저 오는 것. `--precond none`이면 정체 창이 100회다.
- 시간: 24큐비트 실측에서 외삽하면 4개 $L$에 약 5시간(`--precond 5x4`), 자기 헤시안이면 약 7.5시간이다. Stage 9 실측으로 확인한다.
- 예측(5×6, 학습 후 $\epsilon_\text{spec}$): $L$=2 7.8e-6, $L$=4 1.0e-6, $L$=6(Y4) 2.2e-9($\epsilon_\text{blk}$의 1.3배), $L$=10 9.6e-11. 2배 이상 다르면 그 자체가 중요한 결과다.
- 인증 시간을 줄이려면 `--cert-seeds 1`(시작 벡터 1개)로 둘 수 있다. 기본은 2개다(PLAN.md §4.6).
- 보조 실험(선택): `--t 1.35e-3 --L <가장 좋은 L>`, 무질서 `--seed 1`, `--seed 2`.

## 9. Stage 11: 정리

```bash
python scripts/report.py --plot results/fig1.png
```

`results/report.md`에 표 1(PLAN.md §8)과 학습·인증 진단 표를 쓴다.

## 10. 결과 회수

- `results/*.json`, `results/*.log`, `results/report.md`를 커밋해서 push하거나, 내용을 세션에 붙여 넣는다.
- `ckpt/` 디렉터리와 큰 배열(`*.npy`, `*.pt`)은 `.gitignore`로 제외되어 있다.

## 11. 클라우드 세션에서 확인한 것

1. **학습이 오래 정체하는 원인은 헤시안 조건수다. 전처리로 해결했다(PLAN.md §4.5).**
   - Trotter 점의 헤시안은 고윳값이 약 $10^{10}$, $10^5$, $10^1$, 0인 층으로 나뉜다. 조건수가 약 $10^{10}$이다.
   - 그래서 그냥 L-BFGS는 $J/J_0\approx0.7$에서 약 150회 머물렀다. 이전 정체 규칙(20회)은 여기서 멈춰서 $J$를 3~4배 높게 남겼다.
   - 헤시안 전처리를 하면 3×3·3×4·4×4, $L$=4에서 6~10회 만에 수렴값의 1 % 이내에 도달했다. 그냥 L-BFGS는 300~450회가 걸렸다.
   - $L$=6에서는 더 낮은 극소에 갔다(0.170 → 0.143).
   - 학습 후 $\epsilon_\text{spec}$도 3×3, $L$=4, $t^\star$에서 4.9e-7에서 2.3e-7로 좋아졌다.
2. **인증 위상 부호를 PLAN.md에서 고쳤다(§4.6).**
   - 코드는 처음부터 $W=e^{+i\hat\phi}U^\dagger V$를 썼다. 이것이 손실의 $\hat\phi$와 맞는 부호다.
   - PLAN의 $e^{-i\hat\phi}$는 중심을 $-2\hat\phi$로 옮긴다. 그 이동이 $\pi/2$ 근처에 오면 $\arcsin$이 접혀서 값이 틀린다. 시험에서 45 % 오차가 났다.
   - Lanczos 수렴 판정도 $|\lambda|$가 아니라 폭 기준으로 바꿨다. 이제 중심이 어디에 있든 정밀도가 같다(시험 상대오차 ~1e-8).
   - 실제 $\hat\phi$는 약 1e-8이라서 이전 결과에는 영향이 없었다.
3. **$M$=4로 충분하다(PLAN.md §4.4, `results/cloud/check_M_*`).**
   - 학습된 회로를 학습 상태와 무관한 양으로 비교했다(정확한 Haar 평균 손실, 인증 $\epsilon$).
   - $M$=4와 $M$=8~16의 차이는 9큐비트에서 1.1 %/0.4 %, 12큐비트에서 0.09 %/0.05 %, 16큐비트에서 0.01 %/0.01 %였다.
   - 상태별 퍼짐은 $2^{-n/2}$로 줄어 30큐비트에서 약 5e-5가 된다.
   - **인증은 학습 상태를 쓰지 않는다.** $U$, $V$, 별도 seed의 Lanczos 시작 벡터만 쓴다. 학습 상태에서 가져오는 위상 $\hat\phi$는 결과($w$)에 영향을 주지 않는다. 따라서 $M$은 인증의 타당성과 무관하고, 회로 품질에만 작게 영향을 준다.
4. **메모리를 실측해서 줄였다(PLAN.md §4.7).** 인증이 9.2벡터에서 6.2벡터로 줄었다. 원인은 Lanczos 시작 벡터 보유, `vector_norm` 임시 배열, Chebyshev 버퍼였다.
5. **$L$=10, $t^\star$는 float64 바닥에 가깝다.**
   - 3×3에서 S4 초기값의 $J$는 약 3e-23($\epsilon_\text{rHS}\approx8\times10^{-12}$)이다.
   - 잔차의 반올림 오차(~1e-15)가 손실 잡음이 되는 수준이라, 학습 이득은 정밀도로 제한된다.
   - 30큐비트에서는 $J$가 더 크므로 덜하겠지만, 결과를 해석할 때 유의한다.
6. **작은 $n$에서 $L$=10은 학습 상태에 과적합한다.** 2×3(6큐비트, $d$=64)에서 held-out 차이가 36 %였다. 파라미터 77개에 비해 $d$가 작기 때문이다. 16큐비트 이상에서는 1 % 이내다.
7. **Haar 상태는 장치별로 다시 만든다.** 같은 seed라도 CPU와 CUDA의 난수는 다르다. 같은 장치 종류에서는 재현된다.
8. Lanczos의 Ritz 값은 안쪽에서 수렴하므로 $\epsilon_\text{spec}$은 과소평가 방향이다. 두 시작 벡터 중 **큰 값**을 보고한다.
9. 학습 후 인증값이 Trotter 초기값보다 나쁘면 초기값을 보고하고 `result.fallback_to_init = true`로 기록한다.
