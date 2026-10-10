# HHKL-VQA toy: 30큐비트 블록 HVA 학습과 spectral norm 측정

> 이 문서는 `toy` 코드를 작성할 **Claude 클라우드 세션**을 위한 인수인계 문서다.
> 이전 세션의 대화 없이도 이 문서만으로 구현을 시작할 수 있도록, 모든 결정과 수치와 함정을 적었다.
> 작성일: 2026-10-10.
>
> - 코드 저장소: <https://github.com/HHKL-VQA/toy>. 이 문서는 저장소 루트의 `PLAN.md`이다.
> - 이전 계산의 근거: <https://github.com/HHKL-VQA/vqa>의 `scripts/10–14_*.py`. 이 문서에서 `vqa/scripts/…`는 그 저장소를 가리킨다.
> - **역할 분담:** 코드 작성과 작은 단위 테스트는 클라우드 세션이 한다. 데스크탑과 H200 서버에서의 실행은 코드가 완성된 뒤 **사용자가 직접** 한다(§5).

---

## 0. 한 줄 요약

10×10 Heisenberg 모델을 HHKL로 분해했을 때 나오는 블록 하나(ℓ=3, **5×6 = 30큐비트**)에 대해 짧은 시간 진화 $e^{-iH_\square t^\star}$를 근사하는 **palindromic HVA 회로(P1)**를 statevector 손실, adjoint 미분, **L-BFGS**로 학습한다.
학습된 회로의 **spectral norm 오차** $\min_\phi\|U-e^{i\phi}V\|_\infty$를 Lanczos로 인증해서 보고한다.
Trotter와의 비용 비교는 **이번 범위가 아니다**. 다만 Trotter 초기값의 오차는 부산물로 함께 기록한다.

---

## 1. 배경

### 1.1 전체 문제

- 모델: 10×10 정사각 격자, open boundary
  $$H=\sum_{\langle ij\rangle}(X_iX_j+Y_iY_j+Z_iZ_j)+\sum_j h_jZ_j,\qquad h_j\sim U[-1,1]$$
- 조건: $n=100$, 총 시간 $T=100$, 목표 정확도 $\epsilon=0.01$ (spectral norm)
- 방법: HHKL 분해(Haah, Hastings, Kothari, Low 2018, arXiv:1801.03922)로 전체 진화를 작은 블록 진화의 곱으로 쪼갠다. 각 블록은 변분 회로(HVA)로 컴파일한다.
- **보고는 2D 결과만** 한다. 1D는 파이프라인 검증용일 뿐이다.

### 1.2 동작점: 왜 $t^\star=3.41\times10^{-3}$인가

$t$는 자유롭게 고르는 변수가 아니다. **Lieb–Robinson(LR) 오차 예산이 허용하는 최댓값**이다.

- 전체 비용은 $m(t)\times C_\text{blk}$이다. 블록 호출 수는 $m\propto1/t$이다.
- 작은 $t$에서 블록 하나의 비용 $C_\text{blk}$는 최소 층 수 근처에서 거의 변하지 않는다. 그러므로 $t$는 클수록 유리하다.
- $t$의 상한은 LR 오차가 정한다.

**계산식** (`vqa/scripts/13_tiling_comparison.py`의 `tiling()`)

- 경계 1개당 LR 오차: $\epsilon_{LR}(\ell,t)=\alpha\,(t\beta/(\ell+\gamma))^{\ell+\gamma}$, 계수는 $(\alpha,\beta,\gamma)=(0.175,7.9,0.95)$로 Haah Fig. 2의 1D 피팅값이다.
- 단위시간당 블록 수: $N_\text{pu}=\frac{2L}{s_x-\ell}\cdot\frac{2L}{s_y-\ell}$
- 단위시간당 잘리는 경계 수: $B=N_{cut,y}L+N_{dir,y}N_{cut,x}s_y$
- $t$는 $T\cdot B\cdot\epsilon_{LR}(\ell,t)/t=\epsilon/3$을 풀어서 얻는다.
- 블록 호출 수는 $m=N_\text{pu}T/t$, 블록당 허용 오차는 $\epsilon_\text{blk}=\epsilon/(3m)$이다.

**결과** (10×10, $T$=100, $\epsilon$=0.01, $\ell$=3; 2026-10-10 재계산)

| 블록 | LR 속도 | LR 예산 | $t$ | $m$ | $\epsilon_\text{blk}$ |
|---|---|---|---|---|---|
| **5×6** | 1D 피팅 | ε/3 | **3.41e-3** | 1.95e6 | **1.71e-9** |
| 5×6 | 1D 피팅 | 2ε/3 | 4.32e-3 | 1.54e6 | 2.16e-9 |
| 5×6 | ×2 (비관적) | ε/3 | 1.35e-3 | 4.94e6 | 6.75e-10 |
| 6×5 | 1D 피팅 | ε/3 | 3.50e-3 | 1.90e6 | 1.75e-9 |

- 주의: $(\alpha,\beta,\gamma)$는 1D 피팅을 2D로 외삽한 값이다. 2D는 LR 속도가 더 빠르므로 실제 $t$는 3.41e-3과 1.35e-3 사이일 것이다.
- 예산 배분이나 블록 방향을 바꿔도 $t$는 3~27%만 움직인다.

**이번 실험의 시간**

- 주 실험: $t^\star=3.41\times10^{-3}$, 목표 $\epsilon_\text{blk}=1.71\times10^{-9}$
- 보조(선택): $t=1.35\times10^{-3}$, 목표 $6.75\times10^{-10}$. 가장 좋은 $L$ 하나에만 적용한다.
- 이전 계획에 있던 $t$=1e-2와 5e-2는 HHKL에서 나온 값이 아니므로 **제외**한다.

### 1.3 이전 세션의 소규모 결과 (참고)

2×3 패치(6큐비트), $t$=0.05, $L$=10, S4 각도로 초기화한 palindromic HVA의 결과:

| 회로 | 기록된 값 | 올바른 값 |
|---|---|---|
| HVA | 7.07e-6 | 약 3.5e-6 |
| S4 | 1.51e-5 | 약 7.6e-6 |

- HVA가 S4보다 약 **2.1배** 좋았다.
- 기록된 절댓값은 잘못된 공식 $2\sin(\text{arc}/2)$로 계산되어 약 2배 크다. 올바른 공식은 $2\sin(\text{arc}/4)$이다(§4.6). 비율은 영향을 받지 않는다.
- 그 실험의 스크래치 코드는 **남아 있지 않다**. 새로 작성해야 한다.

**30큐비트 예상** (2×3 결과를 $t^5$로 외삽한 대략값): S4 1스텝($L$=10)이 이미 약 $10^{-10}$이라 $\epsilon_\text{blk}$ 아래일 수 있다. 그렇다면 핵심 질문은 이것이다.
**학습된 $L$=2, 4, 6 회로가 $1.71\times10^{-9}$에 도달하는가?**

---

## 2. 실험 목표

### 2.1 주 목표 (필수)

$t^\star=3.41\times10^{-3}$에서 P1 회로를 $L\in\{2,4,6,10\}$층으로 학습하고, 각 $L$마다 다음을 보고한다.

1. 인증된 spectral norm 오차 $\epsilon_\text{spec}$: **Trotter 초기값**과 **학습 후** 둘 다
2. 정규화 HS 오차 $\epsilon_\text{rHS}$: 학습 손실에서 나오는 값, 그리고 held-out 상태에서 잰 값
3. CNOT 수, CNOT 깊이, 파라미터 수
4. $\epsilon_\text{spec}/\epsilon_\text{blk}$: 1보다 작으면 블록 요구를 충족한다는 뜻
5. 학습 이력(반복별 손실), 종료 사유, 실행 시간

### 2.2 부 목표 (선택, 시간이 남으면)

- $t=1.35\times10^{-3}$에서 가장 좋은 $L$ 하나를 다시 학습하고 인증한다.
- 무질서 seed 1, 2로 핵심 점 하나를 다시 확인한다.

### 2.3 범위 밖 (이번에는 하지 않음)

- Trotter와의 같은 정확도 비용 비교, 최적화된 4차 공식(Blanes–Moan 등)
- Adam. 사용자 결정으로 제외했다(L-BFGS만 사용).
- 본드별 각도(P2), 각도 거울 고정(P3), 정방향 ansatz(F1), ZZ 제거(Z0)
- 최악 상태 증강(인증으로 찾은 최악 벡터를 학습 집합에 넣는 방법)
- barren plateau 진단. Trotter 초기값에서 출발하므로 우선순위가 낮다.

---

## 3. 블록 정의

### 3.1 격자와 인덱스

- 크기: $s_x=5$(열), $s_y=6$(행), $n=30$, open boundary
- 큐비트 인덱스: $i=r\cdot s_x+c$ (행 $r\in[0,6)$, 열 $c\in[0,5)$). `vqa/scripts/13·14`와 같은 규칙이다.
- 상태벡터에서 큐비트 $i$는 기저 인덱스의 비트 $i$이다(little-endian). $Z_i$의 고윳값은 비트가 0이면 +1, 1이면 −1이다.
- 무질서: `h = np.random.default_rng(seed).uniform(-1, 1, size=n)`, 주 실험은 seed=0이다. 기존 스크립트와 같은 방식이다.

### 3.2 본드 색칠 (4색, 같은 색의 본드끼리는 겹치지 않음)

| 색 | 정의 | 본드 수 |
|---|---|---|
| A | 가로 본드 $(r,c)$–$(r,c{+}1)$, $c$ 짝수 | 12 |
| B | 가로, $c$ 홀수 | 12 |
| C | 세로 본드 $(r,c)$–$(r{+}1,c)$, $r$ 짝수 | 15 |
| D | 세로, $r$ 홀수 | 10 |
| 합계 | | **49** |

장(field) 그룹은 F $=\sum_jh_jZ_j$이다.

### 3.3 블록 해밀토니안과 스펙트럼

$$H_\square=\sum_{b\in A\cup B\cup C\cup D}(XX+YY+ZZ)_b+\sum_jh_jZ_j$$

- 스펙트럼 범위의 대략값: $E_\text{min}\approx-72$ (작은 패치 정확 대각화 피팅 $\|H\|\approx0.619\cdot\text{본드}+1.3965\cdot n$을 외삽한 값).
- $E_\text{max}\ge49+|\sum_jh_j|$이다(완전 분극 상태).
- **실제 값은 Lanczos로 잰다**(§4.5).

---

## 4. 방법 명세

### 4.1 Ansatz P1 (palindromic, 색별 각도 공유)

**슬롯(게이트 묶음)**

- 색 슬롯 $g\in\{A,B,C,D\}$:
  $$G_g(\alpha,\zeta)=\prod_{b\in g}\exp\{-i[\alpha(XX+YY)_b+\zeta(ZZ)_b]\}$$
  - 같은 본드의 $XX+YY$와 $ZZ$는 교환한다. 같은 색의 본드끼리는 겹치지 않는다. 따라서 이 곱은 정확하고 순서와 무관하다.
  - 본드 하나는 3 CNOT이다. $XX$와 $YY$의 각도를 묶는 이유는 U(1) 대칭($\sum Z$ 보존)을 지키기 위해서다.
- 장 슬롯: $G_F(\varphi)=\exp(-i\varphi\sum_jh_jZ_j)$. CNOT이 없다.

**층과 순서**

- 1층은 한 방향의 sweep이다. 정방향은 A→B→C→D→F, 역방향은 F→D→C→B→A이다.
- $L$층 회로는 1층 정방향, 2층 역방향, 3층 정방향… 순서로 **번갈아** 쌓는다.

**병합**

- 정방향에서 역방향으로 넘어가는 경계에서는 F,F가 붙으므로 F 슬롯 하나로 합친다(CNOT 절약 없음).
- 역방향에서 정방향으로 넘어가는 경계에서는 A,A가 붙으므로 A 슬롯 하나로 합친다(12본드×3 = **36 CNOT 절약**).
- 병합한 뒤의 각 슬롯이 독립 파라미터를 갖는다. 슬롯 수는 $4L+1$이다.

**재매개화** $\theta=t\cdot c$

- 각 슬롯의 실제 각도는 $(\alpha,\zeta)=t\,(c_\alpha,c_\zeta)$, $\varphi=t\,c_\varphi$이다. 학습 변수는 $c$이다.
- 이렇게 하면 Trotter 초기값이 $t$와 무관한 O(1) 숫자가 되고, L-BFGS의 초기 헤시안 추정(단위행렬)도 적절한 크기가 된다.

**Trotter 초기값** (병합된 슬롯은 각도를 더한다)

- S2($\tau$)는 정방향 1층과 역방향 1층이다. 모든 슬롯의 계수는 $\tau/2$이다($c_\alpha=c_\zeta=c_\varphi$).
- $L=2r$: S2$(t/r)^r$. 각 층의 계수는 $1/(2r)$이다.
- $L=10$: S4$(t)$ = S2($ut$)·S2($ut$)·S2($(1-4u)t$)·S2($ut$)·S2($ut$)이고 $u=1/(4-4^{1/3})\approx0.41449$이다.
  - 층 쌍 $(2k-1,2k)$의 계수는 $s_k/2$이고, $s=(u,u,1-4u,u,u)$이다.
  - 즉 계수는 $0.20724$, 그리고 가운데 쌍만 $(1-4u)/2\approx-0.32899$이다.
- 이 그룹 분할(색 4개 + 장)은 `vqa/scripts/14`의 XYZ 분할과 **다르다**. 14번의 looseness 수치를 그대로 옮기면 안 된다.

**비용** (병합 반영)

- CNOT 수: $147L-36\lfloor(L-1)/2\rfloor$
- CNOT 깊이: $12L-3\lfloor(L-1)/2\rfloor$

| $L$ | 초기값 | CNOT | 깊이 | 파라미터 수 |
|---|---|---|---|---|
| 2 | S2 ($r$=1) | 294 | 24 | 17 |
| 4 | S2 ($r$=2) | 552 | 45 | 32 |
| 6 | S2 ($r$=3) | 810 | 66 | 47 |
| 10 | S4 | 1326 | 108 | 77 |

파라미터 수 = 2·(색 슬롯 $4L-\lfloor(L-1)/2\rfloor$) + (F 슬롯 $L-\lceil(L-1)/2\rceil$). 코드에서 다시 세서 확인한다.

### 4.2 본드 게이트의 닫힌 형태

기저 $\{|00\rangle,|01\rangle,|10\rangle,|11\rangle\}$에서

$$e^{-i[\alpha(XX+YY)+\zeta ZZ]}=
\begin{pmatrix}e^{-i\zeta}&&&\\&e^{i\zeta}\cos2\alpha&-ie^{i\zeta}\sin2\alpha&\\&-ie^{i\zeta}\sin2\alpha&e^{i\zeta}\cos2\alpha&\\&&&e^{-i\zeta}\end{pmatrix}$$

- 대각 위상 곱과 01↔10 부분공간의 2×2 회전으로 구현한다. 일반 4×4 행렬 곱이 필요 없다.
- 한 슬롯의 모든 $ZZ$ 성분과 장 성분은 대각이다. 그래서 한 번의 대각 곱으로 묶을 수도 있다(구현 선택).

### 4.3 목표 진화 $V=e^{-iH_\square t}$ (Chebyshev 전개)

$$H_s=\frac{H-c}{s},\qquad e^{-iHt}=e^{-ict}\Big[J_0(st)\,I+2\sum_{k\ge1}(-i)^kJ_k(st)\,T_k(H_s)\Big]$$

- $c=(E_\text{max}+E_\text{min})/2$, $s=(E_\text{max}-E_\text{min})/2\times1.02$ (2% 여유).
- $E_\text{min}$, $E_\text{max}$는 $H$에 Lanczos를 약 50스텝 돌려서 얻는다.
- **범위를 좁게 잡으면 전개가 발산한다.** 안전한 상한으로는 Pauli 1-norm $3\cdot49+\sum|h_j|$를 쓸 수 있지만 항 수가 늘어난다.
- 점화식: $T_{k+1}=2H_sT_k-T_{k-1}$
- 종료: $2|J_k(st)|<10^{-16}$이 될 때. $st\approx0.22$에서 약 9항이다.
- $V^\dagger$는 계수에서 $(-i)^k$를 $(+i)^k$로 바꾸고 $e^{-ict}$를 $e^{+ict}$로 바꾸면 된다.
- $H$ 적용:
  - 대각 부분 $D(x)=\sum_b z_iz_j+\sum_jh_jz_j$를 곱한다(미리 계산하면 float64로 8.6 GB, 아니면 즉석 계산).
  - 각 본드에서 $(XX+YY)$는 01↔10 진폭을 2배해서 교환한다.
- 정밀도: 반드시 complex128을 쓴다. 결과는 $K$와 $K+5$로 계산한 값의 차이로 확인한다.

### 4.4 손실과 adjoint 기울기

**Haar 상태**

- 정규화된 복소 가우시안 벡터이다. 학습용 $M=4$개, held-out 2개를 쓴다.
- 장치마다 정해진 seed로 **다시 생성**한다. $\psi$는 저장하지 않는다.
- $V\psi$는 학습 시작 전에 한 번만 계산해서 캐시한다.

**손실** (차이벡터 형태: 상쇄 오차가 없다)

$$r_m=U\psi_m-e^{i\hat\phi_m}V\psi_m,\quad \hat\phi_m=\arg\langle V\psi_m|U\psi_m\rangle,\quad J=\frac1{M}\sum_m\frac{\|r_m\|^2}{2}$$

- $\|r\|^2/2=1-|\langle V\psi|U\psi\rangle|$이다. 따라서 $1-|\text{ov}|^2$와 같은 목적함수이고 차이는 $(1+|\text{ov}|)/2$ 배수뿐이다.
- 하지만 $1-|\text{ov}|^2$를 직접 계산하면 $J\sim10^{-13}$ 근처에서 상쇄 오차로 정체한다(이전에 겪은 문제). **반드시 잔차 벡터로 계산한다.**
- rHS 오차는 $\epsilon_\text{rHS}\approx\sqrt{2J}$이다. Haar 평균 근사로, 보고용이다.

**Adjoint 기울기** (회로를 $U=G_P\cdots G_1$, $G_k=e^{-i\theta_kA_k}$로 쓴다)

```
φ = ψ;  for k=1..P: φ = G_k φ
r = φ − e^{iφ̂} Vψ;  J_m = ||r||²/2
λ = r
for k = P..1:
    g[k] = Im⟨λ|A_k|φ⟩          # = ∂(||r||²/2)/∂θ_k ;  ∂||r||²/∂θ_k = 2 Im⟨λ|A_k|φ⟩
    φ = G_k† φ;  λ = G_k† λ
∂J/∂c_j = t · Σ_{k가 c_j를 사용} g[k]   (M개 상태 평균)
```

- $\hat\phi$는 $\phi$에 대한 최적점이다. 따라서 $\theta$로 미분할 때 상수로 취급해도 된다(envelope 정리).
- 색 슬롯은 생성자가 둘이다. $c_\alpha$에는 $A=\sum_{b\in g}(XX+YY)_b$, $c_\zeta$에는 $A=\sum_{b\in g}ZZ_b$이다. 한 슬롯 안에서 모든 항이 교환하므로, 슬롯 적용 직후의 $\phi$에서 두 기울기를 함께 구한다.
- 장 슬롯의 생성자는 $A=\sum_jh_jZ_j$(대각)이다.
- **진단:** backward가 끝나면 $\phi$는 $\psi$로 돌아와야 한다. 다시 생성한 $\psi$와의 차이 $\|\phi-\psi\|$를 기록한다. 기대값은 $\lesssim10^{-13}$이다.
- 메모리: 상태 하나당 벡터 3~4개($\phi$, $\lambda$, 캐시된 $V\psi$, 임시). 계산량은 forward의 약 3배이다.

### 4.5 최적화: L-BFGS만 사용

- 구현은 `scipy.optimize.minimize(method="L-BFGS-B", jac=True)`를 권장한다. 파라미터가 100개 미만이라 CPU에서 충분하고, 함수와 기울기 계산만 GPU에서 한다.
- 목적함수는 $J/J_0$로 정규화한다($J_0$ = Trotter 초기값의 $J$). $J$가 $10^{-18}$ 규모이므로 scipy의 허용오차와 스케일을 맞추기 위해서다.
- `maxcor=50`: 파라미터가 적으므로 사실상 BFGS이다.
- `ftol`, `gtol`은 아주 작게 둔다(예: 1e-15, 1e-12). 실제 종료는 아래 사용자 규칙으로 한다.
- 종료 조건 (먼저 오는 쪽):
  - 최근 20회 반복 동안 개선이 0.1% 미만: $J_k>(1-10^{-3})J_{k-20}$
  - 반복 500회
  - scipy 자체 종료
- 체크포인트: 반복마다 파라미터 $c$, $J$, 시간을 JSON에 append한다. 재시작은 마지막 $c$에서 새 L-BFGS로 시작한다(곡률 이력은 잃는다).
- **안전장치:** 최종 결과는 학습 중 가장 좋았던 $c$로 한다. 인증에서 학습 후 결과가 Trotter 초기값보다 나쁘면 초기값을 보고하고 그 사실을 기록한다.

### 4.6 Spectral norm 인증

**측정량:** $\epsilon_\text{spec}=\min_\phi\|U-e^{i\phi}V\|_\infty$

**절차**

1. $W=e^{-i\hat\phi}U^\dagger V$로 둔다. $\hat\phi$는 학습 상태들의 위상 평균처럼 대략적인 값이면 충분하다.
2. Hermitian 연산자 $S=(W-W^\dagger)/(2i)$를 만든다. 고윳값은 $\sin\delta_j$이고, $\delta_j$는 $\hat\phi$ 기준 고유위상 편차이다.
3. Lanczos(3-벡터, 재직교화 없음)로 $S$의 **최대와 최소 고윳값을 동시에** 구한다. 이로부터 $\delta_\text{max}=\arcsin\lambda_\text{max}$, $\delta_\text{min}=\arcsin\lambda_\text{min}$을 얻는다.
4. $w=\delta_\text{max}-\delta_\text{min}$으로 두면
   $$\boxed{\epsilon_\text{spec}=2\sin(w/4)}$$
   이다. **$2\sin(w/2)$가 아니다**(이전 세션의 버그).
   - $w$는 $\hat\phi$와 무관하다. 그래서 위상 추정이 부정확해도 값이 정확하다. 조건은 $|\delta|<\pi/2$뿐이다.

**계산**

- 행렬-벡터 곱: $Sv=\big(e^{-i\hat\phi}U^\dagger(Vv)-e^{i\hat\phi}V^\dagger(Uv)\big)/(2i)$
- 1회 비용은 회로 2번과 Chebyshev 2번이다.
- GPU 두 장으로 나눌 수 있다. GPU0이 $U^\dagger(Vv)$, GPU1이 $V^\dagger(Uv)$를 계산한다.

**한계**

- Ritz 값은 참값의 **안쪽**에 있다. 그래서 $\epsilon$은 아래에서 위로 수렴한다(과소평가 방향).
- 대응:
  - ① 양 끝 Ritz 값의 변화가 10회 동안 상대 $10^{-4}$ 미만이 될 때까지 반복한다(최대 300회).
  - ② 서로 다른 시작 벡터 2개로 결과가 맞는지 확인한다.
  - ③ 20큐비트 이하에서 정확 대각화 값과 비교해 검증한다.

**정밀도**

- $Sv$는 $O(10^{-9})$ 크기인 두 벡터의 차이이다. complex128이면 상대오차가 약 $10^{-6}$으로 충분하다.
- **인증에서 complex64는 금지한다.**

**함께 기록할 값:** $\lambda_\text{max}$, $\lambda_\text{min}$, 반복 수, Ritz 값 이력, 두 시작 벡터의 결과

### 4.7 메모리 예산 (30큐비트, complex128 벡터 하나 = $2^{30}\times16$ B = 17.2 GB)

| 작업 | GPU당 벡터 수 | GPU당 메모리 | 비고 |
|---|---|---|---|
| 학습 (GPU당 상태 2개) | $V\psi$ 2 + 작업 3~4 | 약 86~103 GB | 상태별 순차 처리 시 더 적음 |
| $V\psi$ 계산 (Chebyshev) | 3~4 + 결과 | 약 69~86 GB | 시작 시 1회 |
| 인증 (GPU 분담) | $v$ 사본 + Chebyshev 3~4 + Lanczos 2 | 약 100~120 GB | 빠듯함. 제자리 연산 우선 |
| 대각 $D(x)$ 미리 계산 | float64 | 8.6 GB | 선택 |

H200은 141 GB이다. 실제 수치는 Stage 9 벤치마크로 교체한다.

---

## 5. 계산 자원과 실행 규칙

| 위치 | 누가 | 하는 일 |
|---|---|---|
| **Claude 클라우드 세션** (GPU 없음, RAM 수 GB로 가정) | Claude | 코드 작성, `tests/` 단위 테스트(Stage 1~6), 2×3 같은 아주 작은 학습으로 동작 확인. dense 참조는 **n≤12**, 상태벡터 테스트는 **n≤16**까지. 무거운 실행은 하지 않는다 |
| **데스크탑** (GPU 있음) | **사용자가 직접** | 코드가 완성된 뒤 Stage 7~8(≤24큐비트 검증과 리허설)을 실행 |
| **H200 서버** (H200 ×2, 각 141 GB, ssh 연결 예정) | **사용자가 직접** | Stage 9~10, 30큐비트 벤치마크와 본 실험 |

**클라우드 세션이 할 일**

- 클라우드 세션은 데스크탑이나 H200에 접근할 수 없다.
- 그래서 사용자가 **명령 한 줄**로 각 단계를 실행하고 결과를 가져올 수 있게 만든다.
- **`README.md`에 실행 가이드를 쓴다.**
  - 설치: `pip install -r requirements.txt`와 torch CUDA 빌드 확인
  - Stage 7~10의 명령어. 예: `python scripts/validate.py --device cuda`, `python scripts/run_block.py --patch 4x5 --t 3.41e-3 --L 2 4 6 10`
  - 예상 실행 시간과 메모리
  - 통과 기준
  - 결과 파일의 위치
- **각 스크립트는 다음을 지킨다.**
  - 통과/실패를 마지막 줄에 명확히 출력한다(`PASS`/`FAIL`과 수치).
  - 모든 결과를 `results/` JSON으로 남긴다.
  - 장시간 작업은 체크포인트에서 재시작할 수 있어야 한다(`--resume`).
- **실행 결과의 회수:** 사용자는 `results/`의 JSON과 로그를 저장소에 커밋하거나 세션에 붙여 넣어 전달한다. `.gitignore`에서 `results/`를 제외하지 않는다. 결과 JSON은 작으므로 커밋해도 된다. 큰 배열(`*.npy`, `*.pt`)은 제외한다.
- H200 서버를 처음 쓸 때 확인할 항목도 README에 적는다: `nvidia-smi`, torch 버전, CUDA에서 complex128 연산이 되는지, 호스트 RAM, 디스크.

---

## 6. 코드 목표

### 6.1 원칙

- **최소 구현.** 설정 시스템, 추상 클래스, 플러그인 구조를 만들지 않는다. 인자는 argparse로 받고, 결과는 JSON으로 남긴다.
- 의존성: Python 3, numpy, scipy, torch(CUDA). 작은 크기의 참조 계산은 numpy/scipy(dense `expm`, `eigh`)로 한다.
- 같은 코드가 CPU(작은 $n$)와 GPU(30큐비트)에서 모두 돌아야 한다. 장치와 dtype은 인자로 받는다.
- 학습 경로와 인증 경로는 **complex128을 기본**으로 한다.
- 이번 범위에서는 커스텀 CUDA 커널을 만들지 않는다. torch 텐서 연산(reshape/view와 슬라이스, 제자리 연산)으로 시작하고, 벤치마크에서 병목이 확인되면 그때 고려한다.

### 6.2 디렉터리 구조 (제안)

```
toy/
├─ PLAN.md               이 문서
├─ README.md             사용자용 실행 가이드 (§5)
├─ requirements.txt      numpy, scipy, torch
├─ hhkl_toy/
│  ├─ lattice.py         5×6 등 직사각 패치: 본드, 4색, h_j, dense H(작은 n 참조용)
│  ├─ sv.py              상태벡터 연산: 본드 게이트(닫힌 형태), 슬롯 적용과 역적용,
│  │                     생성자 A 적용, 대각 곱, H 적용, Haar 상태 생성
│  ├─ target.py          스펙트럼 양 끝(Lanczos), Chebyshev e^{∓iHt}, K 자동 결정
│  ├─ ansatz.py          P1 슬롯 열(병합 포함), 파라미터 맵, Trotter 초기값(S2^r, S4),
│  │                     CNOT 수와 깊이
│  ├─ adjoint.py         차이벡터 손실, adjoint 기울기, 역산 drift 진단
│  ├─ train.py           L-BFGS 구동(scipy), J/J0 스케일, 종료 규칙, 체크포인트,
│  │                     2-GPU 상태 분담
│  └─ certify.py         S=(W−W†)/2i Lanczos, 2 sin(w/4), 수렴 기록, 2-GPU 분담
├─ scripts/
│  ├─ validate.py        Stage 7 검증을 한 번에 실행 → PASS/FAIL 요약 + results/validate_*.json
│  ├─ bench.py           게이트·H 적용·forward·adjoint 시간과 메모리 측정 (n 인자)
│  ├─ run_block.py       한 조합 (patch, seed, t, L) → 학습 → 인증 → results JSON
│  └─ report.py          results/*.json → 요약 표(markdown)
├─ tests/
│  └─ test_*.py          §7의 검증 항목 (python 또는 pytest로 실행)
└─ results/
   └─ {patch}_s{seed}_t{t}_L{L}.json
```

### 6.3 결과 JSON 스키마 (조합 하나당 파일 하나)

```json
{
  "patch": "5x6", "n": 30, "seed": 0, "t": 3.41e-3, "eps_blk": 1.71e-9,
  "L": 6, "init": "S2^3", "n_params": 47, "cnot": 810, "cnot_depth": 66,
  "dtype": "complex128", "M_train": 4, "M_heldout": 2, "haar_seeds": [...],
  "spectral_range": [Emin, Emax], "cheb_terms": 9,
  "train": {"J0": ..., "J_final": ..., "iters": ..., "stop_reason": "...",
            "history": [[iter, J, wall_s], ...], "c_final": [...],
            "drift_max": ...},
  "heldout": {"J_init": ..., "J_final": ...},
  "certify": {"init":    {"eps_spec": ..., "lam_max": ..., "lam_min": ..., "iters": ..., "ritz_history": [...]},
              "trained": {"eps_spec": ..., "...": "..."},
              "start_vectors_agree": true},
  "eps_rhs": {"init": ..., "trained": ...},
  "env": {"host": "...", "gpu": "...", "torch": "...", "git": "...", "wall_total_s": ...}
}
```

---

## 7. 단계별 구축 목표와 통과 기준

각 단계는 **앞 단계를 통과해야** 다음으로 넘어간다.

- **Stage 1~6**과 Stage 7의 작은 동작 확인은 클라우드 세션이 직접 실행해서 통과시킨다. dense 참조는 n≤12만 쓴다.
- **Stage 7(전체)~10**은 사용자가 실행한다. 클라우드 세션은 그 실행에 필요한 스크립트와 README 가이드까지만 완성한다.

| Stage | 구축 내용 | 검증 (통과 기준) | 실행 |
|---|---|---|---|
| **1. 격자** | `lattice.py`: 본드 49개, 색별 개수(12/12/15/10), 겹침 없음, $h_j$ 재현성, dense $H$ | 색 안에서 본드가 서로소, 개수 일치. dense $H$를 Pauli 합으로 독립 구성한 것과 일치 (2×3, 3×3) | 클라우드 |
| **2. 상태벡터 커널** | `sv.py`: 본드 게이트, 슬롯과 역슬롯, 생성자 적용, 대각 곱, $H$ 적용, Haar 생성 | dense와 비교해 ≤1e-14 (n≤12). 슬롯을 적용한 뒤 역적용하면 원래로 돌아옴 ≤1e-14. 노름 보존. little-endian 확인 | 클라우드 |
| **3. 목표 진화** | `target.py`: Lanczos로 $E_\text{min}$·$E_\text{max}$, Chebyshev, $K$ 자동 결정 | `eigh`·`expm`과 비교해 ≤1e-13 (n≤12). 스펙트럼 양 끝이 `eigh`와 일치. $K$와 $K+5$의 차이 ≤1e-15 | 클라우드 |
| **4. Ansatz** | `ansatz.py`: 슬롯 열, 병합, 파라미터 맵, S2^r와 S4 초기값, CNOT 수와 깊이 | P1@S2·S4 각도 회로 = 색 그룹으로 직접 만든 S2·S4 (dense, ≤1e-14). §4.1의 CNOT 표 재현. 파라미터 수 확인 | 클라우드 |
| **5. 손실과 미분** | `adjoint.py` | adjoint 기울기와 중앙차분의 상대오차 ≤1e-6 (n≤12, 무작위 $c$와 Trotter $c$ 모두). drift ≤1e-13. 손실 계산에 상쇄 오차 없음($J\sim10^{-20}$에서도 유효 숫자 유지) | 클라우드 |
| **6. 인증** | `certify.py` | 2×3, 3×3, 3×4에서 정확 고유위상으로 구한 $2\sin(w/4)$와 Lanczos 결과의 상대오차 ≤1e-6. dense $\min_\phi\|U-e^{i\phi}V\|$(φ 격자 탐색)와도 일치. 시작 벡터 2개 일치 | 클라우드 |
| **7. 학습 구동** | `train.py`, `run_block.py`, `validate.py` | (a) 학습 결과 ≤ Trotter 초기값(손실과 인증값 모두). (b) 2×3, $t$=0.05, $L$=10, S4 초기값에서 HVA가 S4보다 약 2배 좋음 재현(올바른 공식으로 S4 약 7.6e-6). (c) held-out 손실과 학습 손실 차이 ≤1% (16~20큐비트, $M$=4가 충분한지 확인. 아니면 8로 늘림) | (a)(b)는 클라우드, `validate.py` 전체는 **사용자(데스크탑)** |
| **8. 리허설** | 4×5(20큐비트, 또는 4×6=24큐비트)에서 $t^\star$, $L\in\{2,4,6,10\}$ 전체 파이프라인 | 끝까지 오류 없이 실행. 시간과 메모리를 측정해 30큐비트 비용 외삽. 결과 JSON 완성 | **사용자(데스크탑)** |
| **9. H200 준비** | ssh 연결, 환경 점검, `bench.py`를 26/28/30큐비트로 실행 | complex128 CUDA 동작. 30큐비트 forward, adjoint, Chebyshev 시간 측정. §4.7 메모리 예산 확인 | **사용자(H200)** |
| **10. 본 실험** | 5×6, seed 0, $t^\star$, $L$ = 2 → 4 → 6 → 10 순서 | 각 $L$마다 결과 JSON (학습 + 인증 2회) | **사용자(H200)** |
| **11. 정리** | `report.py`로 요약 표 생성. 부 목표(§2.2)는 선택 | §8 산출물 | 결과를 받은 세션 |

**클라우드 세션의 완료 조건:**
- Stage 1~6의 테스트를 모두 통과한다.
- Stage 7 (a)(b)를 통과한다.
- `validate.py`, `bench.py`, `run_block.py`, `report.py`가 작은 크기(예: `--patch 2x3`, CPU)에서 끝까지 실행되고 JSON을 남긴다.
- README 실행 가이드를 완성한다.
- 위 내용을 GitHub `main`에 push한다.

**시간 감각** (추정, Stage 9 측정으로 교체)

- 30큐비트에서 반복 1회(상태 4개, 두 GPU 분담, 10층)는 약 30초이다.
- L-BFGS 수백 회에 수 시간, 인증 1회에 약 1시간이다.
- 본 실험(4개 $L$ × 학습 1회와 인증 2회)은 대략 **1~3일**이다.

---

## 8. 최종 산출물

**표 1** ($t^\star$, 5×6, seed 0)

| $L$ | CNOT | 깊이 | 파라미터 | $\epsilon_\text{spec}$ (Trotter 초기값) | $\epsilon_\text{spec}$ (학습 후) | 개선 배수 | $\epsilon_\text{spec}/\epsilon_\text{blk}$ | $\epsilon_\text{rHS}$ (학습 후) | spectral/rHS |
|---|---|---|---|---|---|---|---|---|---|

**그림 1:** CNOT(또는 $L$)에 대한 $\epsilon_\text{spec}$ 로그 그래프. Trotter 초기값과 학습 후 두 곡선을 그리고, $\epsilon_\text{blk}$=1.71e-9에 수평선을 긋는다.

**한 문단 결론**

- 학습된 P1이 $\epsilon_\text{blk}$에 도달하는 최소 $L$
- 같은 $L$에서 Trotter 초기값 대비 개선 배수
- 30큐비트에서의 spectral/rHS 비율. 6큐비트에서는 약 5.5배였다.

---

## 9. 이전 세션에서 얻은 교훈 (함정)

1. **상쇄 오차.** $1-|\text{ov}|^2$를 직접 계산하면 $J\approx4\times10^{-13}$에서 정체했다(spectral 약 3.5e-6). 차이벡터 손실로 바꾸자 9.2e-7까지 내려갔다. 기울기도 잔차 $\lambda$에서 출발해 계산해야 한다.
2. **spectral 공식.** $2\sin(\text{arc}/2)$는 틀렸다. $2\sin(\text{arc}/4)$가 맞다. 이전 절댓값은 약 2배 과대평가였다.
3. **위상을 고정한 Lanczos 1회는 위험하다.** $\|U-e^{i\hat\phi}V\|$를 직접 재면 최대 2배까지 과대평가될 수 있다. §4.6의 양 끝 고윳값 방식(원호 폭)을 쓴다.
4. **정방향만 쌓는 ansatz는 S2와 S4를 포함하지 않는다.** 정방향만 적용하는 코드로 만든 "S2"는 실제로는 1차 공식이었다(이전에 잘못된 324배 주장을 철회한 원인). palindromic 순서는 Stage 4에서 dense 비교로 반드시 확인한다.
5. **초기값에 잡음을 넣지 않는다.** S4 초기값에 1e-6 잡음을 넣자 결과가 초기값보다 1만 배 나빠졌다. Trotter 각도에서 그대로 시작한다.
6. **재매개화 없이 작은 $t$에서 학습하면 스텝 크기가 어긋난다.** $\theta=t\cdot c$를 쓴다.
7. **PyTorch autograd는 30큐비트에서 쓸 수 없다.** 중간상태를 게이트마다 저장하므로(각 17 GB) 메모리가 부족하다. adjoint 미분만 쓴다.
8. **클라우드 세션에서 무거운 계산을 돌리지 않는다.** 20큐비트 이상과 긴 학습은 사용자가 데스크탑이나 H200에서 실행한다(§5).

---

## 10. 결정 기록

| 날짜 | 결정 | 이유 |
|---|---|---|
| 2026-09 | 블록 ℓ=3, 5×6(30큐비트) | H200 한 장에 complex128 상태벡터와 작업공간이 들어가는 최대 실용 크기 |
| 2026-09 | palindromic ansatz | S2·S4를 포함하고, 시간가역 구조라 오차 전개에 홀수 차수만 남음. 작은 $t$에서 정방향보다 크게 유리 |
| 2026-09 | $XX$와 $YY$ 각도를 묶고 $ZZ$ 포함 (3 CNOT) | U(1) 대칭 보존. Heisenberg 본드를 정확히 표현하는 최소 CNOT 수가 3 |
| 2026-09 | statevector 차이벡터 손실 (L_HST 대신) | 30큐비트에서 L_HST는 정확히 계산할 수 없음. Haar 평균이 L_HST에 비례($d/(d+1)$ 배) |
| 2026-10-10 | **P1만 사용** | 우선 30큐비트 학습 오차를 확인하는 것이 목표 |
| 2026-10-10 | **Adam 제외, L-BFGS만 사용** | 사용자 결정. Trotter 초기값이 이미 해 근처 |
| 2026-10-10 | **$t^\star$=3.41e-3 단일 시간** (보조 1.35e-3) | HHKL LR 예산에서 유도한 값. 1e-2와 5e-2는 근거가 없어 제외 |
| 2026-10-10 | **Trotter 비교는 나중으로** | 우선 학습된 30큐비트 회로의 spectral 오차를 확인 |
| 2026-10-10 | **코드는 클라우드 세션이 작성, 실행은 사용자가 직접** | 저장소는 github.com/HHKL-VQA/toy. 데스크탑과 H200 실행은 코드 완성 후 사용자가 직접 |

---

## 11. 미결 사항 (새 세션에서 확인)

- H200 서버의 ssh 주소, 환경(CUDA·torch 버전), 호스트 RAM. 사용자에게 확인한다.
- Stage 7(c)에서 held-out 차이가 1%를 넘으면 $M$을 8로 늘린다. 이 경우 메모리 예산을 다시 계산한다.
- 부 목표(§2.2)를 할지는 본 실험 결과를 보고 사용자와 정한다.
