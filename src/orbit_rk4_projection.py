"""
Прогнозирование невозмущённого движения ИСЗ методом Рунге–Кутты 4-го порядка
со стабилизацией по первым интегралам (проекционный метод).

Идея
----
Невозмущённое (двухтельное) движение спутника описывается системой
    r'' = -mu * r / |r|^3 ,   r = (x, y, z).
Её можно проинтегрировать классическим RK4. Но при большом шаге численное
решение медленно "уползает" с истинной орбиты (растёт ошибка энергии и
момента импульса, орбита раскручивается/затухает).

Чтобы этого не происходило, используются ПЕРВЫЕ ИНТЕГРАЛЫ задачи Кеплера —
величины, которые для точного движения строго постоянны:
    * вектор момента импульса          c = r x v            (3 компоненты),
    * вектор Лапласа–Рунге–Ленца (ЛРЛ)  f = v x c - mu*r/|r| (3 компоненты).
Итого 6 первых интегралов (по числу фазовых переменных x, y, z, vx, vy, vz).
Они полностью известны из начальных условий: I0 = [c0, f0].

Алгоритм:
    1. На каждом шаге делаем обычный шаг RK4.
    2. Считаем текущие первые интегралы и их отклонение g = I(u) - I0.
    3. Если относительное отклонение вышло за коридор epsilon — "подтягиваем"
       состояние обратно на многообразие интегралов методом Ньютона
       (Гаусса–Ньютона с псевдообратной матрицей Якоби), решая g(u) = 0.
Так координаты постоянно возвращаются на правильную траекторию.

Автор идеи и реализации: Гефлинг В. И.
"""

import numpy as np

# --- Физические константы и начальные условия (5-й спутник GPS, перигей) ---
MU = 3.9860044e14  # гравитационный параметр Земли, м^3/с^2
U0 = np.array([15845196.0, -1751586.8, 21059119.0,      # x0, y0, z0
               -733.96659, 3727.3516, 862.26895])       # vx0, vy0, vz0
T_PERIOD = 43076.0  # период обращения, с (~11.96 ч)


# ----------------------------- Правая часть ОДУ -----------------------------
def deriv(u):
    """du/dt для двухтельной задачи: u = [x,y,z, vx,vy,vz]."""
    r, v = u[:3], u[3:]
    rn = np.linalg.norm(r)
    return np.concatenate([v, -MU * r / rn**3])


def rk4_step(u, dt):
    """Один шаг классического метода Рунге–Кутты 4-го порядка."""
    k1 = deriv(u)
    k2 = deriv(u + 0.5 * dt * k1)
    k3 = deriv(u + 0.5 * dt * k2)
    k4 = deriv(u + dt * k3)
    return u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


# ----------------------------- Первые интегралы -----------------------------
def first_integrals(u):
    """6 первых интегралов задачи Кеплера: [c(3), f(3)]."""
    r, v = u[:3], u[3:]
    c = np.cross(r, v)                              # момент импульса
    f = np.cross(v, c) - MU * r / np.linalg.norm(r) # вектор Лапласа–Рунге–Ленца
    return np.concatenate([c, f])


def energy(u):
    """Удельная механическая энергия (для контроля точности)."""
    r, v = u[:3], u[3:]
    return 0.5 * v.dot(v) - MU / np.linalg.norm(r)


# ------------------- Коррекция по интегралам (метод Ньютона) ----------------
def _jacobian(u):
    """Якобиан первых интегралов dI/du (6x6), центральными разностями."""
    J = np.zeros((6, 6))
    for i in range(6):
        h = 1e-6 * max(abs(u[i]), 1.0)
        up, um = u.copy(), u.copy()
        up[i] += h
        um[i] -= h
        J[:, i] = (first_integrals(up) - first_integrals(um)) / (2 * h)
    return J


def project(u, I0, Iscale, tol=1e-12, maxit=30):
    """
    Возврат состояния на многообразие первых интегралов.
    Решаем g(u) = I(u) - I0 = 0 методом Гаусса–Ньютона:
        u <- u - pinv(J) @ g(u)
    Псевдообратная матрица берётся из-за того, что 6 интегралов связаны
    одним соотношением (|f|^2 = mu^2 + 2E|c|^2), т.е. матрица Якоби
    вырождена по рангу — pinv находит поправку минимальной нормы
    (ближайшую точку на многообразии).
    """
    for _ in range(maxit):
        g = first_integrals(u) - I0
        if np.max(np.abs(g) / Iscale) < tol:
            break
        u = u - np.linalg.pinv(_jacobian(u), rcond=1e-12) @ g
    return u


# -------------------------------- Интегратор --------------------------------
def integrate(dt, tmax, epsilon=None):
    """
    Проинтегрировать движение на [0, tmax] с шагом dt.
    epsilon=None      — чистый RK4 без коррекции;
    epsilon=<число>   — коррекция по интегралам, когда относительное
                        отклонение max|I(u)-I0|/|I0| превышает epsilon.
    Возвращает: массив траектории (N+1, 3), макс. относительный дрейф энергии,
                число выполненных коррекций.
    """
    u = U0.copy()
    I0 = first_integrals(U0)
    Iscale = np.maximum(np.abs(I0), 1.0)
    e0 = energy(U0)

    traj = [u[:3].copy()]
    max_edrift = 0.0
    n_proj = 0
    for _ in range(int(tmax / dt)):
        u = rk4_step(u, dt)
        if epsilon is not None:
            g = first_integrals(u) - I0
            if np.max(np.abs(g) / Iscale) > epsilon:
                u = project(u, I0, Iscale)
                n_proj += 1
        max_edrift = max(max_edrift, abs(energy(u) - e0) / abs(e0))
        traj.append(u[:3].copy())
    return np.array(traj), max_edrift, n_proj


# ------------------------------- Демонстрация -------------------------------
def _demo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dt = 600.0            # крупный шаг: здесь чистый RK4 заметно уползает
    n_orbits = 30
    tmax = n_orbits * T_PERIOD
    eps = 1e-8

    tr_plain, ed_plain, _ = integrate(dt, tmax, epsilon=None)
    tr_proj,  ed_proj, npj = integrate(dt, tmax, epsilon=eps)

    print(f"Шаг dt = {dt:.0f} с, {n_orbits} оборотов:")
    print(f"  чистый RK4:      макс. относительный дрейф энергии = {ed_plain:.2e}")
    print(f"  RK4 + коррекция: макс. относительный дрейф энергии = {ed_proj:.2e}"
          f"  (коррекций: {npj})")

    # (1) 3D-орбита + Земля
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(tr_proj[:, 0], tr_proj[:, 1], tr_proj[:, 2], color="royalblue", lw=1.0)
    ax.scatter(*U0[:3], color="crimson", s=25)          # старт (перигей)
    R = 6.371e6
    us, vs = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    ax.plot_surface(R*np.cos(us)*np.sin(vs), R*np.sin(us)*np.sin(vs),
                    R*np.cos(vs), color="tan", alpha=0.9, linewidth=0)
    ax.set_xlabel("x (м)"); ax.set_ylabel("y (м)"); ax.set_zlabel("z (м)")
    ax.set_title("Орбита ИСЗ (RK4 + коррекция по первым интегралам)")
    fig.tight_layout(); fig.savefig("orbit_3d.png", dpi=130)

    # (2) Сравнение дрейфа энергии
    def edrift_series(dt, tmax, epsilon):
        u = U0.copy(); I0 = first_integrals(U0)
        Iscale = np.maximum(np.abs(I0), 1.0); e0 = energy(U0)
        t, d = [0.0], [0.0]
        for k in range(int(tmax / dt)):
            u = rk4_step(u, dt)
            if epsilon is not None:
                g = first_integrals(u) - I0
                if np.max(np.abs(g)/Iscale) > epsilon:
                    u = project(u, I0, Iscale)
            t.append((k+1)*dt); d.append(abs(energy(u)-e0)/abs(e0))
        return np.array(t), np.array(d)

    tp, dp = edrift_series(dt, tmax, None)
    tc, dc = edrift_series(dt, tmax, eps)
    fig2, ax2 = plt.subplots(figsize=(7, 4.2))
    ax2.semilogy(tp/T_PERIOD, dp+1e-18, label="чистый RK4", color="crimson")
    ax2.semilogy(tc/T_PERIOD, dc+1e-18, label="RK4 + коррекция", color="royalblue")
    ax2.set_xlabel("число оборотов"); ax2.set_ylabel("относительный дрейф энергии")
    ax2.set_title(f"Стабилизация по первым интегралам (dt = {dt:.0f} с)")
    ax2.legend(); ax2.grid(True, which="both", alpha=0.3)
    fig2.tight_layout(); fig2.savefig("energy_drift.png", dpi=130)
    print("Сохранены графики: orbit_3d.png, energy_drift.png")


if __name__ == "__main__":
    _demo()
