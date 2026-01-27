#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from numba import njit
from time import perf_counter
import matplotlib.pyplot as plt
import os
os.environ['VISPY_DPI'] = '96'

# =========================
# PARAMÉTEREK
# =========================
COUNT = 100_000
MODE = "mpl"      # "mpl" vagy "opengl"

# =========================
# PRÍMSZÁM GENERÁLÁS
# =========================
@njit
def sieve(n):
    isprime = np.ones(n + 1, dtype=np.bool_)
    isprime[0:2] = False

    limit = int(n ** 0.5) + 1
    for i in range(2, limit):
        if isprime[i]:
            for j in range(i * i, n + 1, i):
                isprime[j] = False

    return np.where(isprime)[0]

# =========================
# SPIRÁL SZÁMÍTÁS
# =========================
def prime_spiral(primes):
    p = primes.astype(np.float64)
    c = np.cos(p)
    s = np.sin(p)
    qx = p * (c - s)
    qy = p * (s + c)
    return qx, qy

# =========================
# SZÍNEZÉS
# =========================
def make_colors(primes):
    idx_norm = np.linspace(0, 1, len(primes))
    log_norm = np.log(primes) / np.log(primes[-1])
    return plt.cm.turbo(0.6 * idx_norm + 0.4 * log_norm)

# =========================
# MATPLOTLIB – INTERAKTÍV
# =========================
def plot_matplotlib(qx, qy, colors):
    plt.figure(figsize=(10, 10), dpi=120)
    plt.scatter(
        qx, qy,
        s=1,
        facecolors='none',
        edgecolors=colors,
        linewidths=0.4
    )
    plt.axis("equal")
    plt.title("Prime spiral – matplotlib (zoom / pan)")
    plt.show()

# =========================
# OPENGL – VISPY
# =========================
def plot_opengl(qx, qy, colors):
    from vispy import scene, app

    canvas = scene.SceneCanvas(
        keys='interactive',
        bgcolor='black',
        size=(1000, 1000),
        show=True
    )

    view = canvas.central_widget.add_view()
    view.camera = 'panzoom'

    pos = np.column_stack((qx, qy, np.zeros_like(qx)))

    scatter = scene.visuals.Markers()
    scatter.set_data(
        pos,
        face_color=None,
        edge_color=colors,
        size=2
    )

    view.add(scatter)
    view.camera.set_range()

    app.run()

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    print(f"Prime spiral | count = {COUNT}")
    print("-" * 40)

    t0 = perf_counter()
    primes = sieve(COUNT)
    t1 = perf_counter()
    print(f"Prímek száma     : {len(primes)}")
    print(f"Prím generálás   : {t1 - t0:.4f} s")

    t0 = perf_counter()
    qx, qy = prime_spiral(primes)
    t1 = perf_counter()
    print(f"Spirál számítás  : {t1 - t0:.4f} s")

    colors = make_colors(primes)

    print("-" * 40)
    print(f"Mód: {MODE}")

    if MODE == "mpl":
        plot_matplotlib(qx, qy, colors)
    elif MODE == "opengl":
        plot_opengl(qx, qy, colors)
    else:
        print("Ismeretlen MODE (mpl / opengl)")

    print("Kész ✅")

