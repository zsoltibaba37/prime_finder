import numpy as np
import torch
from time import perf_counter
from vispy import app, scene
from vispy.scene import visuals

DEVICE = torch.device("cuda")


# =========================
# CUDA SIEVE
# =========================
def sieve_cuda(n: int) -> torch.Tensor:
    arr = torch.ones(n + 1, dtype=torch.bool, device=DEVICE)
    arr[:2] = False

    limit = int(n**0.5) + 1
    for p in range(2, limit):
        if arr[p]:
            arr[p*p:n+1:p] = False

    return torch.nonzero(arr).squeeze(1).to(torch.long)


# =========================
# POLÁRIS SPIRÁL CUDA-N
# =========================
def spiral_polar_cuda(primes: torch.Tensor):
    n = primes.to(torch.float32)

    # Spirál paraméterei
    theta = n * 0.1
    r = n * 0.02

    x = r * torch.cos(theta)
    y = r * torch.sin(theta)

    return x.to(torch.float32), y.to(torch.float32)


# =========================
# CUDA COLORS
# =========================
def colors_cuda(primes: torch.Tensor):
    n = primes.shape[0]
    t = torch.linspace(0, 1, n, device=DEVICE)

    r = t
    g = 1 - torch.abs(t - 0.5) * 2
    g = torch.clamp(g, 0, 1)
    b = 1 - t

    colors = torch.stack([r, g, b], dim=1)
    colors = torch.clamp(colors * 1.3, 0, 1)
    return colors


# =========================
# OPENGL / VISPY PLOT
# =========================
def plot_opengl(qx, qy, colors):
    canvas = scene.SceneCanvas(
        keys='interactive',
        show=True,
        bgcolor='#202020',
        size=(1000, 1000)
    )

    view = canvas.central_widget.add_view()

    pts = np.column_stack((qx, qy))

    scatter = visuals.Markers()
    scatter.set_data(
        pts,
        face_color=colors,
        size=3.5,
        edge_width=0.0
    )
    view.add(scatter)

    span_x = qx.max() - qx.min()
    span_y = qy.max() - qy.min()
    span = max(span_x, span_y)

    view.camera = scene.cameras.TurntableCamera(
        fov=0,
        elevation=90,
        azimuth=0,
        distance=span * 1.2
    )

    view.camera.set_range(
        x=(qx.min(), qx.max()),
        y=(qy.min(), qy.max())
    )

    app.run()


# =========================
# MAIN
# =========================
COUNT = 200000

if __name__ == "__main__":

    print(f"Prime spiral | count = {COUNT}")
    print("-" * 40)

    t0 = perf_counter()
    primes = sieve_cuda(COUNT)
    t1 = perf_counter()
    print(f"Prímek száma     : {primes.shape[0]}")
    print(f"Prím generálás   : {t1 - t0:.4f} s")

    t0 = perf_counter()
    qx_t, qy_t = spiral_polar_cuda(primes)
    t1 = perf_counter()
    print(f"Spirál számítás  : {t1 - t0:.4f} s")

    t0 = perf_counter()
    colors_t = colors_cuda(primes)
    t1 = perf_counter()
    print(f"Színezés         : {t1 - t0:.4f} s")

    qx = qx_t.cpu().numpy().astype(np.float32)
    qy = qy_t.cpu().numpy().astype(np.float32)
    colors = colors_t.cpu().numpy().astype(np.float32)

    print("-" * 40)
    print("OpenGL mód")

    plot_opengl(qx, qy, colors)

    print("Kész")

