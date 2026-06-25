import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
})

SPHERE_ALPHA = 0.08
SPHERE_COLOR = "#aaaacc"
WIRE_COLOR   = "#ccccdd"
AXIS_COLOR   = "#444444"

def draw_bloch_sphere(ax, title):
    """Draw a unit Bloch sphere with wireframe and axes."""
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 40)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color=SPHERE_COLOR, alpha=SPHERE_ALPHA, zorder=0)
    ax.plot_wireframe(x, y, z, color=WIRE_COLOR, linewidth=0.3, alpha=0.4, zorder=0)

    # Equator and meridians
    phi = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(phi), np.sin(phi), 0, color=WIRE_COLOR, lw=0.6, alpha=0.7)
    ax.plot(np.cos(phi), np.zeros_like(phi), np.sin(phi), color=WIRE_COLOR, lw=0.6, alpha=0.7)
    ax.plot(np.zeros_like(phi), np.cos(phi), np.sin(phi), color=WIRE_COLOR, lw=0.6, alpha=0.7)

    # Cartesian axes
    for vec, lbl in zip([[1.35,0,0],[0,1.35,0],[0,0,1.35]], ['x','y','z']):
        ax.quiver(0, 0, 0, *vec, color=AXIS_COLOR, linewidth=0.8,
                  arrow_length_ratio=0.12, zorder=5)
        ax.text(*[v*1.12 for v in vec], lbl, color=AXIS_COLOR,
                fontsize=8, ha='center', va='center')

    # Poles
    ax.text(0, 0,  1.18, r'$|0\rangle$', ha='center', va='bottom', fontsize=8)
    ax.text(0, 0, -1.18, r'$|1\rangle$', ha='center', va='top',    fontsize=8)

    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    ax.set_zlim([-1.3, 1.3])
    ax.set_box_aspect([1, 1, 1])
    ax.axis('off')
    ax.set_title(title, pad=6)

# ── Figure setup ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 4.5))
axes = [fig.add_subplot(131, projection='3d'),
        fig.add_subplot(132, projection='3d'),
        fig.add_subplot(133, projection='3d')]

view = (22, -55)
for ax in axes:
    ax.view_init(*view)

# ══════════════════════════════════════════════════════════════════════════════
# Panel 1 — Pure state
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[0]
draw_bloch_sphere(ax, "(a) Pure State")

# A generic pure state: θ=50°, φ=40°
theta, phi = np.radians(50), np.radians(40)
bx = np.sin(theta) * np.cos(phi)
by = np.sin(theta) * np.sin(phi)
bz = np.cos(theta)

# State vector (arrow from origin to surface)
ax.quiver(0, 0, 0, bx, by, bz, color="#c0392b", linewidth=2.0,
          arrow_length_ratio=0.10, zorder=10)
ax.scatter([bx], [by], [bz], color="#c0392b", s=28, zorder=11)

# Dashed projection lines
ax.plot([bx, bx], [by, by], [0, bz], '--', color="#c0392b", lw=0.8, alpha=0.6)
ax.plot([0, bx], [0, by],   [0, 0],  '--', color="#c0392b", lw=0.8, alpha=0.6)

# Labels
ax.text(bx + 0.08, by + 0.08, bz + 0.08,
        r'$|\psi\rangle$', color="#c0392b", fontsize=9, fontweight='bold')
ax.text(0.05, 0.05, -1.45,
        r'$r = 1$ (surface)', ha='center', fontsize=7.5, color='#555555', style='italic')

# ══════════════════════════════════════════════════════════════════════════════
# Panel 2 — Mixed state
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[1]
draw_bloch_sphere(ax, "(b) Mixed State")

# Mixed state point inside sphere
mx, my, mz = 0.30, 0.25, 0.28

ax.scatter([mx], [my], [mz], color="#27ae60", s=60, zorder=11, marker='o')
ax.plot([0, mx], [0, my], [0, mz], '-', color="#27ae60", lw=1.6, zorder=10)

# Dashed projections
ax.plot([mx, mx], [my, my], [0, mz], '--', color="#27ae60", lw=0.8, alpha=0.6)
ax.plot([0, mx], [0, my],   [0, 0],  '--', color="#27ae60", lw=0.8, alpha=0.6)

# Labels
ax.text(mx + 0.08, my + 0.08, mz + 0.10,
        r'$\rho_{\rm mix}$', color="#27ae60", fontsize=9, fontweight='bold')
ax.text(0.05, 0.05, -1.45,
        r'$r < 1$ (interior)', ha='center', fontsize=7.5, color='#555555', style='italic')

# Indicate purity qualitatively
ax.text(-1.1, -1.1, -1.2,
        r'$\mathrm{Tr}(\rho^2) < 1$', fontsize=7.5, color='#27ae60', alpha=0.85)

# ══════════════════════════════════════════════════════════════════════════════
# Panel 3 — SIC-POVM
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[2]
draw_bloch_sphere(ax, "(c) SIC-POVM on Bloch Sphere")

# SIC-POVM: 4 vertices of a regular tetrahedron inscribed in the Bloch sphere
# Canonical tetrahedron vertices
sic_vertices = np.array([
    [ 0,           0,           1         ],
    [ 2*np.sqrt(2)/3,  0,      -1/3       ],
    [-np.sqrt(2)/3,  np.sqrt(6)/3, -1/3   ],
    [-np.sqrt(2)/3, -np.sqrt(6)/3, -1/3   ],
])

sic_colors = ["#e74c3c", "#3498db", "#e67e22", "#8e44ad"]
sic_labels = [r'$M_1$', r'$M_2$', r'$M_3$', r'$M_4$']

for i, (v, c, lbl) in enumerate(zip(sic_vertices, sic_colors, sic_labels)):
    # Arrow from origin to vertex
    ax.quiver(0, 0, 0, v[0], v[1], v[2],
              color=c, linewidth=1.8, arrow_length_ratio=0.10, zorder=10)
    ax.scatter(*v, color=c, s=40, zorder=11)
    offset = v * 1.18
    ax.text(*offset, lbl, color=c, fontsize=8.5, fontweight='bold',
            ha='center', va='center')

# Draw tetrahedron edges (dashed)
from itertools import combinations
for i, j in combinations(range(4), 2):
    xs = [sic_vertices[i][0], sic_vertices[j][0]]
    ys = [sic_vertices[i][1], sic_vertices[j][1]]
    zs = [sic_vertices[i][2], sic_vertices[j][2]]
    ax.plot(xs, ys, zs, '--', color='#999999', lw=0.7, alpha=0.6, zorder=4)

# Legend-style annotation
ax.text(-1.35, -1.35, -1.35,
        "4 equidistant\noutcomes",
        fontsize=7, color='#555555', style='italic', va='bottom')

# ── Final layout & export ──────────────────────────────────────────────────────
fig.suptitle("Bloch Sphere Representations", fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout(pad=1.5)
plt.show()