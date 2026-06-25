import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.proj3d import proj_transform

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "mathtext.fontset": "cm",
    "figure.dpi": 300,
})

# Palette --------------------------------------------------------------------
SPHERE_COLOR = "#e8ecf4"   # very light fill
GRID_COLOR   = "#aeb8cc"   # meridians / parallels
AXIS_COLOR   = "#3a3f4b"   # cartesian axes
LABEL_GREY   = "#5a606b"   # captions

C_PURE  = "#c1272d"   # pure-state red
C_MIXED = "#1f8a70"   # mixed-state teal-green
SIC_COLORS = ["#d1495b", "#2e6fb0", "#e08e0b", "#7d4f9e"]   # tetrahedron
PVM_COLORS = ["#2e6fb0", "#d1495b"]                          # antipodal pair


# ── Nice 3D arrows (depth-sorted FancyArrowPatch) ────────────────────────────
class Arrow3D(FancyArrowPatch):
    """A 3D arrow with clean, consistent heads and correct z-ordering."""

    def __init__(self, origin, vec, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._origin = np.asarray(origin, dtype=float)
        self._vec = np.asarray(vec, dtype=float)

    def _project(self):
        x0, y0, z0 = self._origin
        x1, y1, z1 = self._origin + self._vec
        xs, ys, zs = proj_transform((x0, x1), (y0, y1), (z0, z1), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

    def do_3d_projection(self, renderer=None):
        return self._project()

    def draw(self, renderer):
        self._project()
        super().draw(renderer)


def add_arrow(ax, vec, color, lw=2.4, scale=14, origin=(0, 0, 0), zorder=10,
              alpha=1.0):
    a = Arrow3D(origin, vec, arrowstyle='-|>', mutation_scale=scale,
                lw=lw, color=color, zorder=zorder, alpha=alpha,
                shrinkA=0, shrinkB=0)
    ax.add_artist(a)
    return a


# ── Sphere helper ────────────────────────────────────────────────────────────
def draw_bloch_sphere(ax):
    """Draw a clean unit Bloch sphere: light fill, sparse grid, labelled axes."""
    u = np.linspace(0, 2 * np.pi, 90)
    v = np.linspace(0, np.pi, 70)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=SPHERE_COLOR, alpha=0.15,
                    linewidth=0, antialiased=True, zorder=0,
                    rcount=45, ccount=45)

    # Sparse guide circles: equator + two great meridians only
    phi = np.linspace(0, 2 * np.pi, 160)
    zeros = np.zeros_like(phi)
    ax.plot(np.cos(phi), np.sin(phi), zeros, color=GRID_COLOR, lw=0.9, alpha=0.9)
    ax.plot(np.cos(phi), zeros, np.sin(phi), color=GRID_COLOR, lw=0.7, alpha=0.5)
    ax.plot(zeros, np.cos(phi), np.sin(phi), color=GRID_COLOR, lw=0.7, alpha=0.5)

    # Cartesian axes
    for vec, lbl in zip([[1.32, 0, 0], [0, 1.32, 0], [0, 0, 1.32]],
                        ['x', 'y', 'z']):
        add_arrow(ax, vec, AXIS_COLOR, lw=1.1, scale=9, zorder=5)
        ax.text(*[c * 1.18 for c in vec], lbl, color=AXIS_COLOR,
                fontsize=15, ha='center', va='center')

    # Computational-basis poles
    ax.text(0.12, 0,  1.30, r'$|0\rangle$', ha='center', va='bottom', fontsize=17)
    ax.text(0.12, 0, -1.30, r'$|1\rangle$', ha='center', va='top',    fontsize=17)

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    # zoom enlarges the sphere inside the axes box -> less empty margin
    ax.set_box_aspect([1, 1, 1], zoom=1.18)
    ax.axis('off')


view = (20, -58)


# ── Panel drawers (no titles) ────────────────────────────────────────────────
def panel_pure_mixed(ax):
    """(a) Pure state on the surface vs. mixed state in the interior."""
    draw_bloch_sphere(ax)

    # Pure state on the surface (|r| = 1)
    theta, phi = np.radians(50), np.radians(35)
    pure = np.array([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)])
    add_arrow(ax, pure, C_PURE, lw=2.6, scale=15, zorder=12)
    ax.scatter(*pure, color=C_PURE, s=46, zorder=13, edgecolor='white', linewidth=0.6)
    ax.text(pure[0] + 0.10, pure[1] + 0.12, pure[2] + 0.04, r'$|\psi\rangle$',
            color=C_PURE, fontsize=17, fontweight='bold')

    # Mixed state in the interior (|r| < 1), different direction for clarity
    theta_m, phi_m = np.radians(66), np.radians(158)
    mixed = 0.5 * np.array([np.sin(theta_m) * np.cos(phi_m),
                            np.sin(theta_m) * np.sin(phi_m),
                            np.cos(theta_m)])
    add_arrow(ax, mixed, C_MIXED, lw=2.6, scale=15, zorder=11)
    ax.scatter(*mixed, color=C_MIXED, s=70, zorder=13, edgecolor='white', linewidth=0.7)
    ax.text(mixed[0] - 0.10, mixed[1] + 0.10, mixed[2] + 0.12, r'$\rho_{\mathrm{mix}}$',
            color=C_MIXED, fontsize=17, fontweight='bold', ha='right')

    # Compact in-panel legend
    legend_handles = [
        Line2D([0], [0], color=C_PURE, lw=2.6,
               label=r'pure  $|\psi\rangle$ :  $|\vec{r}\,| = 1$'),
        Line2D([0], [0], color=C_MIXED, lw=2.6,
               label=r'mixed  $\rho$ :  $|\vec{r}\,| < 1$'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.02, 0.97),
              frameon=False, fontsize=16, handlelength=1.4, borderaxespad=0.0)


def panel_pvm(ax):
    """(b) Standard PVM: two orthogonal (antipodal) projectors."""
    draw_bloch_sphere(ax)

    pvm_vertices = np.array([[0, 0, 1], [0, 0, -1]], dtype=float)
    for v, c in zip(pvm_vertices, PVM_COLORS):
        add_arrow(ax, v, c, lw=2.8, scale=15, zorder=11)
        ax.scatter(*v, color=c, s=56, zorder=13, edgecolor='white', linewidth=0.6)

    ax.text(0.18, 0.10,  0.62, r'$|0\rangle\langle 0|$', color=PVM_COLORS[0],
            fontsize=18.5, fontweight='bold', ha='left', va='center')
    ax.text(0.18, 0.10, -0.62, r'$|1\rangle\langle 1|$', color=PVM_COLORS[1],
            fontsize=18.5, fontweight='bold', ha='left', va='center')


def panel_sic(ax):
    """(c) SIC-POVM: regular tetrahedron with one vector along +z."""
    draw_bloch_sphere(ax)

    # Canonical orientation: apex on the +z axis (s_1 aligned with |0>),
    # the other three forming the base at z = -1/3.
    sic_vertices = np.array([
        [0,                 0,            1.0],
        [2 * np.sqrt(2) / 3, 0,          -1 / 3],
        [-np.sqrt(2) / 3,   np.sqrt(6) / 3, -1 / 3],
        [-np.sqrt(2) / 3,  -np.sqrt(6) / 3, -1 / 3],
    ])
    # Spin the tripod so the three base legs fall in the gaps between the
    # cartesian axes -> none overlaps the x/y axes, none hides behind the sphere.
    a = np.radians(-15)
    Rz = np.array([[np.cos(a), -np.sin(a), 0],
                   [np.sin(a),  np.cos(a), 0],
                   [0,          0,         1]])
    sic_vertices = sic_vertices @ Rz.T
    sic_labels = [r'$\vec{s}_1$', r'$\vec{s}_2$', r'$\vec{s}_3$', r'$\vec{s}_4$']
    # Per-vertex label offsets (apex pushed off the z-axis to avoid the |0> pole)
    label_pos = [
        sic_vertices[0] + np.array([0.20, 0.12, 0.10]),
        sic_vertices[1] * 1.24,
        # s_3 points away from the viewer (foreshortened): lift the label up
        # and out so it clears its own arrow head.
        sic_vertices[2] * 1.28 + np.array([0.0, 0.18, 0.08]),
        sic_vertices[3] * 1.28,
    ]

    # Tetrahedron edges first, so the vectors render on top
    for i, j in combinations(range(4), 2):
        seg = np.vstack([sic_vertices[i], sic_vertices[j]])
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], '-',
                color="#9aa3b2", lw=1.0, alpha=0.5, zorder=4)

    for v, c, lbl, lp in zip(sic_vertices, SIC_COLORS, sic_labels, label_pos):
        add_arrow(ax, v, c, lw=2.4, scale=14, zorder=11)
        ax.scatter(*v, color=c, s=50, zorder=13, edgecolor='white', linewidth=0.6)
        ax.text(*lp, lbl, color=c, fontsize=22, fontweight='bold',
                ha='center', va='center')


# ── Render & export each panel as its own (title-free) image ─────────────────
PANELS = {
    "bloch_pure_mixed": panel_pure_mixed,
    "bloch_pvm":        panel_pvm,
    "bloch_sic_povm":   panel_sic,
}

_here = os.path.dirname(os.path.abspath(__file__))
for name, draw in PANELS.items():
    fig = plt.figure(figsize=(5.6, 5.6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(*view)
    draw(ax)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    #fig.savefig(os.path.join(_here, f"{name}.png"), dpi=400, bbox_inches='tight')
    fig.savefig(os.path.join(_here, f"{name}.svg"), bbox_inches='tight')

plt.show()
