#########################################################################
#                                      ###     ###    #######   ###     #
#                                      ###     ###   ###        ###     #
#                                      ###     ###   ###        ###     #
#                                      ###     ###   ###        ###     #
#                                       #########     #######   #########
#                                                                       #
#                                                                       #
#                GRAPH REPRESENTATION OF INTELLIGENCE                   #
#                                                                       #
# Chord diagram for an (n × n) connection-strength matrix.              #
#                                                                       #
# Node arc sizes are proportional to total connection strength          #
# (sum of row + column). Each arc is subdivided among its outgoing      #
# connections so chord widths at each endpoint reflect the individual   #
# connection strength. Chords are filled quadratic Bezier patches       #
# through the origin; opacity scales with connection strength.          #
#                                                                       #
# Usage (as a module):                                                  #
#   from plot_chord import plot_chord                                   #
#   plot_chord(matrix, labels=[...], colors=[...])                      #
#                                                                       #
# authors: Bey, Patrik                                                  #
#                                                                       #
# last update: 2026/07/07.                                              #
#                                                                       #
#########################################################################


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


# ── helpers ────────────────────────────────────────────────────────────

def _arc_pts(t0, t1, r, n=60):
    """Points along a circle arc from angle t0 to t1 at radius r."""
    th = np.linspace(t0, t1, n)
    return np.column_stack([np.cos(th) * r, np.sin(th) * r])


def _draw_chord_gradient(ax, t1s, t1e, t2s, t2e, r, color_i, color_j, alpha,
                         n_strips=200):
    """
    Draw a chord from arc [t1s, t1e] to arc [t2s, t2e] at radius r with a
    smooth linear gradient from color_i (node i end) to color_j (node j end).

    The chord interior is subdivided into n_strips thin quadrilateral slices,
    each filled with the interpolated colour. Arc end-caps are filled with the
    respective node colour. A PolyCollection is used for the strips so all
    geometry is submitted to the renderer in a single call.
    """
    arc1 = _arc_pts(t1s, t1e, r)
    arc2 = _arc_pts(t2s, t2e, r)

    ci = np.array(color_i[:3], dtype=float)
    cj = np.array(color_j[:3], dtype=float)
    ctrl = np.zeros(2)

    # Two lateral bezier edges, both parameterised t=0 (node i) → t=1 (node j)
    n_pts = n_strips + 1
    tv    = np.linspace(0, 1, n_pts)[:, None]
    bez1  = (1 - tv)**2 * arc1[-1] + 2*(1-tv)*tv * ctrl + tv**2 * arc2[0]
    bez2r = (1 - tv)**2 * arc1[0]  + 2*(1-tv)*tv * ctrl + tv**2 * arc2[-1]

    # Arc end-caps (thin circular-segment slivers at each node)
    ax.fill(arc1[:, 0], arc1[:, 1],
            color=(*ci, alpha), zorder=1, linewidth=0)
    ax.fill(arc2[:, 0], arc2[:, 1],
            color=(*cj, alpha), zorder=1, linewidth=0)

    # Gradient strip quads: shape (n_strips, 4, 2)
    quads = np.stack([bez1[:-1], bez1[1:], bez2r[1:], bez2r[:-1]], axis=1)

    # Per-strip RGBA colours interpolated at each strip's midpoint
    t_mids = (np.arange(n_strips) + 0.5) / n_strips
    rgb    = np.outer(1 - t_mids, ci) + np.outer(t_mids, cj)
    rgba   = np.hstack([rgb, np.full((n_strips, 1), alpha)])

    coll = PolyCollection(quads, facecolors=rgba, edgecolors='none',
                          zorder=1, antialiased=False)
    ax.add_collection(coll)


# ── main function ──────────────────────────────────────────────────────

def plot_chord(matrix, labels=None, colors=None,
               title='', gap=0.03, radius=1.0,
               arc_width=0.06, label_pad=0.12,
               chord_alpha_max=0.75, chord_alpha_min=0.04,
               figsize=(10, 10), output_path=None):
    """
    Parameters
    ----------
    matrix          : (n, n) ndarray — connection strengths (non-negative).
                      Diagonal is ignored.
    labels          : list of n strings.  Default: '0', '1', ...
    colors          : list of n (r, g, b) tuples.  Default: tab20 colormap.
    title           : figure title.
    gap             : fraction of circumference used as gap between nodes.
    radius          : outer radius of the node arcs.
    arc_width       : radial thickness of the node arcs.
    label_pad       : extra radial distance between arc and label.
    chord_alpha_max : opacity of the strongest chord.
    chord_alpha_min : minimum chord opacity (weakest non-zero connection).
    figsize         : matplotlib figure size.
    output_path     : if given, save the figure here.
    """
    matrix = np.asarray(matrix, dtype=float).copy()
    np.fill_diagonal(matrix, 0)
    n = matrix.shape[0]
    assert matrix.shape == (n, n), 'matrix must be square'

    if labels is None:
        labels = [str(i) for i in range(n)]
    if colors is None:
        cmap = plt.colormaps.get_cmap('tab20')
        colors = [cmap(i / max(n - 1, 1))[:3] for i in range(n)]

    inner_r = radius - arc_width

    # ── arc sizes ∝ total connection strength per node ──────────────────
    totals = matrix.sum(axis=1) + matrix.sum(axis=0)
    totals = np.maximum(totals, totals.max() * 0.02)   # minimum visible arc
    fracs  = totals / totals.sum()

    total_gap = gap * n
    arc_space = 1.0 - total_gap

    node_start = np.zeros(n)
    node_end   = np.zeros(n)
    pos = 0.0
    for i in range(n):
        node_start[i] = pos * 2 * np.pi
        pos += fracs[i] * arc_space
        node_end[i]   = pos * 2 * np.pi
        pos += gap

    # ── subdivide each arc among outgoing connections ───────────────────
    # chord_seg[i][j] = (t_start, t_end) on node i's arc for connection i→j
    chord_seg = [[None] * n for _ in range(n)]
    for i in range(n):
        row_total = matrix[i].sum()
        if row_total == 0:
            continue
        arc_len = node_end[i] - node_start[i]
        cur = node_start[i]
        for j in range(n):
            if matrix[i, j] > 0:
                seg = (matrix[i, j] / row_total) * arc_len
                chord_seg[i][j] = (cur, cur + seg)
                cur += seg

    # ── figure ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.axis('off')
    lim = radius + label_pad + 0.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # ── chords (draw first, behind arcs) ────────────────────────────────
    vmax = matrix.max()
    for i in range(n):
        for j in range(i + 1, n):
            strength = max(matrix[i, j], matrix[j, i])
            if strength == 0:
                continue
            if chord_seg[i][j] is None or chord_seg[j][i] is None:
                continue

            alpha = chord_alpha_min + (chord_alpha_max - chord_alpha_min) * (strength / vmax) ** 0.5

            _draw_chord_gradient(ax, *chord_seg[i][j], *chord_seg[j][i],
                                 inner_r, colors[i], colors[j], alpha)

    # ── node arcs (drawn on top of chords) ──────────────────────────────
    for i in range(n):
        th = np.linspace(node_start[i], node_end[i], 200)
        outer = np.column_stack([np.cos(th) * radius, np.sin(th) * radius])
        inner = np.column_stack([np.cos(th[::-1]) * inner_r,
                                  np.sin(th[::-1]) * inner_r])
        verts = np.vstack([outer, inner])
        ax.fill(verts[:, 0], verts[:, 1], color=colors[i], zorder=2)

        # Label
        mid = (node_start[i] + node_end[i]) / 2
        lx  = np.cos(mid) * (radius + label_pad)
        ly  = np.sin(mid) * (radius + label_pad)
        rot = np.degrees(mid) % 360
        if 90 < rot < 270:
            rot += 180
        ha  = 'left' if np.cos(mid) >= 0 else 'right'
        ax.text(lx, ly, labels[i],
                ha=ha, va='center', fontsize=9,
                rotation=rot, rotation_mode='anchor')

    if title:
        ax.set_title(title, fontsize=13, pad=10)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved → {output_path}')
    plt.show()
    return fig, ax


# ── example ────────────────────────────────────────────────────────────

import os
import re

path = '/mnt/h/RT/data/RESULTS/split_threshold'
task = 'Foreperiod_Long_tau'
# task = 'GoNoGo_tau'
# task = 'SATO_Accuracy_tau'


_block_dir  = os.path.join(path, f'SBM_Schaefer2018-400_{task}_singleflip', 'block_niftis')
_files      = os.listdir(_block_dir)
block_files = sorted(f for f in _files if f.endswith('_regions.txt'))

# Extract block numbers using the same regex as plot_tracts_nilearn.py.
# block IDs are used directly as row/column indices into the connectivity matrix.
blocks = sorted(
    int(re.search(r'_block(\d+)_', f).group(1))
    for f in block_files
    if re.search(r'_block(\d+)_', f)
)
node_labels = [f'B-{b}' for b in blocks]

# Load full connectivity matrix and subset to the blocks present in block_files.
# matrix = np.genfromtxt(
#     os.path.join(path, f'SBM_Schaefer2018-400_{task}_singleflip',
#                  f'SBM_block_connectivity_lvl0_{task}.csv'),
#     delimiter=','
# )
matrix = np.genfromtxt(
    os.path.join(path, f'SBM_Schaefer2018-400_{task}_singleflip',
                 f'Lvl0_block_connectome_{task}.tsv'),
    delimiter='\t'
)


red_matrix = matrix[np.ix_(blocks, blocks)].copy()
np.fill_diagonal(red_matrix, 0)

plot_chord(red_matrix, labels=node_labels, title=f'{task} – block white matter', output_path=os.path.join(path,f'SBM_Schaefer2018-400_{task}_singleflip', f'block_white_matter_chord_{task}.svg'))
