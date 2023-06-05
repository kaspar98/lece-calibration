import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from sklearn.preprocessing import normalize


class Arrow3D(FancyArrowPatch):
    # https://stackoverflow.com/questions/58903383/fancyarrowpatch-in-3d
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def draw_classic_arrow(arrow_start, arrow_end, ax, alpha=1, color="black", lw=1, zorder=20, arrowstyle="->",
                       mutation_scale=12, shrinkA=0, shrinkB=0, **kwargs):
    arw = Arrow3D([arrow_start[0], arrow_end[0]],
                  [arrow_start[1], arrow_end[1]],
                  [arrow_start[2], arrow_end[2]],
                  arrowstyle=arrowstyle, color=color, lw=lw, mutation_scale=mutation_scale, zorder=zorder, alpha=alpha,
                  shrinkA=shrinkA, shrinkB=shrinkB, **kwargs)
    ax.add_artist(arw)


def generate_points_on_triangle(step_size=0.005, rounding_to=4, start=0.0):
    xx, yy = np.meshgrid(np.arange(start, 1 + step_size / 2, step_size),
                         np.arange(start, 1 + step_size / 2, step_size))
    xx, yy = xx[xx + yy <= 1], yy[xx + yy <= 1]
    xx = np.round(xx, rounding_to)
    yy = np.round(yy, rounding_to)
    z = 1 - xx - yy
    z = np.round(z, rounding_to)

    p = np.dstack((xx, yy, z))[0]
    p = p[(p[:, 2] >= start)]
    return p


def set_up_3d_simplex():
    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_zlim(0, 1)

    ax.view_init(elev=20, azim=45)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.grid(False)

    return fig, ax


def add_3d_axis(ax):
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    ax.zaxis.labelpad = 10
    ticks = np.round(np.arange(0.0, 1.01, 0.2), 4)

    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)

    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)

    ax.set_zticks(ticks)
    ax.set_zticklabels(ticks)

    ax.text(1.29, 0, 0.55, s="$\hat{p}_3$", fontsize=14)
    ax.text(1.29, 0.6, -0.1, s="$\hat{p}_2$", fontsize=14)
    ax.text(0.6, 1.26, -0.1, s="$\hat{p}_1$", fontsize=14)


def turn_off_axis(ax):
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    plt.axis('off')


def draw_on_simplex_guidelines(ax):
    for i in np.arange(0, 1, 0.2):
        ax.plot([0, 1 - i], [1 - i, 0], [i, i], color="black", alpha=0.5, lw=0.5, zorder=20,
                rasterized=True)
        ax.plot([i, i], [1 - i, 0], [0, 1 - i], color="black", alpha=0.5, lw=0.5, zorder=20,
                rasterized=True)
        ax.plot([0, 1 - i], [i, i], [1 - i, 0], color="black", alpha=0.5, lw=0.5, zorder=20,
                rasterized=True)


def draw_axis_to_simplex_guidelines(ax):
    for i in np.arange(0, 1, 0.2):
        ax.plot([i, i], [1, 1 - i], [0, 0], color="black", alpha=0.5, lw=0.5, zorder=20,
                rasterized=True)
        ax.plot([1, 1 - i], [i, i], [0, 0], color="black", alpha=0.5, lw=0.5, zorder=20,
                rasterized=True)
        ax.plot([1, 1 - i], [0, 0], [i, i], color="black", alpha=0.5, lw=0.5, zorder=20,
                rasterized=True)


def draw_calibration_arrows(ax, cal_fun):
    # Arrows
    n_arrows = 14
    start_coord = 0.02375
    end_coord = 1 - 2 * 0.02375
    step_size = (end_coord - start_coord) / (n_arrows - 1)

    arrow_starts = generate_points_on_triangle(step_size=step_size, rounding_to=7,
                                               start=start_coord)
    arrow_ends = cal_fun(arrow_starts)
    # elongate the arrows a bit as matplotlib draws them a bit shorter than they actually are
    arrow_ends += 0.007 * normalize(arrow_ends - arrow_starts, "l1")

    for idx in range(len(arrow_starts)):
        draw_classic_arrow(arrow_starts[idx], arrow_ends[idx], ax=ax, alpha=0.7,
                           color="black", lw=1.7, mutation_scale=18)


def draw_background_colors(ax, cal_fun):
    points = generate_points_on_triangle(step_size=0.0025,
                                         rounding_to=4)
    points = points[points[:, 0] > 0]
    points = points[points[:, 1] > 0]
    points[points == 0] = 1e-8

    cal_points = cal_fun(points)
    color_intensity = 3.5
    colors = np.clip(((points - cal_points) * color_intensity + 0.55), a_min=0, a_max=1)

    cmap = copy.copy(plt.get_cmap('YlOrRd'))
    norm = None

    # if len(colors.shape) == 1:
    #    colors = np.ma.masked_where(colors == 0, colors)

    ax.scatter(points[:, 0],
               points[:, 1],
               points[:, 2],
               alpha=1, c=colors, s=0.25, zorder=1, cmap=cmap, norm=norm, rasterized=True, marker="o")


def draw_cf_triangle(cal_fun, title, data_test, name, draw_axis=False):
    true_ce = np.round(np.mean(np.abs(cal_fun(data_test["p"]) - data_test['c'])), 7)
    print(title)
    print(f"True multiclass mean absolute CE: {true_ce}")

    # Fig
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12

    fig, ax = set_up_3d_simplex()

    if draw_axis:
        add_3d_axis(ax)
        draw_axis_to_simplex_guidelines(ax)
    else:
        turn_off_axis(ax)

    draw_on_simplex_guidelines(ax)
    draw_calibration_arrows(ax, cal_fun)
    draw_background_colors(ax, cal_fun)

    ax.text(0, 0, 1.06, s=title, fontsize=18, ha="center")

    plt.savefig("../figures/" + name + ".pdf", dpi=300, bbox_inches="tight", transparent=True)
    plt.show()
