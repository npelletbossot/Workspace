"""
nucleo.plot_functions
------------------------
Plot functions for writing results, etc.
"""


# ─────────────────────────────────────────────
# 1 : Librairies
# ─────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec


# ─────────────────────────────────────────────
# 2 : Functions
# ─────────────────────────────────────────────


fontsize = 16


# - Fig1. Line 1 - #


def plot_obstacle(s, l, origin, alpha_mean, xmin = 10_000, xmax = 1_000, text_size=fontsize, ax=None):
    ax.set_title(f'Accessibility for s={s}, l={l}', size=text_size)
    ax.plot(alpha_mean[xmin:xmin+xmax], c='b', ls='-', label='mean obstacle', lw=1)
    # ax.fill_between(np.arange(0, len(alpha_mean), 1), alpha_mean, step='post', color='b', alpha=0.3, label='accessible binding sites')
    ax.axvline(x=origin, c='r', ls='--', label=f'origin={origin}')
    ax.set_xlabel('x', fontsize=text_size)
    ax.set_ylabel((r"$\alpha$"), fontsize=text_size)
    ax.set_xlim([0, xmax])
    ax.set_ylim([-0.10, 1.10])
    ax.grid(True, which='both')
    # ax.legend(fontsize=text_size, loc='upper right')


def plot_obs_linker_distrib(s, s_points, s_distrib, l_points, l_distrib, text_size=16, ax=None, plot_s=True):
    
    # Convert to numpy arrays if needed
    s_points = np.asarray(s_points)
    s_distrib = np.asarray(s_distrib)
    l_points = np.asarray(l_points)
    l_distrib = np.asarray(l_distrib)

    # Create axes
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6), sharex=True)
    else:
        fig = ax.figure
        ax.clear()
        ax.set_visible(False)
        gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=ax.get_subplotspec(), hspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        
    # Linker distribution
    ax1.plot(l_points, l_distrib, label='linkers',
             color='r', alpha=0.75, marker='o')

    # ax2.set_title('Linker distribution', size=text_size)
    ax1.set_xlabel('Size of linker', fontsize=text_size)
    ax1.set_ylabel('distribution', fontsize=text_size)
    ax1.set_ylim([-0.10, 1.10])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=text_size)
        
    # Filtering logic
    if plot_s:
        mask_s = (s_distrib != 0)
        s_points = s_points[mask_s]
        s_distrib = s_distrib[mask_s]
        ax2.plot(s_points//s, s_distrib, label='obstacles',
             color='b', alpha=0.75, marker='o')
        ax2.set_xticks(np.arange(1, 6, 1, dtype=int))
        ax2.set_xlabel("Consecutive roadblocks", size=text_size)

    else:
    # Obstacle distribution
        ax2.plot(s_points, s_distrib, label='obstacles',
             color='b', alpha=0.75, marker='o')
        ax2.set_xlabel("Lenght of obstacle (a.u.)", size=text_size)

    # ax1.set_title('Obstacle distribution', size=text_size)
    ax2.set_ylabel('distribution', fontsize=text_size)
    ax2.set_ylim([-0.10, 1.10])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=text_size)

    return fig


def plot_linker_view(link_view, xmax = 5_000, text_size=fontsize, ax=None):
    ax.set_title(f'Linker view as an obstacle', size=text_size)
    ax.plot(link_view, label="link_view", ls="-", color="orange")
    ax.set_xlabel('x (a.u.)', fontsize=text_size)
    ax.set_ylabel('alpha', fontsize=text_size)
    ax.set_xlim(0, xmax)
    ax.set_ylim([-0.10, 1.10])
    ax.grid(True, which='both')
    ax.legend(fontsize=text_size, loc='upper right')


def plot_probabilities(mu, theta, p, text_size=fontsize, ax=None):
    ax.set_title(f'Capture probability', size=text_size)
    ax.plot(p, label=f'mu={mu} - theta={theta}', c='r', lw=2)
    ax.set_xlim([0, 0+1000])
    ax.set_ylim([-0.005, 0.025])
    ax.set_ylabel((r"$p(\Delta x)$"), size=text_size)
    ax.set_xlabel((r"$\Delta x$"), size=text_size)
    ax.grid(True, which='both')
    ax.legend(fontsize=text_size, loc='upper right')


def plot_trajectories(tmax, times, results, results_mean, results_med, results_std, v_mean, v_med, text_size=fontsize, ax=None):
    ax.set_title(f'Trajectories', size=text_size)
    # ax.plot(results[0], drawstyle='steps-mid', lw=0.50, c='r', label='trajectories')
    # for _ in range(1, len(results)):
    #     ax.plot(results[_], drawstyle='steps-mid', lw=0.50, c='r')
    for i in range(9, 12):
        ax.plot(results[i], drawstyle='steps-mid', lw=2, ls="--", label=f"trajectory_{i-8}")
    # ax.errorbar(x=times, y=results_mean, yerr=results_std, c='b', ls='-', label=f'mean_trajectory', lw=1)
    ax.plot(times, results_mean, c='r', ls='-', label=f'mean_trajectory \nv_mean={np.round(v_mean,2)}', lw=2)
    # ax.plot(times, results_med, c='g', ls='--', label=f'med_trajectory', lw=1)
    ax.set_xlabel(r'time in ($1 / k_0$) unit', fontsize=text_size)
    ax.set_ylabel('x', fontsize=text_size)
    ax.set_xlim([0, tmax])
    ax.set_ylim([0, 7_000])
    ax.grid(True, which='both')
    ax.legend(fontsize=text_size, loc='upper left')


# - Fig1. Line 2 - #


def plot_fpt_distrib_2d(fpt_distrib_2D, tmax, time_bin, text_size=fontsize, ax=None):
    ax.set_title('Distribution of fpts', size=text_size)
    im = ax.imshow(fpt_distrib_2D, aspect='auto', cmap='bwr', origin='lower', vmin=0, vmax=0.01)
    num_bins = fpt_distrib_2D.shape[1]
    x_ticks = np.arange(0, num_bins, step=max(1, num_bins // 10))
    x_labels = x_ticks * time_bin
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('x (a.u.)', size=text_size)
    ax.set_ylabel('t (a.u.)', size=text_size)
    # ax.set_xlim([0, 10_000])
    ax.set_ylim([0, tmax - 1])
    plt.colorbar(im, ax=ax, label='Value')
    ax.grid(True, which='both')


def plot_fpt_number(nt, tmax, fpt_number, time_bin, text_size=fontsize, ax=None):
    ax.set_title(f'Number of trajectories that reached', size=text_size)
    x_values = np.arange(len(fpt_number)) * time_bin
    ax.plot(x_values, fpt_number, label='number', color='b', alpha=0.7, marker='s')
    ax.set_xlabel('x (a.u.)', fontsize=text_size)
    ax.set_ylabel('number of trajectories', fontsize=text_size)
    ax.set_xlim([0, 10_000])
    ax.set_ylim([-200, nt+200])
    ax.grid(True, which='both')
    ax.legend(fontsize=text_size, loc='upper right')


def plot_waiting_times(tbj_points, tbj_distrib, text_size=fontsize, ax=None):
    ax.set_title(f'Distribution of waiting times', size=text_size)
    ax.plot(tbj_points, tbj_distrib, c='b', label='time between jumps')
    ax.grid(True, which='both')
    ax.set_xlabel('time between jumps', size=text_size)
    ax.set_ylabel('distribution', size=text_size)
    ax.set_ylim([1e-5, 1e-1])
    ax.set_xlim([1e-1, 1e6])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=text_size)


def plot_speed_distribution(
    vi_points,
    vi_distrib,
    v_mean,
    vi_mean,
    vi_med,
    vi_mp,
    text_size=16,
    ax=None,
    color="b",
    label=None,
    title="homogeneous",
    plot_vertical=False
):

    if ax is None:
        ax = plt.gca()

    # --- Clean title ---
    if title == "constant_mean":
        title = "Homogeneous"
    elif title == "nt_random":
        title = "Random"
    elif title == "periodic":
        title = "Periodic"

    # ax.set_title(f"{title}", size=text_size)

    # --- Simulation curve (ONLY thing in legend for color) ---
    ax.plot(
        vi_points,
        vi_distrib,
        c=color,
        ls="-",
        lw=2,
        alpha=0.80,
        label=label
    )

    # # --- Mean line (no legend entry) ---
    # if plot_vertical:
    #     ax.axvline(
    #         x=v_mean / 2,
    #         c=color,
    #         ls=':',
    #         lw=2,
    #         alpha=1,
    #         label="_nolegend_"
    #     )

    # --- Formatting ---
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlabel(r'speeds ($\sigma k_0$)', fontsize=text_size)
    ax.set_ylabel('Distribution', size=text_size)

    ax.set_ylim([1e-5, 1e0])
    ax.set_xlim([5e-1, 1e3])

    ax.set_xscale('log')
    ax.set_yscale('log')

    return ax


# - Fig2. Line 1 + Line 2 - #


def plot_fitting_summary(times, positions, v_mean,
                         xt_over_t, G,
                         vf, vf_std, Cf, Cf_std, wf, wf_std,
                         bound_low=5, bound_high=80,
                         rf=3, text_size=16, ax=None):
    """
    Plot all fitting steps in a 2x4 panel grid.
    Designed to be called inside a larger subplot layout.
    """

    # --- Early exit if NaNs are found in any input --- #
    def contains_nan(arr):
        try:
            return np.isnan(arr).any()
        except TypeError:
            return False  # Not a numeric array, so we ignore

    arrays_to_check = [times, positions, xt_over_t, G]
    if any(contains_nan(arr) for arr in arrays_to_check):
        print("NaNs detected in one or more input arrays — skipping plot_fitting_summary.")
        return

    # --- If the values are without NaNs --- #

    if ax is None:
        fig, axes = plt.subplots(2, 4, figsize=(25, 12))
    else:
        fig = ax.figure
        axes = ax

    axes = axes.reshape(2, 4)  # In case a flattened array is passed

    times_to_plot = np.insert(times, 0, 0)
    pos_to_plot = np.insert(positions, 0, 0)

    # --- Subplot 1: x(t) - Cartesian ---
    axes[0, 0].plot(times_to_plot, pos_to_plot, marker='o', alpha=0.5, label='data', c='b')
    axes[0, 0].plot(times_to_plot, v_mean * times_to_plot, marker='+', label='linear_fit', c='r')
    axes[0, 0].axvline(x=bound_low, ls=':')
    axes[0, 0].axvline(x=bound_high, ls='--')
    axes[0, 0].set_title("x(t) - Cartesian Scale", size=text_size)
    axes[0, 0].set_xlabel("Time (t)", size=text_size)
    axes[0, 0].set_ylabel("Position (x)", size=text_size)
    axes[0, 0].legend(fontsize=text_size)
    axes[0, 0].grid(True)

    # --- Subplot 2: x(t) - Log-Log ---
    axes[1, 0].plot(times_to_plot, pos_to_plot, marker='o', alpha=0.5, label='data', c='b')
    axes[1, 0].plot(times_to_plot, v_mean * times_to_plot, marker='+', label='linear_fit', c='r')
    axes[1, 0].axvline(x=bound_low, ls=':')
    axes[1, 0].axvline(x=bound_high, ls='--')
    axes[1, 0].set_title("x(t) - Log-Log Scale", size=text_size)
    axes[1, 0].set_xlabel("Time (t)", size=text_size)
    axes[1, 0].set_ylabel("Position (x)", size=text_size)
    axes[1, 0].loglog()
    axes[1, 0].legend(fontsize=text_size)
    axes[1, 0].grid(True, which="both", linestyle='--')

    # --- Subplot 3: x(t)/t - Cartesian ---
    axes[0, 1].plot(times[1:], xt_over_t, marker='o', alpha=0.5, label='x(t)/t', c='g')
    axes[0, 1].axvline(x=bound_low, ls=':')
    axes[0, 1].axvline(x=bound_high, ls='--')
    axes[0, 1].axhline(y=vf, c='r', ls=':', label = f"vf = {np.round(vf, rf)} ± {np.round(vf_std, rf)}")
    axes[0, 1].set_title("x(t)/t - Cartesian Scale", size=text_size)
    axes[0, 1].set_xlabel("Time (t)", size=text_size)
    axes[0, 1].set_ylabel("x(t)/t", size=text_size)
    axes[0, 1].legend(fontsize=text_size)
    axes[0, 1].grid(True)

    # --- Subplot 4: x(t)/t - Log-Log ---
    axes[1, 1].plot(times[1:], xt_over_t, marker='o', alpha=0.5, label='x(t)/t', c='g')
    axes[1, 1].axvline(x=bound_low, ls=':')
    axes[1, 1].axvline(x=bound_high, ls='--')
    axes[1, 1].axhline(y=vf, c='r', ls=':', label = f"vf = {np.round(vf, rf)} ± {np.round(vf_std, rf)}")
    axes[1, 1].set_title("x(t)/t - Log-Log Scale", size=text_size)
    axes[1, 1].set_xlabel("Time (t)", size=text_size)
    axes[1, 1].set_ylabel("x(t)/t", size=text_size)
    axes[1, 1].loglog()
    axes[1, 1].legend(fontsize=text_size)
    axes[1, 1].grid(True, which="both", linestyle='--')

    # --- Subplot 5: G - Cartesian ---
    axes[0, 2].plot(times[1:-1], G, marker='o', alpha=0.5, label='G', c='orange')
    axes[0, 2].axvline(x=bound_low, ls=':')
    axes[0, 2].axvline(x=bound_high, ls='--')
    axes[0, 2].axhline(y=wf, c='r', ls='--', label = f"wf = {np.round(wf, rf)} ± {np.round(wf_std, rf)}")
    axes[0, 2].set_title("Log Derivative (G) - Cartesian", size=text_size)
    axes[0, 2].set_xlabel("Time (t)", size=text_size)
    axes[0, 2].set_ylabel("G", size=text_size)
    axes[0, 2].legend(fontsize=text_size)
    axes[0, 2].grid(True)

    # --- Subplot 6: G - Log-Log ---
    axes[1, 2].plot(times[1:-1], G, marker='o', alpha=0.5, label='G', c='orange')
    axes[1, 2].axvline(x=bound_low, ls=':')
    axes[1, 2].axvline(x=bound_high, ls='--')
    axes[1, 2].axhline(y=wf, c='r', ls='--', label = f"wf = {np.round(wf, rf)} ± {np.round(wf_std, rf)}")
    axes[1, 2].set_title("Log Derivative (G) - Log-Log", size=text_size)
    axes[1, 2].set_xlabel("Time (t)", size=text_size)
    axes[1, 2].set_ylabel("G", size=text_size)
    axes[1, 2].loglog()
    axes[1, 2].legend(fontsize=text_size)
    axes[1, 2].grid(True, which="both", linestyle='--')

    # --- Subplot 7: Final result - Cartesian ---
    axes[0, 3].plot(times_to_plot, pos_to_plot, marker='o', alpha=0.5, label='data', c='b')
    axes[0, 3].plot(times[:bound_low], times[:bound_low] * vf, label = f"vf = {np.round(vf, rf)} ± {np.round(vf_std, rf)}", c='r', marker='x')
    axes[0, 3].plot(times[bound_high:], Cf * np.power(times[bound_high:], wf), label = f"wf = {np.round(wf, rf)} ± {np.round(wf_std, rf)}", c='r', marker='+')
    axes[0, 3].axvline(x=bound_low, ls=':')
    axes[0, 3].axvline(x=bound_high, ls='--')
    axes[0, 3].set_title("Final Result - Cartesian", size=text_size)
    axes[0, 3].set_xlabel("Time (t)", size=text_size)
    axes[0, 3].set_ylabel("Position (x)", size=text_size)
    axes[0, 3].legend(fontsize=text_size)
    axes[0, 3].grid(True)

    # --- Subplot 8: Final result - Log-Log ---
    axes[1, 3].plot(times_to_plot, pos_to_plot, marker='o', alpha=0.5, label='data', c='b')
    axes[1, 3].plot(times[:bound_low], vf * times[:bound_low], label = f"vf = {np.round(vf, rf)} ± {np.round(vf_std, rf)}", c='r', marker='x')
    axes[1, 3].plot(times[bound_high:], Cf * np.power(times[bound_high:], wf), label = f"wf = {np.round(wf, rf)} ± {np.round(wf_std, rf)}", c='r', marker='+')
    axes[1, 3].axvline(x=bound_low, ls=':')
    axes[1, 3].axvline(x=bound_high, ls='--')
    axes[1, 3].set_title("Final Result - Log-Log", size=text_size)
    axes[1, 3].set_xlabel("Time (log)", size=text_size)
    axes[1, 3].set_ylabel("Position (log)", size=text_size)
    axes[1, 3].loglog()
    axes[1, 3].legend(fontsize=text_size)
    axes[1, 3].grid(True, which="both", linestyle='--')

    # Done
    plt.tight_layout()
    plt.show()