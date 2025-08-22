# plot_tools



# -------------------------------------------------------------------------------- Librairies -------------------------------------------------------------------------------- #
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

# plt.rcParams['font.size'] = 16
fontsize = 16

# -------------------------------------------------------------------------------- Nucleo (ncl) functions -------------------------------------------------------------------------------- #


# - Fig1. Line 1 - #

def ncl_plot_obstacle(s, l, origin, alpha_mean, text_size=fontsize, ax=None):
    ax.set_title(f'Mean obstacle for s={s} and l={l}', size=text_size)
    ax.plot(alpha_mean, c='b', ls='-', label='mean obstacle', lw=0.10)
    # ax.fill_between(np.arange(0, len(alpha_mean), 1), alpha_mean, step='post', color='b', alpha=0.3, label='accessible binding sites')
    ax.axvline(x=origin, c='r', ls='--', label=f'origin={origin}')
    ax.set_xlabel('x (bp)', fontsize=text_size)
    ax.set_ylabel('alpha', fontsize=text_size)
    ax.set_xlim([0, 50_000])
    ax.set_ylim([0, 1])
    ax.grid(True, which='both')
    ax.legend(fontsize=text_size, loc='upper right')

def ncl_plot_obs_linker_distrib(obs_points, obs_distrib, link_points, link_distrib, text_size=16, ax=None):
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6), sharex=True)
    else:
        fig = ax.figure
        # clear original ax
        ax.clear()
        ax.set_visible(False)
        # split the subplot
        gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=ax.get_subplotspec(), hspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
    # Top plot: obstacles
    ax1.plot(obs_points, obs_distrib, label='obstacles', color='b', alpha=0.75, marker='o')
    ax1.set_title('Obstacle distribution', size=text_size)
    ax1.set_ylabel('distribution', fontsize=text_size)
    # ax1.set_xlim([0,500])
    ax1.set_ylim([-0.1, 1.1])
    ax1.grid(True)
    ax1.legend(fontsize=text_size)
    # Bottom plot: linkers
    ax2.plot(link_points, link_distrib, label='linkers', color='r', alpha=0.75, marker='o')
    ax2.set_title('Linker distribution', size=text_size)
    ax2.set_xlabel('bp', fontsize=text_size)
    ax2.set_ylabel('distribution', fontsize=text_size)
    ax2.set_ylim([-0.1, 1.1])
    # ax2.set_xlim([0,250])
    ax2.grid(True)
    ax2.legend(fontsize=text_size)
    return fig

def ncl_plot_probabilities(mu, theta, p, text_size=fontsize, ax=None):
    ax.set_title(f'Input probability', size=text_size)
    ax.plot(p, label=f'mu={mu} - theta={theta}', c='r', lw=2)
    ax.set_xlim([0, 0+1000])
    ax.set_ylim([-0.005, 0.025])
    ax.set_ylabel('p(d)', size=text_size)
    ax.set_xlabel('d', size=text_size)
    ax.grid(True, which='both')
    ax.legend(fontsize=text_size, loc='upper right')

def ncl_plot_trajectories(tmax, times, results, results_mean, results_med, results_std, v_mean, v_med, text_size=fontsize, ax=None):
    ax.set_title(f'Trajectories', size=text_size)
    # ax.plot(results[0], drawstyle='steps-mid', lw=0.50, c='r', label='trajectories')
    # for _ in range(1, len(results)):
    #     ax.plot(results[_], drawstyle='steps-mid', lw=0.50, c='r')
    for i in range(9, 12):
        ax.plot(results[i], drawstyle='steps-mid', lw=2, ls="--")
    # ax.errorbar(x=times, y=results_mean, yerr=results_std, c='b', ls='-', label=f'mean_trajectory', lw=1)
    ax.plot(times, results_mean, c='r', ls='-', label=f'mean_trajectory \nv_mean={np.round(v_mean,2)}', lw=2)
    # ax.plot(times, results_med, c='g', ls='--', label=f'med_trajectory', lw=1)
    ax.set_xlabel('t', fontsize=text_size)
    ax.set_ylabel('x (bp)', fontsize=text_size)
    ax.set_xlim([0, tmax])
    ax.set_ylim([0, 5_000])
    ax.grid(True, which='both')
    ax.legend(fontsize=text_size, loc='upper left')


# - Fig1. Line 2 - #

def ncl_plot_fpt_distrib_2d(fpt_distrib_2D, tmax, time_bin, text_size=fontsize, ax=None):
    ax.set_title('Distribution of fpts', size=text_size)
    im = ax.imshow(fpt_distrib_2D, aspect='auto', cmap='bwr', origin='lower', vmin=0, vmax=0.01)
    num_bins = fpt_distrib_2D.shape[1]
    x_ticks = np.arange(0, num_bins, step=max(1, num_bins // 10))
    x_labels = x_ticks * time_bin
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('x (bp)', size=text_size)
    ax.set_ylabel('t', size=text_size)
    # ax.set_xlim([0, 10_000])
    ax.set_ylim([0, tmax - 1])
    plt.colorbar(im, ax=ax, label='Value')
    ax.grid(True, which='both')

def ncl_plot_fpt_number(nt, tmax, fpt_number, time_bin, text_size=fontsize, ax=None):
    ax.set_title(f'Number of trajectories that reached', size=text_size)
    x_values = np.arange(len(fpt_number)) * time_bin
    ax.plot(x_values, fpt_number, label='number', color='b', alpha=0.7, marker='s')
    ax.set_xlabel('x (bp)', fontsize=text_size)
    ax.set_ylabel('number of trajectories', fontsize=text_size)
    ax.set_xlim([0, 10_000])
    ax.set_ylim([-200, nt+200])
    ax.grid(True, which='both')
    ax.legend(fontsize=text_size, loc='upper right')

def ncl_plot_waiting_times(tbj_points, tbj_distrib, text_size=fontsize, ax=None):
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

def ncl_plot_speed_distribution(vi_points, vi_distrib, vi_mean, vi_med, vi_mp, text_size=fontsize, ax=None):
    ax.set_title(f'Distribution of instantaneous speeds', size=text_size)
    # ax.axvline(x=vi_mp, label=f'most probable : {np.round(vi_mp,2)}', c='r', ls='-')
    ax.axvline(x=vi_med, label=f'vi_med = {np.round(vi_med,2)}', c='r', ls='--')
    ax.plot(vi_points, vi_distrib, c='b')
    ax.grid(True, which='both')
    ax.set_xlabel('speeds', size=text_size)
    ax.set_ylabel('distribution', size=text_size)
    ax.set_ylim([1e-5, 1e-1])
    ax.set_xlim([1e-1, 1e6])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=text_size)


# - Fig2. Line 1 + Line 2 - #

def ncl_plot_fitting_summary(times, positions, v_mean,
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


# -------------------------------------------------------------------------------- Marcand (mrc) functions -------------------------------------------------------------------------------- #


# - Fig1. Line 1 + Line 2 - #

def mrc_plot_obstacle(alpha_mean, text_size=fontsize, ax=None):
    ax.set_title(f'Mean obstacle', size=text_size)
    ax.plot(alpha_mean, c='b', ls='-', label='mean obstacle')
    ax.fill_between(np.arange(0, len(alpha_mean), 1), alpha_mean, step='post', color='b', alpha=0.3, label='accessible binding sites')
    ax.set_xlabel('x (bp)', fontsize=text_size)
    ax.set_ylabel('alpha', fontsize=text_size)
    ax.set_xlim([0, 2_000])
    ax.set_ylim([0, 1])
    ax.grid(True, which='both')
    ax.legend(fontsize=text_size, loc='upper right')

def mrc_plot_probabilities(mu, theta, p, text_size=fontsize, ax=None):
    ax.set_title(f'Input probability with for mu={mu} and theta={theta}', size=text_size)
    ax.plot(p, label='probability distribution', c='r', lw=2)
    ax.set_xlim([0, 0+1000])
    ax.set_ylim([-0.05, 0.20])
    ax.set_ylabel('p(d)', size=text_size)
    ax.set_xlabel('d', size=text_size)
    ax.grid(True, which='both')
    ax.legend(fontsize=text_size, loc='upper right')

def mrc_plot_trajectories(tmax, times, results, results_mean, results_med, results_std, v_mean, v_med, text_size=fontsize, ax=None):
    ax.set_title(f'Trajectories', size=text_size)
    ax.plot(results[0], drawstyle='steps-mid', lw=0.25, c='r', label='trajectories')
    for _ in range(1, len(results)):
        ax.plot(results[_], drawstyle='steps-mid', lw=0.25, c='r')
    # ax.errorbar(x=times, y=results_mean, yerr=results_std, c='b', ls='-', label=f'mean_trajectory', lw=1)
    ax.plot(times, results_mean, c='b', ls='-', label=f'mean_trajectory', lw=1)
    # ax.plot(times, results_med, c='g', ls='--', label=f'med_trajectory', lw=1)
    ax.set_xlabel('t', fontsize=text_size)
    ax.set_ylabel('x (bp)', fontsize=text_size)
    ax.set_xlim([0, tmax])
    # ax.set_ylim([0, 10_000])
    ax.grid(True, which='both')
    ax.legend(fontsize=text_size, loc='upper left')

def mrc_plot_pos1D(shifted_data, num_bins, L, alpha_mean, Lmax, origin, text_size=fontsize, ax=None):
    ax.set_title(f'Histogram of positions 1D', size=text_size)
    ax.plot(L, alpha_mean * np.mean(shifted_data), label='mean_obstacle', linewidth=0.5, color='b', alpha=0.2)
    ax.hist(shifted_data, bins=num_bins, label='all_position_distribution', color='r')
    ax.axvline(x=origin, label='interval_of_jumps', color='r', linestyle='--')
    ax.axvline(x=Lmax -origin, color='r', linestyle='--')
    ax.axvline(x=0, color='b', linestyle='--')
    ax.axvline(x=Lmax, label='interval_of_obstacles', color='b', linestyle='--')
    ax.set_xlabel('x (bp)', size=text_size)
    ax.set_ylabel('count', size=text_size)
    ax.grid(True, which='both')
    ax.legend(fontsize=text_size, loc='upper left')
  
def mrc_plot_for_pos_2D(p_hist_list, tmax, Lmax, origin, text_size=fontsize, ax=None):
    ax.set_title(f'Histogram of positions 2D', size=text_size)
    ax.imshow(p_hist_list, aspect='auto', origin='lower', cmap='bwr', vmin=0.00001, vmax=0.001)
    ax.set_xlabel('t', fontsize=text_size)
    ax.set_ylabel('x (bp)', fontsize=text_size)
    ax.set_xticks(np.arange(0, int(tmax + 1), 20))
    ax.set_yticks(np.arange(0, int(Lmax - (2 * origin) + 1), 2000))
    ax.grid(True, which='both')
    ax.legend(fontsize=text_size, loc='upper left')

def mrc_plot_for_jump_distribution(jt_bj, jt_bj_hist, num_bins, vmean, text_size=fontsize, ax=None):   
    ax.set_title('Distribution of times between jumps') 
    if vmean != 0:
        plt.hist(jt_bj, bins=num_bins, density=1, label='hist', color='b', alpha=0.7)
    ax.plot(np.arange(len(jt_bj_hist)), jt_bj_hist, 'r-', label='data')
    ax.set_xlabel('t', size=text_size)
    ax.set_ylabel('density', size=text_size)
    ax.grid(True, which='both')
    ax.legend(fontsize=text_size, loc='upper left')

def mrc_plot_fpt_distrib_2d(fpt_distrib_2D, fpt_number, Lmax, origin, text_size=fontsize, ax=None):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), num='fpt_distrib')
    ax.set_title('Distribution of fpts', size=text_size)   
    axes[0].set_title('Histogram', fontsize=text_size)
    axes[0].set_xlabel('x (bp)', fontsize=text_size)
    axes[0].set_ylabel('t', fontsize=text_size)
    axes[0].imshow(fpt_distrib_2D, aspect='auto', origin='lower', cmap='bwr', vmin=0, vmax=0.1)
    axes[0].set_xlim(0, int(Lmax - 2 * origin))
    axes[0].grid(True, which='both')
    axes[1].set_title('Number of Trajectories That Reached the Positions', fontsize=text_size)
    axes[1].set_xlabel('x (bp)', fontsize=text_size)
    axes[1].set_ylabel('n', fontsize=text_size)
    axes[1].plot(fpt_number, c='b')
    axes[1].set_xlim(0, int(Lmax - 2 * origin))
    axes[1].grid(True, which='both')
    plt.tight_layout()

def mrc_plot_speed_distribution(vi_points, vi_distrib, vi_mean, vi_med, vi_mp, text_size=fontsize, ax=None):
    ax.set_title(f'Distribution of instantaneous speeds', size=text_size)
    ax.axvline(x=vi_mp, label=f'most probable : {np.round(vi_mp,2)}', c='r', ls='-')
    ax.axvline(x=vi_med, label=f'median : {np.round(vi_med,2)}', c='r', ls='--')
    ax.plot(vi_points, vi_distrib, c='b', label='instantaneous speeds')
    ax.grid(True, which='both')
    ax.set_xlabel('speeds', size=text_size)
    ax.set_ylabel('distribution', size=text_size)
    ax.set_ylim([1e-5, 1e-1])
    ax.set_xlim([1e-1, 1e6])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=text_size)


# - Fig2. Line 1 + Line 2 - #

def mrc_plot_mean_fpt(mean_results, mean_alpha, text_size=fontsize, ax=None):
    ax.set_title(f'FPT mean', size=text_size)
    ax.plot(np.arange(0,len(mean_results)), mean_results, 'r', label='Mean_fpt')
    ax.plot(np.arange(len(mean_alpha)), mean_alpha*int(np.mean(mean_results)), 'b-', lw=0.5, label='Mean_obstacle')
    ax.xlabel('x (bp)')
    ax.ylabel('t')
    ax.axvline(x=int(2000))
    ax.axvline(x=int(1000-477))
    ax.axvline(x=int(1000+477))
    ax.axvline(x=2000+442)
    ax.grid(True, which='both')
    ax.legend(fontsize=text_size)

def mrc_plot_map_2D_fpt(fpt_2D, text_size=fontsize, ax=None):
    ax.set_title('Distribution of FPTs', fontsize=text_size)
    ax.imshow(fpt_2D.T, cmap='bwr', aspect='auto', origin='lower', vmin=0.001, vmax=0.1)
    ax.xlabel('x (bp)')
    ax.ylabel('t')
    ax.grid(True, which='both')
    ax.legend(fontsize=text_size)
    plt.colorbar()

def mrc_plot_distrib_fpt_xmax(bins_fpt_xmax, counts_fpt_xmax_normalized, text_size=fontsize, ax=None):
    ax.set_title('Distribution of FPT (xmax)', fontsize=text_size)
    # ax.hist(fpt_x_max, bins=bins_fpt, color='b', edgecolor='k', alpha=0.7, label='hist', density=1)
    ax.plot(bins_fpt_xmax, counts_fpt_xmax_normalized, marker='+', ls='-', c='r', label='plot')        
    ax.xlabel('t')
    ax.ylabel('density')
    ax.grid(True, which='both')
    ax.legend(fontsize=text_size)

def mrc_plot_distrib_diff_fpt(fpt_xmax, bins_fpt, bins_fpt_xmax, counts_fpt_xmax_normalized, text_size=fontsize, ax=None):
    title_fpt_x_max = 'Distribution of FPT (xmax)'
    ax.set_title(title_fpt_x_max, fontsize=text_size)
    ax.hist(fpt_xmax, bins=bins_fpt, color='b', edgecolor='k', alpha=0.7, label='hist', density=1)
    ax.plot(bins_fpt_xmax, counts_fpt_xmax_normalized, marker='+', ls='-', c='r', label='plot')        
    ax.xlabel('t')
    ax.ylabel('density')
    ax.grid(True, which='both')
    ax.legend(fontsize=text_size)

def mrc_plot_proba_pass(counts_fptxmax, text_size=fontsize, ax=None):
    ax.set_title('tau (fpt (xmax))', fontsize=text_size)
    # ax.hist(result_x_max, bins=bins_fpt, color='b', edgecolor='k', alpha=0.7, label='hist', density=1)
    ax.plot(counts_fptxmax, marker='+', ls='-', c='r', label='plot')        
    ax.xlabel('t')
    ax.ylabel('density')
    ax.grid(True, which='both')
    ax.legend(fontsize=text_size)

def mrc_plot_delay(delay, text_size=fontsize, ax=None) :
    ax.set_title('delay')
    ax.plot(np.arange(len(delay)), delay, label='abs(delay)', c='b')
    ax.xlabel('x (bp)')
    ax.ylabel('delay')
    ax.grid(True, which='both')
    ax.legend(fontsize=text_size)

def mrc_plot_t_pass(t_pass, bins_t_pass, text_size=fontsize, ax=None):
    ax.set_title('t_pass')
    ax.plot(bins_t_pass, t_pass, label='t_pass', marker='+', ls='-', c='b')
    ax.grid(True, which='both')
    ax.legend(fontsize=text_size)

def mrc_plot_for_results_marcand(fpt_mean, fpt_2D, fpt_xmax, tau_fpt_x_max, 
                             bins_fpt_x_max, counts_fpt_xmax_normalized, 
                             v_marcand, p_pass, alpha_mean, t_pass, bins_t_pass):
    
    # I : Plot the average first pass time for each position
    mrc_plot_mean_fpt(fpt_mean, alpha_mean)
    # II : Plot the 2D histogram of all first pass times
    mrc_plot_map_2D_fpt(fpt_2D)
    # III : Plot the histogram of the last first pass time
    mrc_plot_distrib_fpt_xmax(bins_fpt_x_max, counts_fpt_xmax_normalized)
    # IV : Plot the cumulative probability of passage (tau)
    mrc_plot_proba_pass(tau_fpt_x_max)
    # V : p_pass
    mrc_plot_delay(p_pass)
    # VI : t_pass
    mrc_plot_t_pass(t_pass, bins_t_pass)


# - Fig3. Line 1 + Line 2 + Line 3 - #

def mrc_plot_summary(gap_value, mu_value, theta_value, alpha_mean, p, fpt_mean, fpt_2D, fpt_xmax, p_tau, v_marcand, text_size=fontsize):
    # Plotting
    fig, axs = plt.subplots(3, 2, figsize=(14, 18))

    # Obstacle (mean_alpha)
    axs[0, 0].plot(alpha_mean, label='alpha', color='b', lw=1)
    axs[0, 0].set_title("Obstacle", fontsize=text_size)
    axs[0, 0].set_xlabel('x (bp)', fontsize=text_size)
    axs[0, 0].set_ylabel('acceptance', fontsize=text_size)
    axs[0, 0].grid(True)
    axs[0, 0].legend(fontsize=text_size, loc='upper right')

    # Probabilities
    axs[0, 1].plot(np.arange(len(p)), p, 'r-', lw=2, label='jump_probability')
    axs[0, 1].set_title("Probabilities of jump distance", fontsize=text_size)
    axs[0, 1].set_xlabel('d (bp)', fontsize=text_size)
    axs[0, 1].set_ylabel('p(d ; mu,theta)', fontsize=text_size)
    axs[0, 1].set_xlim(0, 1e3)
    axs[0, 1].grid(True)
    axs[0, 1].legend(fontsize=text_size, loc='upper right')

    # Mean FPT
    axs[1, 0].plot(fpt_mean, 'r', label='Mean_fpt')
    axs[1, 0].plot(alpha_mean * np.mean(fpt_mean), 'b-', lw=0.5, label='Mean_obstacle')
    axs[1, 0].axvline(x=2000 + 442, ls='--', c='k', label="limit")
    axs[1, 0].set_title("Mean FPT", fontsize=text_size)
    axs[1, 0].set_xlabel('x (bp)', fontsize=text_size)
    axs[1, 0].set_ylabel('t', fontsize=text_size)
    axs[1, 0].grid(True)
    axs[1, 0].legend(fontsize=text_size, loc='upper left')

    # FPT 2D heatmap
    im = axs[1, 1].imshow(fpt_2D.T, cmap='bwr', aspect='auto', origin='lower', vmin=0.001, vmax=0.01)
    axs[1, 1].set_title("Distribution of FPTs", fontsize=text_size)
    axs[1, 1].set_xlabel('x (bp)', fontsize=text_size)
    axs[1, 1].set_ylabel('t', fontsize=text_size)
    axs[1, 1].grid(True)
    fig.colorbar(im, ax=axs[1, 1])

    # FPT xmax
    axs[2, 0].plot(fpt_xmax, marker='+', ls='-', c='r', label='plot')
    axs[2, 0].set_title("distrib_fpt_x_max", fontsize=text_size)
    axs[2, 0].set_xlabel('t', fontsize=text_size)
    axs[2, 0].set_ylabel('density', fontsize=text_size)
    axs[2, 0].grid(True)
    axs[2, 0].legend(fontsize=text_size)

    # p_tau
    axs[2, 1].plot(p_tau, marker='+', ls='-', c='r', label='plot')
    axs[2, 1].set_title("p_tau", fontsize=text_size)
    axs[2, 1].set_xlabel('t', fontsize=text_size)
    axs[2, 1].set_ylabel('density', fontsize=text_size)
    axs[2, 1].grid(True)
    axs[2, 1].legend(fontsize=text_size)

    # Speed value
    fig.text(0.35, 0.01, f"Processivity : {v_marcand:.2f} bp/s", fontsize=16, weight='bold')

    plt.suptitle(f'[gap={gap_value} - mu={mu_value} - theta={theta_value}]', fontsize=20)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()