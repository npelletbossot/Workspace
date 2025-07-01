# plot_tools



# --- Librairies --- #
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec


# --- Nucleo functions --- #

# plt.rcParams['font.size'] = 16
fontsize = 16


# - Line 1 - #

def plot_obstacle(s, l, origin, alpha_mean, text_size=fontsize, ax=None):
    ax.set_title(f'Mean obstacle for s={s} and l={l}', size=text_size)
    ax.plot(alpha_mean, c='b', ls='-', label='mean obstacle')
    ax.fill_between(np.arange(0, len(alpha_mean), 1), alpha_mean, step='post', color='b', alpha=0.3, label='accessible binding sites')
    ax.axvline(x=origin, c='r', ls='--', label=f'origin={origin}')
    ax.set_xlabel('x (bp)', fontsize=text_size)
    ax.set_ylabel('alpha', fontsize=text_size)
    ax.set_xlim([0, 50_000])
    ax.set_ylim([0, 1])
    ax.grid(True, which='both')
    ax.legend(fontsize=text_size, loc='upper right')

def plot_obs_linker_distrib(obs_points, obs_distrib, link_points, link_distrib, text_size=16, ax=None):
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
    ax1.grid(True)
    ax1.legend(fontsize=text_size)

    # Bottom plot: linkers
    ax2.plot(link_points, link_distrib, label='linkers', color='r', alpha=0.75, marker='o')
    ax2.set_title('Linker distribution', size=text_size)
    ax2.set_xlabel('bp', fontsize=text_size)
    ax2.set_ylabel('distribution', fontsize=text_size)
    # ax2.set_xlim([0,250])
    ax2.grid(True)
    ax2.legend(fontsize=text_size)

    return fig

def plot_probabilities(mu, theta, p, text_size=fontsize, ax=None):
    ax.set_title(f'Input probability with for mu={mu} and theta={theta}', size=text_size)
    ax.plot(p, label='probability distribution', c='r', lw=2)
    ax.set_xlim([0, 0+1000])
    ax.set_ylim([-0.05, 0.20])
    ax.set_ylabel('p(d)', size=text_size)
    ax.set_xlabel('d', size=text_size)
    ax.grid(True, which='both')
    ax.legend(fontsize=text_size, loc='upper right')

def plot_trajectories(tmax, times, results, results_mean, results_med, results_std, v_mean, v_med, text_size=fontsize, ax=None):
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


# - Line 2 - #

def plot_fpt_distrib_2d(fpt_distrib_2D, tmax, time_bin, text_size=fontsize, ax=None):
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

def plot_fpt_number(nt, tmax, fpt_number, time_bin, text_size=fontsize, ax=None):
    ax.set_title(f'Number of trajectories that reached', size=text_size)
    x_values = np.arange(len(fpt_number)) * time_bin
    ax.plot(x_values, fpt_number, label='number', color='b', alpha=0.7, marker='s')
    ax.set_xlabel('x (bp)', fontsize=text_size)
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

def plot_speed_distribution(vi_points, vi_distrib, vi_mean, vi_med, vi_mp, text_size=fontsize, ax=None):
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


# - Line 3 + Line 4 - #


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



# --- Marcand functions --- #




# #-- Probabilities plot --#
# def plot_for_probabilities(_alpha_list_, _P_, _title_0_, _title_size_, _x_size_, _y_size_, _legend_size_, _saving_):
#     if _saving_:
#         fig_proba = plt.figure(figsize=(8, 6), num='probabilities')
#         title_proba = 'probabilities__' + _title_0_
#         plt.title(title_proba, fontsize=_title_size_)
#         plt.step(np.arange(0, len(_alpha_list_[0]), 1), _alpha_list_[0], color='b', lw=0.5, label='portion_of_obstacle')
#         plt.plot(np.arange(0, len(_P_), 1), _P_, 'r-', lw=2, label='jump_probability')
#         plt.xlabel('x_in_bp', fontsize=_x_size_)
#         plt.ylabel('p__x_k_theta', fontsize=_y_size_)
#         plt.xlim(0, 1e3)
#         plt.grid(True)
#         plt.legend(fontsize=_legend_size_, loc='upper right')
        
#         savepath = os.path.join(_title_0_, title_proba + '.png')
#         plt.savefig(savepath)
#         plt.close(fig_proba)

#         return None
# #--   --#


# #-- Obstacles plot --#
# def plot_for_obstacles(_dict_plot_x_, _dict_plot_y_, _title_0_, _title_size_, _x_size_, _y_size_, _legend_size_, _saving_):
#     if _saving_:
#         fig_obs = plt.figure(figsize=(8, 6), num='obs_disp')
#         title_obs_disp = 'obstacle_dispersion__' + _title_0_
#         plt.title(title_obs_disp, fontsize=_title_size_)
#         plt.plot(_dict_plot_x_, _dict_plot_y_, label='plot', color='b')
#         plt.scatter(_dict_plot_x_, _dict_plot_y_, label='points', marker='D', color='r', alpha=0.3)
#         plt.xlabel('size_of_nuclosome_in_bp', fontsize=_x_size_)
#         plt.ylabel('count', fontsize=_y_size_)
#         plt.grid(True)
#         plt.legend(fontsize=_legend_size_, loc='upper right')

#         savepath = os.path.join(_title_0_, title_obs_disp + '.png')
#         plt.savefig(savepath)
#         plt.close(fig_obs)

#         return None
# #--   --#


# #-- Trajectories plot --#
# def plot_for_trajectories(_results_, _mean_results_, _v_mean_, _err_v_, _t_max_, _L_max_, _origin_, _title_0_, _dt_, _title_size_, _x_size_, _y_size_, _legend_size_, _saving_):
#     if _saving_:
#         fig_traj = plt.figure(figsize=(8, 6), num='trajectories')
        
#         for n in range(len(_results_)):
#             plt.step(np.arange(1, _t_max_ + 1, 1), _results_[n], lw=0.2, c='b')

#         title_trajectories = 'trajectories__' + _title_0_
#         plt.title(title_trajectories, fontsize=_title_size_)
#         plt.axhline(y=_L_max_- int(2 * _origin_), color='grey', linestyle='--', label=f'perturbations {_L_max_ - int(2 * _origin_)} [bp]')
#         plt.axvline(x=_t_max_, color='grey', linestyle=':', label=f't_max {_t_max_} [s]')
#         plt.plot(np.arange(1, _t_max_ + 1, 1), _mean_results_, color='r', label=f'mean_trajectory')
#         plt.plot(np.arange(1, _t_max_ + 1, 1), np.round(_v_mean_, 2) * np.arange(0, _t_max_, 1), label=f'linear_speed v_mean: {_v_mean_} __ err_v: {_err_v_}', ls='--', c='r')
#         plt.xlabel(f't_in_{_dt_}s', fontsize=_x_size_)
#         plt.ylabel('x_in_bp', fontsize=_y_size_)
#         plt.grid(True)
#         plt.legend(fontsize=_legend_size_, loc='upper left')
        
#         savepath = os.path.join(_title_0_, title_trajectories + '.png')
#         plt.savefig(savepath)
#         plt.close(fig_traj)

#         return None
# #--   --#



# #-- All positions plot --#
# def plot_for_pos_1D(_shifted_data_, _num_bins_, _L_, _mean_alpha_, _L_max_, _origin_, _title_0_, _title_size_, _x_size_, _y_size_, _legend_size_, _saving_):
#     if _saving_:
#         fig_all_pos = plt.figure(figsize=(8, 6), num='all_positions')
#         title_all_pos = 'positions__' + _title_0_
#         plt.title(title_all_pos, fontsize=_title_size_)
#         plt.plot(_L_[0:_L_max_], _mean_alpha_[0:_L_max_] * np.mean(_shifted_data_), label='mean_obstacle', linewidth=0.5, color='b', alpha=0.2)
#         plt.hist(_shifted_data_, bins=_num_bins_, label='all_position_distribution', color='r')
#         plt.axvline(x=_origin_, label='interval_of_jumps', color='r', linestyle='--')
#         plt.axvline(x=_L_max_-_origin_, color='r', linestyle='--')
#         plt.axvline(x=0, color='b', linestyle='--')
#         plt.axvline(x=_L_max_, label='interval_of_obstacles', color='b', linestyle='--')
#         plt.xlabel('x_in_bp', fontsize=_x_size_)
#         plt.ylabel('count', fontsize=_y_size_)
#         plt.grid(True)
#         plt.legend(fontsize=_legend_size_, loc='upper left')
        
#         savepath = os.path.join(_title_0_, title_all_pos + '.png')
#         plt.savefig(savepath)
#         plt.close(fig_all_pos)

#         return None
# #--   --#


# #-- Position hist 2D --#
# def plot_for_pos_2D(_p_hist_list_, _t_max_, _L_max_, _origin_, _title_0_, _x_size_, _y_size_, _title_size_, _saving_):
#     if _saving_:
#         fig_hist_pos = plt.figure(figsize=(8, 6), num='pos_distrib')
#         title_hist_pos = 'position_hists__' + _title_0_
#         plt.title(title_hist_pos, fontsize=_title_size_)
#         plt.imshow(_p_hist_list_, aspect='auto', origin='lower', cmap='bwr', vmin=0.00001, vmax=0.001)
#         plt.xlabel('t_in_s', fontsize=_x_size_)
#         plt.ylabel('x_in_bp', fontsize=_y_size_)
#         plt.xticks(np.arange(0, int(_t_max_ + 1), 20))
#         plt.yticks(np.arange(0, int(_L_max_ - (2 * _origin_) + 1), 2000))
#         plt.grid(True)
#         # Sauvegarde du graphique
#         savepath = os.path.join(_title_0_, title_hist_pos + '.png')
#         plt.savefig(savepath)
#         plt.close(fig_hist_pos)

#         return None
# #--   --#


# #-- Time between jumps --#
# def plot_for_jump_distribution(_jt_bj_, _jt_bj_hist_, _title_0_, _title_size_, _x_size_, _y_size_, _legend_size_, _v_mean_, _num_bins_, _saving_):
#     if _saving_:
        
#         fig_hist_jump = plt.figure(figsize=(8, 6), num='jump_distrib')
#         title_jumps = 'time_between_jumps_hist__' + _title_0_
#         plt.title(title_jumps, fontsize=_title_size_)
#         plt.xlabel('t_in_s', fontsize=_x_size_)
#         plt.ylabel('density', fontsize=_y_size_)
        
#         if _v_mean_ != 0:
#             plt.hist(_jt_bj_, bins=_num_bins_, density=1, label='hist', color='b', alpha=0.7)
        
#         plt.plot(np.arange(len(_jt_bj_hist_)), _jt_bj_hist_, 'r-', label='data')
#         plt.legend(fontsize=_legend_size_, loc='upper right')
#         plt.grid(True)
        
#         savepath = os.path.join(_title_0_, title_jumps + '.png')
#         plt.savefig(savepath)
#         plt.close(fig_hist_jump)

#         return None
# #--   --#


# #--   --#
# def plot_for_fpt_distribution(_fpt_results_, _fpt_number_, _title_0_, _title_size_, _x_size_, _y_size_, _L_max_, _origin_, nt, _saving_):
#     if _saving_:

#         fig_hist_fpt, axes = plt.subplots(2, 1, figsize=(8, 6), num='fpt_distrib')
#         title_hist_fpt = 'first_pass_time__' + _title_0_
#         plt.suptitle(title_hist_fpt, fontsize=_title_size_)
        
#         axes[0].set_title('Histogram', fontsize=_title_size_)
#         axes[0].set_xlabel('x_in_bp', fontsize=_x_size_)
#         axes[0].set_ylabel('t_in_s', fontsize=_y_size_)
#         axes[0].imshow(_fpt_results_, aspect='auto', origin='lower', cmap='bwr', vmin=0, vmax=0.1)
#         axes[0].set_xlim(0, int(_L_max_ - 2 * _origin_))
#         axes[0].grid(True)
        
#         axes[1].set_title('Number of Trajectories That Reached the Positions', fontsize=_title_size_)
#         axes[1].set_xlabel('x_in_bp', fontsize=_x_size_)
#         axes[1].set_ylabel('n', fontsize=_y_size_)
#         axes[1].plot(_fpt_number_, c='b')
#         axes[1].set_xlim(0, int(_L_max_ - 2 * _origin_))
#         axes[1].set_ylim(0, nt)
#         axes[1].grid(True)
        
#         plt.tight_layout()
#         savepath = os.path.join(_title_0_, title_hist_fpt + '.png')
#         plt.savefig(savepath)
#         plt.close(fig_hist_fpt)

#         return None
# #--   --#


# #-- Instanteneous speeds plots --#
# def plot_for_instantaneous_speeds(_bin_centers_, _speed_hist_, _v_inst_mean_, _v_inst_med_, _mp_v_center_, _title_0_, _title_size_, _x_size_, _y_size_, _legend_size_, _saving_, _epsilon_):
#     if _saving_:
#         fig_hist_speeds = plt.figure(figsize=(8, 6), num='speeds')
#         title_speeds = 'speed_hists__' + _title_0_
#         plt.title(title_speeds, fontsize=_title_size_)
#         plt.plot(_bin_centers_, _speed_hist_, 'b-', label='distribution_plot')
#         plt.axvline(x=_v_inst_mean_, c='r', ls=':', label=f'v_mean:{_v_inst_mean_:.2f}')
#         plt.axvline(x=_v_inst_med_, c='r', ls='--', label=f'v_med:{_v_inst_med_:.2f}')
#         plt.axvline(x=_mp_v_center_, c='r', ls='-.', label=f'v_most_probable:{_mp_v_center_:.2f}')
#         plt.xlabel('speeds', fontsize=_x_size_)
#         plt.ylabel('density', fontsize=_y_size_)

#         # if (_speed_hist_ > _epsilon_).all():
#         #     plt.xscale('log')
            
#         _speed_hist_ = _speed_hist_[_speed_hist_ > _epsilon_]
#         if len(_speed_hist_) > 0:
#             plt.xscale('log')


#         plt.xlim(1e-1, 1e5)
#         plt.grid(True)
#         plt.legend(fontsize=_legend_size_, loc='upper right')
        
#         savepath = os.path.join(_title_0_, title_speeds + '.png')
#         plt.savefig(savepath)
#         plt.close(fig_hist_speeds)

#         return None
# #--   --#



#----------------------------------------------------------------------------------------------- Plot functions -----------------------------------------------------------------------------------------------#



# #- Mean fpt in function of x -#
# def plot_mean_fpt(mean_results, mean_alpha, title_0, title_size, legend_size, saving):
#     if saving :
#         fig_fpt_mean = plt.figure(figsize=(8,6), num='mean_fpt')
#         title_mean = 'mean_fpt__'+title_0
#         plt.title(title_mean, fontsize=title_size)
#         plt.plot(np.arange(0,len(mean_results)), mean_results, 'r', label='Mean_fpt')
#         plt.plot(np.arange(len(mean_alpha)), mean_alpha*int(np.mean(mean_results)), 'b-', lw=0.5, label='Mean_obstacle')
#         plt.xlabel('x_[bp]')
#         plt.ylabel('t_[s]')
#         plt.axvline(x=int(2000))
#         plt.axvline(x=int(1000-477))
#         plt.axvline(x=int(1000+477))
#         plt.axvline(x=2000+442)
#         plt.grid(True)
#         plt.legend(fontsize=legend_size, loc='upper left')
#         # show
#         savepath = os.path.join(title_0, title_mean+'.png')
#         plt.savefig(savepath)
#         plt.close(fig_fpt_mean)
#     return None
# #- -#


# #- 2D mapping of fpts -#
# def plot_map_2D_fpt(fpt_2D, title_0, title_size, saving):
#     if saving: 
#         fig_fpt_marcand_2D = plt.figure(figsize=(8,6), num='fpt_2D')
#         title_2D = 'distrib_of_all_fpt__'+title_0
#         plt.title(title_2D, fontsize=title_size)
#         plt.imshow(fpt_2D.T, cmap='bwr', aspect='auto', origin='lower', vmin=0.001, vmax=0.1)
#         plt.xlabel('x_[bp]')
#         plt.ylabel('t_[s]')
#         plt.grid(True)
#         plt.colorbar()
#         # show
#         savepath = os.path.join(title_0, title_2D+'.png')
#         plt.savefig(savepath)
#         plt.close(fig_fpt_marcand_2D)
#     return None
# #- -#


# #- Distribution of fpt(x_max) -#
# def plot_distrib_fpt_x_max(bins_fpt_x_max, counts_fpt_xmax_normalized, title_0, title_size, legend_size, saving):
#     if saving :
#         fig_fpt_x_max = plt.figure(figsize=(8,6), num='fpt_x_max')
#         title_fpt_x_max = 'distrib_fpt_x_max__'+title_0
#         plt.title(title_fpt_x_max, fontsize=title_size)
#         # plt.hist(fpt_x_max, bins=bins_fpt, color='b', edgecolor='k', alpha=0.7, label='hist', density=1)
#         plt.plot(bins_fpt_x_max, counts_fpt_xmax_normalized, marker='+', ls='-', c='r', label='plot')        
#         plt.xlabel('t_[s]')
#         plt.ylabel('density')
#         plt.grid(True)
#         plt.legend(fontsize=legend_size, loc='upper left')
#         # show
#         savepath = os.path.join(title_0, title_fpt_x_max+'.png')
#         plt.savefig(savepath)
#         plt.close(fig_fpt_x_max)
#     return None
# #- -#


# #- Distribution of fpt(x_max) -#
# def plot_distrib_diff_fpt(fpt_x_max, bins_fpt, bins_fpt_x_max, counts_fpt_xmax_normalized, title_0, title_size, legend_size, saving):
#     if saving :
#         fig_fpt_x_max = plt.figure(figsize=(8,6), num='fpt_x_max')
#         title_fpt_x_max = 'distrib_fpt_x_max__'+title_0
#         plt.title(title_fpt_x_max, fontsize=title_size)
#         plt.hist(fpt_x_max, bins=bins_fpt, color='b', edgecolor='k', alpha=0.7, label='hist', density=1)
#         plt.plot(bins_fpt_x_max, counts_fpt_xmax_normalized, marker='+', ls='-', c='r', label='plot')        
#         plt.xlabel('t_[s]')
#         plt.ylabel('density')
#         plt.grid(True)
#         plt.legend(fontsize=legend_size, loc='upper left')
#         # show
#         savepath = os.path.join(title_0, title_fpt_x_max+'.png')
#         plt.savefig(savepath)
#         plt.close(fig_fpt_x_max)
#     return None
# #- -#


# #- Probabilities : 1-ditrib(fpt(f_max))
# def plot_proba_pass(counts_fpt_xmax, title_0, title_size, legend_size, saving):
#     if saving : 
#         fig_fpt_proba = plt.figure(figsize=(8,6), num='fpt_proba')
#         title_fpt_proba = 'tau_fpt_x_max__'+title_0
#         plt.title(title_fpt_proba, fontsize=title_size)
#         # plt.hist(result_x_max, bins=bins_fpt, color='b', edgecolor='k', alpha=0.7, label='hist', density=1)
#         plt.plot(counts_fpt_xmax, marker='+', ls='-', c='r', label='plot')        
#         plt.xlabel('t_[s]')
#         plt.ylabel('density')
#         plt.grid(True)
#         plt.legend(fontsize=legend_size, loc='upper left')
#         # show
#         savepath = os.path.join(title_0, title_fpt_proba+'.png')
#         plt.savefig(savepath)
#         plt.close(fig_fpt_proba)
#     return None
# #- -#


# def plot_delay(_delay_, title_0, saving) :
#     if saving : 
#         fig_delay = plt.figure(figsize=(8,6), num='delay_fig')
#         title_delay_marcand = 'delay__'+title_0
#         plt.figure(figsize=(8,6))
#         plt.title(title_delay_marcand)
#         plt.plot(np.arange(len(_delay_)), _delay_, label='abs(delay)', c='b')
#         plt.xlabel('x_[bp]')
#         plt.ylabel('delay_[s]')
#         plt.grid(True)
#         plt.legend()
#         # show
#         savepath = os.path.join(title_0, title_delay_marcand+'.png')
#         plt.savefig(savepath)
#         plt.close(fig_delay)    
#     return None

# def plot_t_pass(t_pass, bins_t_pass, title_0, saving):
#     if saving : 
#         title_t_marcand = 't_pass__'+title_0
#         plt.figure(figsize=(8,6))
#         plt.title(title_t_marcand)
#         plt.plot(bins_t_pass, t_pass, label='t_pass', marker='+', ls='-', c='b')
#         plt.grid(True)
#         plt.legend()
#         # show
#         savepath = os.path.join(title_0, title_t_marcand+'.png')
#         plt.savefig(savepath)
#         plt.close(title_t_marcand)    
#     return None


# #- -#
# def plot_for_results_marcand(fpt_mean, fpt_2D, fpt_x_max, tau_fpt_x_max, 
#                              bins_fpt_x_max, counts_fpt_xmax_normalized, 
#                              v_marcand, p_pass, obs_normalized, alpha_mean, dist_diff_fpt, t_pass, bins_t_pass,
#                              title_0, title_size, legend_size, saving):
    
#     # I : Plot the average first pass time for each position
#     plot_mean_fpt(fpt_mean, alpha_mean, title_0, title_size, legend_size, saving)

#     # II : Plot the 2D histogram of all first pass times
#     plot_map_2D_fpt(fpt_2D, title_0, title_size, saving)

#     # III : Plot the histogram of the last first pass time
#     # plot_distrib_fpt_x_max(bins_fpt_x_max, counts_fpt_xmax_normalized, title_0, title_size, legend_size, saving)

#     # IV : Plot the cumulative probability of passage (tau)
#     plot_proba_pass(tau_fpt_x_max, title_0, title_size, legend_size, saving)

#     # V : p_pass
#     plot_delay(p_pass, title_0, saving)

#     # VI : t_pass
#     plot_t_pass(t_pass, bins_t_pass, title_0, saving)

#     return None
# #- -#


import os
import numpy as np
import matplotlib.pyplot as plt

def plot_jump_probabilities(obstacle_profile, jump_probabilities, title, save_dir,
                            title_size=12, x_label_size=10, y_label_size=10, legend_size=10, save=True):
    """
    Plot the jump probabilities along with the obstacle portion.

    Parameters:
    - obstacle_profile: list of arrays, typically containing one array with obstacle data
    - jump_probabilities: array of jump probabilities
    - title: base string used in plot title and filename
    - save_dir: path to directory where the image should be saved
    - title_size, x_label_size, y_label_size, legend_size: font sizes
    - save: if True, saves the plot as PNG
    """
    if not save:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f'Probabilities — {title}', fontsize=title_size)

    x_obstacles = np.arange(len(obstacle_profile[0]))
    x_prob = np.arange(len(jump_probabilities))

    ax.step(x_obstacles, obstacle_profile[0], color='blue', linewidth=0.5, label='Obstacle fraction')
    ax.plot(x_prob, jump_probabilities, color='red', linewidth=2, label='Jump probability')

    ax.set_xlabel('Position (bp)', fontsize=x_label_size)
    ax.set_ylabel('Probability', fontsize=y_label_size)
    ax.set_xlim(0, 1000)
    ax.grid(True)
    ax.legend(fontsize=legend_size, loc='upper right')

    save_path = os.path.join(save_dir, f'probabilities__{title}.png')
    plt.savefig(save_path)
    plt.close(fig)


def plot_obstacle_distribution(x_values, y_values, title, save_dir,
                               title_size=12, x_label_size=10, y_label_size=10, legend_size=10, save=True):
    """
    Plot the distribution of obstacles.

    Parameters:
    - x_values, y_values: data points to plot
    - title: base string used in plot title and filename
    - save_dir: path to directory where the image should be saved
    - title_size, x_label_size, y_label_size, legend_size: font sizes
    - save: if True, saves the plot as PNG
    """
    if not save:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f'Obstacle dispersion — {title}', fontsize=title_size)

    ax.plot(x_values, y_values, color='blue', label='Line plot')
    ax.scatter(x_values, y_values, color='red', alpha=0.3, marker='D', label='Data points')

    ax.set_xlabel('Nucleosome size (bp)', fontsize=x_label_size)
    ax.set_ylabel('Count', fontsize=y_label_size)
    ax.grid(True)
    ax.legend(fontsize=legend_size, loc='upper right')

    save_path = os.path.join(save_dir, f'obstacle_dispersion__{title}.png')
    plt.savefig(save_path)
    plt.close(fig)
