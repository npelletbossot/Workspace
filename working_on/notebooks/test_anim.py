# ------------------- Imports ------------------- #
import sys
import math
import random
import pygame
import imageio
import numpy as np

# Kymograph (matplotlib)
import matplotlib.pyplot as plt
from matplotlib import animation

# =================== CONFIG GLOBALE =================== #

# --- Résolution / Qualité --- #
RENDER_SCALE     = 2   # 1=normal, 2=2x, 3=3x...
S                 = RENDER_SCALE

# --- Affichage & grille (logique) --- #
VISIBLE_CASES    = 16
TOTAL_CASES      = 30 * 5
CASE_SIZE        = 40               # ↑ augmente cette valeur pour des cases plus grosses
WIDTH            = VISIBLE_CASES * CASE_SIZE
HEIGHT           = 200
FPS              = 30

# Dimensions haute résolution (pour l'enregistrement)
CASE_SIZE_S      = CASE_SIZE * S
WIDTH_S          = WIDTH * S
HEIGHT_S         = HEIGHT * S

# ------------------- Obstacles ------------------- #
SHOW_OBSTACLES   = True            # ⬅️ afficher visuellement les obstacles
AVOID_OBSTACLES  = True            # ⬅️ éviter d'atterrir sur une case inaccessible
OBSTACLE_ARRAY   = None            # ⬅️ mettez un tableau 0/1 pour forcer un paysage
OBSTACLE_RATE    = 0.2             # ⬅️ si OBSTACLE_ARRAY=None, ratio de cases bloquées (hors extrémités)
RNG_SEED         = 21              # reproductibilité du paysage aléatoire

# --- Kymographe (sortie Matplotlib) --- #
KYM_GIF_PATH     = "kymographe.gif"
KYM_T_MAX        = 20.0            # durée max affichée
KYM_Y_MIN        = 0
KYM_Y_MAX        = 100
KYM_FPS          = FPS             # synchro avec la simu
KYM_FIGSIZE      = (6, 3)          # inches
KYM_DPI          = 150
KYM_LABELPAD_X   = 12              # label↔axe (X)
KYM_LABELPAD_Y   = 14              # label↔axe (Y)
KYM_TITLE_PAD    = 8

# ------------------- Couleurs ------------------- #
WHITE     = (255, 255, 255)
BLACK     = (0,   0,   0)
GRAY      = (238, 244, 251)   # accessible (bande)
DARK_GRAY = (42,  58,  58)    # inaccessible (case + “cap” optionnel)
BLUE      = (149, 170, 211)
RED       = (200, 30,  30)

# --- Enregistrement (GIF principal) --- #
RECORD_GIF          = True
GIF_PATH            = "animation.gif"      # animation principale (Pygame)
GIF_FPS             = FPS                  # mêmes fps que la simu

# ---- Géométrie verticale (bande collée en bas) ---- #
BOTTOM_MARGIN   = 20
BOTTOM_MARGIN_S = BOTTOM_MARGIN * S
STRIP_BOTTOM_Y_S= HEIGHT_S - BOTTOM_MARGIN_S
STRIP_TOP_Y_S   = STRIP_BOTTOM_Y_S - CASE_SIZE_S
BALL_RADIUS_S   = max(6 * S, CASE_SIZE_S // 4)
BASELINE_Y_S    = STRIP_TOP_Y_S + CASE_SIZE_S // 2   # centre vertical de la bande

# ---- Arc du saut ---- #
ARC_FIXED_CELLS = 2.0   # hauteur de l’arc en "cases" (indépendante de la distance)

# ---- Timing ---- #
SECS_PER_JUMP   = 0.5   # durée d'un saut (constante)
WAIT_SECS       = 0.5   # attente entre sauts (constante)

# ---- Distance de saut ~ Gamma ---- #
# Distance (en cases) tirée depuis Gamma(shape=k, scale=mu/k), arrondie >= 1
JUMP_MU_CELLS   = 3.0   # μ en cases
JUMP_SHAPE      = 2.0   # k (shape)

# ---- Objets visuels des obstacles ---- #
DRAW_CAP_ABOVE  = True  # dessiner un “bouchon” au-dessus des cases inaccessibles (plus lisible)

# =================== OUTILS =================== #
def clamp(val, a, b):
    return max(a, min(b, val))

def make_obstacle_array():
    # """Construit un tableau 0/1 de longueur TOTAL_CASES (1=accessible, 0=inaccessible)."""
    # if OBSTACLE_ARRAY is not None:
    #     arr = [1 if int(v) != 0 else 0 for v in OBSTACLE_ARRAY]
    #     if len(arr) >= TOTAL_CASES:
    #         arr = arr[:TOTAL_CASES]
    #     else:
    #         arr = arr + [1] * (TOTAL_CASES - len(arr))
    # else:
    #     rng = random.Random(RNG_SEED)
    #     arr = [1] * TOTAL_CASES
    #     # bloque un sous-ensemble (hors extrémités)
    #     pool = list(range(1, TOTAL_CASES - 1))
    #     rng.shuffle(pool)
    #     nb_block = int(len(pool) * OBSTACLE_RATE)
    #     for idx in pool[:nb_block]:
    #         arr[idx] = 0
    # # extrémités accessibles
    # arr[0] = 1
    # arr[-1] = 1

    arr = [1] * 150
    # arr = [1, 1, 0, 0] * 20
    # arr = [1, 1, 0, 0, 1 ,1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0 ,1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1,
    #        1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]

    return arr

def draw_cells_strip_scaled(surf, access, start_case):
    """Bande visible collée en bas (hi-res) + obstacles visibles si SHOW_OBSTACLES."""
    for i in range(VISIBLE_CASES):
        idx = start_case + i
        if not (0 <= idx < len(access)):
            continue
        x0 = i * CASE_SIZE_S

        # base : la case elle-même
        color = GRAY if access[idx] == 1 else DARK_GRAY
        base_rect = pygame.Rect(x0, STRIP_TOP_Y_S, CASE_SIZE_S, CASE_SIZE_S)
        pygame.draw.rect(surf, color, base_rect)
        pygame.draw.rect(surf, BLACK, base_rect, max(1, S))

        # “cap” optionnel au-dessus (uniquement pour inaccessibles)
        if SHOW_OBSTACLES and DRAW_CAP_ABOVE and access[idx] == 0:
            cap_rect = pygame.Rect(x0, STRIP_TOP_Y_S - CASE_SIZE_S, CASE_SIZE_S, CASE_SIZE_S)
            if cap_rect.bottom > 0:
                pygame.draw.rect(surf, DARK_GRAY, cap_rect)
                pygame.draw.rect(surf, BLACK, cap_rect, max(1, S))

def draw_ball_scaled(surf, x_abs_px_s, y_px_s, start_case):
    x_screen = int(x_abs_px_s - start_case * CASE_SIZE_S)
    pygame.draw.circle(surf, RED, (x_screen, int(y_px_s)), int(BALL_RADIUS_S))

def arc_y(center_y_px_s, progress_01):
    """Arc à hauteur fixe (indépendante de la distance)."""
    amp_px_s = ARC_FIXED_CELLS * CASE_SIZE_S
    return center_y_px_s - amp_px_s * (4 * progress_01 * (1 - progress_01))

def sample_jump_length_cells(rng):
    """Échantillonne un saut (en cases) suivant Gamma(μ, k), arrondi et >= 1."""
    theta = JUMP_MU_CELLS / JUMP_SHAPE  # scale
    val = rng.gamma(shape=JUMP_SHAPE, scale=theta)
    return max(1, int(np.rint(val)))   # arrondi au plus proche, min 1

# =================== KYMOGRAPH MATPLOTLIB =================== #
def save_kymo_matplotlib(times, positions, out_path=KYM_GIF_PATH,
                         t_max=KYM_T_MAX, y_min=KYM_Y_MIN, y_max=KYM_Y_MAX, fps=KYM_FPS):
    """Crée un GIF du kymographe via Matplotlib (labels non gras + labelpad augmenté)."""
    fig, ax = plt.subplots(figsize=KYM_FIGSIZE, dpi=KYM_DPI)
    ax.set_xlim(0, t_max)
    ax.set_ylim(y_min, y_max)
    # Labels non gras avec labelpad augmenté
    ax.set_xlabel("Time", labelpad=KYM_LABELPAD_X, fontweight='normal')
    ax.set_ylabel("Position [int bp]", labelpad=KYM_LABELPAD_Y, fontweight='normal')
    ax.set_title("Kymograph", pad=KYM_TITLE_PAD, fontweight='normal')

    # ticks non gras
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('normal')

    # Ligne rouge
    line, = ax.plot([], [], lw=2, color="red")

    # Pré-troncature pour limiter à t_max
    t_clip, p_clip = [], []
    for t, p in zip(times, positions):
        if t > t_max:
            break
        t_clip.append(t)
        p_clip.append(p)

    def init():
        line.set_data([], [])
        return (line,)

    def update(frame):
        n = min(frame + 1, len(t_clip))
        line.set_data(t_clip[:n], p_clip[:n])
        return (line,)

    frames = max(1, min(len(t_clip), int(t_max * fps)))
    ani = animation.FuncAnimation(fig, update, init_func=init, frames=frames, interval=1000/fps, blit=True)
    ani.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)

# =================== SIMULATION =================== #
def run_simulation():
    rng = np.random.default_rng(123)

    pygame.init()
    # Fenêtre d'aperçu en taille logique (non upscalée)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Sauts 1D (Gamma) + obstacles visibles + kymo Matplotlib")
    clock = pygame.time.Clock()

    # Obstacles (visuels et logiques)
    access = make_obstacle_array() if SHOW_OBSTACLES else [1] * TOTAL_CASES

    # Surface hi-res pour l'enregistrement
    canvas = pygame.Surface((WIDTH_S, HEIGHT_S))       # animation

    # État
    pos = 0
    jumping = False
    jump_target = None
    jump_progress = 0.0
    frames_per_jump = max(2, int(SECS_PER_JUMP * FPS))
    jump_step = 1.0 / frames_per_jump
    wait_timer = 0.0

    trajectory = []
    gif_frames_main = []
    cum_t = 0.0
    traj_times = []

    def choose_next_target(cur_pos):
        """Choisit une cible en tirant une distance de saut ~ Gamma(μ, k)."""
        jump_len = sample_jump_length_cells(rng)
        tgt = min(cur_pos + jump_len, TOTAL_CASES - 1)
        if not AVOID_OBSTACLES or access[tgt] == 1:
            return tgt
        # si on évite : cherche la prochaine accessible vers la droite
        idx = tgt
        while idx < TOTAL_CASES and access[idx] == 0:
            idx += 1
        return min(idx, TOTAL_CASES - 1) if idx < TOTAL_CASES else cur_pos

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        cum_t += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if jumping:
            jump_progress += jump_step
            if jump_progress >= 1.0:
                pos = jump_target
                jumping = False
                jump_target = None
                jump_progress = 0.0
                wait_timer = WAIT_SECS
        else:
            if wait_timer > 0.0:
                wait_timer -= dt
            else:
                if pos < TOTAL_CASES - 1 and cum_t < KYM_T_MAX:
                    nxt = choose_next_target(pos)
                    if nxt == pos:
                        running = False
                    else:
                        jumping = True
                        jump_target = nxt
                        jump_progress = 0.0
                else:
                    running = False

        # --- Rendu hi-res sur canvas ---
        canvas.fill(WHITE)
        start_case = clamp(pos - VISIBLE_CASES // 2, 0, TOTAL_CASES - VISIBLE_CASES)
        draw_cells_strip_scaled(canvas, access, start_case)

        if jumping and jump_target is not None:
            eased = 0.5 - 0.5 * math.cos(math.pi * jump_progress)
            x0 = pos * CASE_SIZE_S + CASE_SIZE_S // 2
            x1 = jump_target * CASE_SIZE_S + CASE_SIZE_S // 2
            x_abs = (1 - eased) * x0 + eased * x1
            y = arc_y(center_y_px_s=BASELINE_Y_S, progress_01=jump_progress)
            draw_ball_scaled(canvas, x_abs, y, start_case)
        else:
            x_abs = pos * CASE_SIZE_S + CASE_SIZE_S // 2
            draw_ball_scaled(canvas, x_abs, BASELINE_Y_S, start_case)

        # --- Aperçu fenêtre (downscale lisse) ---
        preview = pygame.transform.smoothscale(canvas, (WIDTH, HEIGHT))
        screen.blit(preview, (0, 0))
        pygame.display.flip()

        # --- Mémoriser trajectoire & temps ---
        trajectory.append(pos)
        traj_times.append(cum_t)

        # --- Capture du GIF principal ---
        if RECORD_GIF:
            frame = pygame.surfarray.array3d(canvas)
            frame = np.transpose(frame, (1, 0, 2))
            gif_frames_main.append(frame)

    pygame.quit()

    # Sauvegardes
    if RECORD_GIF and gif_frames_main:
        imageio.mimsave(GIF_PATH, gif_frames_main, fps=GIF_FPS)
        print(f"GIF animation enregistré sous '{GIF_PATH}' ({WIDTH_S}x{HEIGHT_S})")

    # Kymographe Matplotlib (ligne rouge)
    if len(traj_times) >= 2:
        save_kymo_matplotlib(traj_times, trajectory, out_path=KYM_GIF_PATH, fps=KYM_FPS)
        print(f"GIF kymographe enregistré sous '{KYM_GIF_PATH}'")

# =================== MAIN =================== #
if __name__ == "__main__":
    run_simulation()
