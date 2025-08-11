# ------------------- Imports ------------------- #
import sys
import math
import random
import pygame
import imageio
import numpy as np

# =================== CONFIG GLOBALE =================== #

# --- Affichage & grille --- #
VISIBLE_CASES   = 10
TOTAL_CASES     = VISIBLE_CASES * 5
CASE_SIZE       = 40
WIDTH           = VISIBLE_CASES * CASE_SIZE
HEIGHT          = 300
FPS             = 60

# --- Kymographe --- #
KYM_WIDTH       = WIDTH
KYM_HEIGHT      = HEIGHT
KYM_ZOOM        = 5
KYM_WINDOW      = 300

# ------------------- Couleurs ------------------- #
WHITE     = (255, 255, 255)
BLACK     = (0,   0,   0)
GRAY      = (238, 244, 251)   # accessible (bande)
DARK_GRAY = (42,  58, 58)     # bouchon
RED       = (200, 30,  30)
CAP_COLOR = (42, 58, 58)

# --- Enregistrement (GIF) --- #
RECORD_GIF          = True
GIF_PATH            = "animation.gif"      # animation principale
KYM_GIF_PATH        = "kymographe.gif"     # kymographe séparé
GIF_FPS             = 30

# ---- Obstacles (optionnels) ---- #
# 0 = obstacle, 1 = accessible. Si None, tout est accessible.
# Exemple: OBSTACLE_ARRAY = [1, 1, 0, 1, 1]
OBSTACLE_ARRAY  = None
OBSTACLE_ARRAY = [1, 0, 0] * 20
OBSTACLE_ARRAY = np.random.randint(low=0, high=2, size=TOTAL_CASES)


# ---- Géométrie verticale (bande collée en bas) ---- #
BOTTOM_MARGIN   = 20                               # marge visuelle sous la bande
STRIP_BOTTOM_Y  = HEIGHT - BOTTOM_MARGIN           # y bas de la bande
STRIP_TOP_Y     = STRIP_BOTTOM_Y - CASE_SIZE       # y haut de la bande (cases collées en bas)
BALL_RADIUS     = max(6, CASE_SIZE // 4)
BASELINE_Y      = STRIP_TOP_Y + CASE_SIZE // 2     # centre vertical de la bande

# ---- Arc du saut ---- #
ARC_BASE        = 0.20
ARC_PER_CELL    = 0.10
ARC_GAIN        = 7.0

# ---- Timing fixe ---- #
SECS_PER_JUMP   = 0.5   # durée d'un saut (constante)
WAIT_SECS       = 0.5   # attente entre sauts (constante)

# ---- Distance de saut ~ Gamma ---- #
# Distance (en cases) tirée depuis Gamma(shape=k, scale=mu/k), arrondie >= 1
JUMP_MU_CELLS   = 2.0   # μ en cases
JUMP_SHAPE      = 2.0   # k (shape)


# --- Kymographe (axes fixes) --- #
KYM_T_MAX        = 20.0  # secondes affichées (fixe)
KYM_Y_MIN        = 0
KYM_Y_MAX        = TOTAL_CASES - 1

# Mise en page / axes
KYM_MARGIN_LEFT   = 50
KYM_MARGIN_RIGHT  = 20
KYM_MARGIN_TOP    = 20
KYM_MARGIN_BOTTOM = 40
GRID_COLOR        = (220, 220, 220)
AXIS_COLOR        = (0, 0, 0)
TICKS_X           = 6
TICKS_Y           = 6
FONT_MAIN_SIZE    = 16
FONT_TICK_SIZE    = 12



# =================== OUTILS =================== #
def clamp(val, a, b):
    return max(a, min(b, val))

def normalize_obstacles_array(arr):
    if arr is None:
        return [1] * TOTAL_CASES
    tmp = [1 if int(v) != 0 else 0 for v in arr]
    if len(tmp) >= TOTAL_CASES:
        return tmp[:TOTAL_CASES]
    else:
        return tmp + [1] * (TOTAL_CASES - len(tmp))

def draw_cells_strip(screen, access, start_case):
    """Bande visible collée en bas + un bouchon coloré au-dessus des inaccessibles (la bande reste GRAY)."""
    for i in range(VISIBLE_CASES):
        idx = start_case + i

        base_rect = pygame.Rect(i * CASE_SIZE, STRIP_TOP_Y, CASE_SIZE, CASE_SIZE)
        pygame.draw.rect(screen, GRAY, base_rect)
        pygame.draw.rect(screen, BLACK, base_rect, 1)

        if 0 <= idx < len(access) and access[idx] == 0:
            cap_rect = pygame.Rect(i * CASE_SIZE, STRIP_TOP_Y - CASE_SIZE, CASE_SIZE, CASE_SIZE)
            if cap_rect.bottom > 0:
                pygame.draw.rect(screen, CAP_COLOR, cap_rect)
                pygame.draw.rect(screen, BLACK, cap_rect, 1)

def draw_ball(screen, x_abs_px, y_px, start_case):
    x_screen = int(x_abs_px - start_case * CASE_SIZE)
    pygame.draw.circle(screen, RED, (x_screen, int(y_px)), BALL_RADIUS)

def arc_y(center_y_px, progress_01, jump_len_cells):
    amp_cells = (ARC_BASE + ARC_PER_CELL * max(1, jump_len_cells)) * ARC_GAIN
    amp_px = amp_cells * CASE_SIZE
    return center_y_px - amp_px * (4 * progress_01 * (1 - progress_01))

def sample_jump_length_cells(rng):
    theta = JUMP_MU_CELLS / JUMP_SHAPE  # scale
    val = rng.gamma(shape=JUMP_SHAPE, scale=theta)
    return max(1, int(np.rint(val)))   # arrondi au plus proche, min 1

def render_kymo_surface(trajectory, times):
    """Kymographe à axes fixes: X=[0, KYM_T_MAX], Y=[KYM_Y_MIN, KYM_Y_MAX]."""
    surf = pygame.Surface((KYM_WIDTH, KYM_HEIGHT))
    surf.fill(WHITE)

    # Zone tracé
    plot_left   = KYM_MARGIN_LEFT
    plot_right  = KYM_WIDTH - KYM_MARGIN_RIGHT
    plot_top    = KYM_MARGIN_TOP
    plot_bottom = KYM_HEIGHT - KYM_MARGIN_BOTTOM
    plot_w = max(1, plot_right - plot_left)
    plot_h = max(1, plot_bottom - plot_top)

    # Polices
    pygame.font.init()
    font_main = pygame.font.SysFont(None, FONT_MAIN_SIZE)
    font_tick = pygame.font.SysFont(None, FONT_TICK_SIZE)

    # Fonctions de mapping fixes
    def x_to_px(t):
        t_clamped = max(0.0, min(KYM_T_MAX, t))
        return plot_left + (t_clamped / KYM_T_MAX) * plot_w

    def y_to_px(y):
        y_clamped = max(KYM_Y_MIN, min(KYM_Y_MAX, y))
        return plot_top + (KYM_Y_MAX - y_clamped) / (KYM_Y_MAX - KYM_Y_MIN + 1e-9) * plot_h

    # Grille + ticks X (0..KYM_T_MAX)
    for i in range(TICKS_X + 1):
        frac = i / TICKS_X
        t_val = frac * KYM_T_MAX
        x = int(x_to_px(t_val))
        pygame.draw.line(surf, GRID_COLOR, (x, plot_top), (x, plot_bottom), 1)
        pygame.draw.line(surf, AXIS_COLOR, (x, plot_bottom), (x, plot_bottom + 4), 1)
        lbl = font_tick.render(f"{t_val:.1f}", True, AXIS_COLOR)
        surf.blit(lbl, (x - lbl.get_width() // 2, plot_bottom + 6))

    # Grille + ticks Y (KYM_Y_MIN..KYM_Y_MAX)
    for j in range(TICKS_Y + 1):
        frac = j / TICKS_Y
        y_val = KYM_Y_MIN + frac * (KYM_Y_MAX - KYM_Y_MIN)
        y_px = int(y_to_px(y_val))
        pygame.draw.line(surf, GRID_COLOR, (plot_left, y_px), (plot_right, y_px), 1)
        pygame.draw.line(surf, AXIS_COLOR, (plot_left - 4, y_px), (plot_left, y_px), 1)
        lbl = font_tick.render(f"{y_val:.0f}", True, AXIS_COLOR)
        surf.blit(lbl, (plot_left - 8 - lbl.get_width(), y_px - lbl.get_height() // 2))

    # Courbe (trace progressive sans glissement)
    if len(trajectory) >= 2 and len(times) >= 2:
        pts = []
        for t, y in zip(times, trajectory):
            if t > KYM_T_MAX:
                break  # au-delà, on n'affiche plus (axes fixes)
            pts.append((x_to_px(t), y_to_px(y)))
        if len(pts) > 1:
            pygame.draw.lines(surf, RED, False, pts, 2)

    # Cadre + labels
    pygame.draw.rect(surf, AXIS_COLOR, pygame.Rect(plot_left, plot_top, plot_w, plot_h), 1)
    xlabel = font_main.render("Time", True, AXIS_COLOR)
    surf.blit(xlabel, (plot_left + (plot_w - xlabel.get_width()) // 2, KYM_HEIGHT - KYM_MARGIN_BOTTOM + 8))
    ylabel = font_main.render("Position (bp units)", True, AXIS_COLOR)
    surf.blit(pygame.transform.rotate(ylabel, 90), (8, plot_top + (plot_h - ylabel.get_height()) // 2))
    title = font_main.render("Kymograph", True, AXIS_COLOR)
    surf.blit(title, (plot_left + (plot_w - title.get_width()) // 2, 2))

    return surf



# =================== SIMULATION =================== #
def run_simulation():

    cum_t = 0.0
    traj_times = []

    rng = np.random.default_rng(123)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Sauts 1D (obstacles optionnels, distance ~ Gamma)")
    clock = pygame.time.Clock()

    access = normalize_obstacles_array(OBSTACLE_ARRAY)

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
    gif_frames_kymo = []

    def choose_next_target(cur_pos):
        jump_len = sample_jump_length_cells(rng)
        tgt = min(cur_pos + jump_len, TOTAL_CASES - 1)
        # évite d'atterrir sur un obstacle -> cherche la prochaine accessible à droite
        idx = tgt
        while idx < TOTAL_CASES and access[idx] == 0:
            idx += 1
        if idx >= TOTAL_CASES:
            return cur_pos
        return idx

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
                if pos < TOTAL_CASES - 1:
                    nxt = choose_next_target(pos)
                    if nxt == pos:
                        running = False
                    else:
                        jumping = True
                        jump_target = nxt
                        jump_progress = 0.0
                else:
                    running = False

        # --- Rendu principal (bande collée en bas) ---
        screen.fill(WHITE)
        start_case = clamp(pos - VISIBLE_CASES // 2, 0, TOTAL_CASES - VISIBLE_CASES)
        draw_cells_strip(screen, access, start_case)

        if jumping and jump_target is not None:
            eased = 0.5 - 0.5 * math.cos(math.pi * jump_progress)
            x0 = pos * CASE_SIZE + CASE_SIZE // 2
            x1 = jump_target * CASE_SIZE + CASE_SIZE // 2
            x_abs = (1 - eased) * x0 + eased * x1
            y = arc_y(center_y_px=BASELINE_Y, progress_01=jump_progress,
                      jump_len_cells=abs(jump_target - pos))
            draw_ball(screen, x_abs, y, start_case)
        else:
            x_abs = pos * CASE_SIZE + CASE_SIZE // 2
            draw_ball(screen, x_abs, BASELINE_Y, start_case)

        pygame.display.flip()

        # --- Mémorise position pour le kymo ---
        trajectory.append(pos)
        traj_times.append(cum_t)


        # --- Capture des GIFs ---
        if RECORD_GIF:
            # animation principale
            frame = pygame.surfarray.array3d(screen)
            frame = np.transpose(frame, (1, 0, 2))
            gif_frames_main.append(frame)

            # kymographe (surface séparée)
            kymo_surface = render_kymo_surface(trajectory, traj_times)
            frame_k = pygame.surfarray.array3d(kymo_surface)
            frame_k = np.transpose(frame_k, (1, 0, 2))
            gif_frames_kymo.append(frame_k)

    pygame.quit()

    if RECORD_GIF:
        if gif_frames_main:
            imageio.mimsave(GIF_PATH, gif_frames_main, fps=GIF_FPS)
            print(f"GIF animation enregistré sous '{GIF_PATH}'")
        if gif_frames_kymo:
            imageio.mimsave(KYM_GIF_PATH, gif_frames_kymo, fps=GIF_FPS)
            print(f"GIF kymographe enregistré sous '{KYM_GIF_PATH}'")

# =================== MAIN =================== #
if __name__ == "__main__":
    # Exemple obstacles :
    # OBSTACLE_ARRAY = [1, 1, 0, 1, 1, 0, 1]
    run_simulation()
