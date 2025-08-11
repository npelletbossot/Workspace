# ------------------- Imports ------------------- #
import sys
import math
import random
import pygame
import imageio
import numpy as np

# =================== CONFIG GLOBALE =================== #

# --- Résolution / Qualité --- #
RENDER_SCALE     = 2   # 1=normal, 2=2x, 3=3x...
S                 = RENDER_SCALE

# --- Affichage & grille (logique) --- #
VISIBLE_CASES    = 10
TOTAL_CASES      = 30 * 5
CASE_SIZE        = 40
WIDTH            = VISIBLE_CASES * CASE_SIZE
HEIGHT           = 200
FPS              = 30

# Dimensions haute résolution (pour l'enregistrement)
CASE_SIZE_S      = CASE_SIZE * S
WIDTH_S          = WIDTH * S
HEIGHT_S         = HEIGHT * S

# --- Kymographe --- #
KYM_HEIGHT       = 200
KYM_HEIGHT_S     = KYM_HEIGHT * S
KYM_WIDTH_S      = WIDTH_S

# --- Kymographe (axes fixes) --- #
KYM_T_MAX        = 20.0     
KYM_Y_MIN        = 0
KYM_Y_MAX        = 100      

# Mise en page / axes (augmenter marge du bas pour éviter le chevauchement)
KYM_MARGIN_LEFT   = 50 * S
KYM_MARGIN_RIGHT  = 20 * S
KYM_MARGIN_TOP    = 20 * S
KYM_MARGIN_BOTTOM = 60 * S  

GRID_COLOR        = (220, 220, 220)
AXIS_COLOR        = (0, 0, 0)
TICKS_X           = 6
TICKS_Y           = 6
FONT_MAIN_SIZE    = 16 * S
FONT_TICK_SIZE    = 12 * S

LINE_W            = max(1, S)        # épaisseur axes/grille
KYM_TRACE_W       = 2 * S           


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
GIF_FPS             = FPS                  # ✅ même fps que la simu pour parfaite synchro

# ---- Obstacles (optionnels) ---- #
# 0 = obstacle, 1 = accessible. Si None, tout est accessible.
# OBSTACLE_ARRAY  = None
# OBSTACLE_ARRAY = [1, 0, 0] * 50
OBSTACLE_ARRAY = np.random.randint(low=0, high=2, size=TOTAL_CASES)

# ---- Géométrie verticale (bande collée en bas) ---- #
BOTTOM_MARGIN   = 20
BOTTOM_MARGIN_S = BOTTOM_MARGIN * S
STRIP_BOTTOM_Y_S= HEIGHT_S - BOTTOM_MARGIN_S
STRIP_TOP_Y_S   = STRIP_BOTTOM_Y_S - CASE_SIZE_S
BALL_RADIUS_S   = max(6 * S, CASE_SIZE_S // 4)
BASELINE_Y_S    = STRIP_TOP_Y_S + CASE_SIZE_S // 2   # centre vertical de la bande

# ---- Arc du saut ---- #
ARC_BASE        = 0.20
ARC_PER_CELL    = 0.0
ARC_GAIN        = 7.0
ARC_FIXED_CELLS = 2.0 

# ---- Timing fixe ---- #
SECS_PER_JUMP   = 0.5   # durée d'un saut (constante)
WAIT_SECS       = 0.5   # attente entre sauts (constante)

# ---- Distance de saut ~ Gamma ---- #
# Distance (en cases) tirée depuis Gamma(shape=k, scale=mu/k), arrondie >= 1
JUMP_MU_CELLS   = 2.0   # μ en cases
JUMP_SHAPE      = 2.0   # k (shape)

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

def draw_cells_strip_scaled(surf, access, start_case):
    """Bande visible collée en bas (hi-res) + un bouchon coloré au-dessus des inaccessibles."""
    for i in range(VISIBLE_CASES):
        idx = start_case + i
        x0 = i * CASE_SIZE_S

        base_rect = pygame.Rect(x0, STRIP_TOP_Y_S, CASE_SIZE_S, CASE_SIZE_S)
        pygame.draw.rect(surf, GRAY, base_rect)
        pygame.draw.rect(surf, BLACK, base_rect, LINE_W)

        if 0 <= idx < len(access) and access[idx] == 0:
            cap_rect = pygame.Rect(x0, STRIP_TOP_Y_S - CASE_SIZE_S, CASE_SIZE_S, CASE_SIZE_S)
            if cap_rect.bottom > 0:
                pygame.draw.rect(surf, CAP_COLOR, cap_rect)
                pygame.draw.rect(surf, BLACK, cap_rect, LINE_W)

def draw_ball_scaled(surf, x_abs_px_s, y_px_s, start_case):
    x_screen = int(x_abs_px_s - start_case * CASE_SIZE_S)
    pygame.draw.circle(surf, RED, (x_screen, int(y_px_s)), int(BALL_RADIUS_S))

def arc_y(center_y_px_s, progress_01, jump_len_cells):
    """Arc à hauteur fixe (indépendante de la distance)."""
    amp_px_s = ARC_FIXED_CELLS * CASE_SIZE_S
    return center_y_px_s - amp_px_s * (4 * progress_01 * (1 - progress_01))

def sample_jump_length_cells(rng):
    theta = JUMP_MU_CELLS / JUMP_SHAPE  # scale
    val = rng.gamma(shape=JUMP_SHAPE, scale=theta)
    return max(1, int(np.rint(val)))   # arrondi au plus proche, min 1

def render_kymo_surface_scaled(trajectory, times):
    """Kymographe hi-res à axes fixes: X=[0, KYM_T_MAX], Y=[KYM_Y_MIN, KYM_Y_MAX]."""
    surf = pygame.Surface((KYM_WIDTH_S, KYM_HEIGHT_S))
    surf.fill(WHITE)

    # Zone tracé
    plot_left   = KYM_MARGIN_LEFT
    plot_right  = KYM_WIDTH_S - KYM_MARGIN_RIGHT
    plot_top    = KYM_MARGIN_TOP
    plot_bottom = KYM_HEIGHT_S - KYM_MARGIN_BOTTOM
    plot_w = max(1, plot_right - plot_left)
    plot_h = max(1, plot_bottom - plot_top)

    pygame.font.init()
    font_main = pygame.font.SysFont(None, int(FONT_MAIN_SIZE))
    font_tick = pygame.font.SysFont(None, int(FONT_TICK_SIZE))

    def x_to_px(t):
        t_clamped = max(0.0, min(KYM_T_MAX, t))
        return plot_left + (t_clamped / KYM_T_MAX) * plot_w

    def y_to_px(y):
        y_clamped = max(KYM_Y_MIN, min(KYM_Y_MAX, y))
        return plot_top + (KYM_Y_MAX - y_clamped) / (KYM_Y_MAX - KYM_Y_MIN + 1e-9) * plot_h

    # Grille + ticks X
    for i in range(TICKS_X + 1):
        frac = i / TICKS_X
        t_val = frac * KYM_T_MAX
        x = int(x_to_px(t_val))
        pygame.draw.line(surf, GRID_COLOR, (x, plot_top), (x, plot_bottom), LINE_W)
        pygame.draw.line(surf, AXIS_COLOR, (x, plot_bottom), (x, plot_bottom + 4*S), LINE_W)
        lbl = font_tick.render(f"{t_val:.1f}", True, AXIS_COLOR)
        surf.blit(lbl, (x - lbl.get_width() // 2, plot_bottom + 6*S))

    # Grille + ticks Y
    for j in range(TICKS_Y + 1):
        frac = j / TICKS_Y
        y_val = KYM_Y_MIN + frac * (KYM_Y_MAX - KYM_Y_MIN)
        y_px = int(y_to_px(y_val))
        pygame.draw.line(surf, GRID_COLOR, (plot_left, y_px), (plot_right, y_px), LINE_W)
        pygame.draw.line(surf, AXIS_COLOR, (plot_left - 4*S, y_px), (plot_left, y_px), LINE_W)
        lbl = font_tick.render(f"{y_val:.0f}", True, AXIS_COLOR)
        surf.blit(lbl, (plot_left - 8*S - lbl.get_width(), y_px - lbl.get_height() // 2))

    # Courbe
    if len(trajectory) >= 2 and len(times) >= 2:
        pts = []
        for t, yv in zip(times, trajectory):
            if t > KYM_T_MAX:
                break
            pts.append((x_to_px(t), y_to_px(yv)))
        if len(pts) > 1:
            pygame.draw.lines(surf, RED, False, pts, KYM_TRACE_W)

    # Cadre + labels
    pygame.draw.rect(surf, AXIS_COLOR, pygame.Rect(plot_left, plot_top, plot_w, plot_h), LINE_W)

    xlabel = font_main.render("Time", True, AXIS_COLOR)
    # ✅ place le label X bien sous le cadre, sans chevauchement
    xlabel_y = plot_bottom + 12 * S
    surf.blit(xlabel, (plot_left + (plot_w - xlabel.get_width()) // 2, xlabel_y))

    ylabel = font_main.render("Position [int bp]", True, AXIS_COLOR)
    ylabel_rot = pygame.transform.rotate(ylabel, 90)
    surf.blit(ylabel_rot, (8 * S, plot_top + (plot_h - ylabel_rot.get_height()) // 2))

    title = font_main.render("Kymograph", True, AXIS_COLOR)
    surf.blit(title, (plot_left + (plot_w - title.get_width()) // 2, 2 * S))


    return surf

# =================== SIMULATION =================== #
def run_simulation():
    rng = np.random.default_rng(123)

    pygame.init()
    # Fenêtre d'aperçu en taille logique (non upscalée)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Sauts 1D (hi-res GIF + kymo synchro)")
    clock = pygame.time.Clock()

    access = normalize_obstacles_array(OBSTACLE_ARRAY)

    # Surfaces hi-res pour l'enregistrement
    canvas = pygame.Surface((WIDTH_S, HEIGHT_S))       # animation
    # kymographe généré via render_kymo_surface_scaled

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
    cum_t = 0.0
    traj_times = []

    def choose_next_target(cur_pos):
        jump_len = sample_jump_length_cells(rng)
        tgt = min(cur_pos + jump_len, TOTAL_CASES - 1)
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
            y = arc_y(center_y_px_s=BASELINE_Y_S, progress_01=jump_progress,
                      jump_len_cells=abs(jump_target - pos))
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

        # --- Capture des GIFs (synchro: 1 frame par tick, même fps) ---
        if RECORD_GIF:
            # animation principale
            frame = pygame.surfarray.array3d(canvas)
            frame = np.transpose(frame, (1, 0, 2))
            gif_frames_main.append(frame)

            # kymographe (surface séparée hi-res)
            kymo_surface = render_kymo_surface_scaled(trajectory, traj_times)
            frame_k = pygame.surfarray.array3d(kymo_surface)
            frame_k = np.transpose(frame_k, (1, 0, 2))
            gif_frames_kymo.append(frame_k)

    pygame.quit()

    if RECORD_GIF:
        if gif_frames_main:
            imageio.mimsave(GIF_PATH, gif_frames_main, fps=GIF_FPS)
            print(f"GIF animation enregistré sous '{GIF_PATH}' ({WIDTH_S}x{HEIGHT_S})")
        if gif_frames_kymo:
            imageio.mimsave(KYM_GIF_PATH, gif_frames_kymo, fps=GIF_FPS)
            print(f"GIF kymographe enregistré sous '{KYM_GIF_PATH}' ({KYM_WIDTH_S}x{KYM_HEIGHT_S})")

# =================== MAIN =================== #
if __name__ == "__main__":
    # Exemple obstacles :
    # OBSTACLE_ARRAY = [1, 1, 0, 1, 1, 0, 1]
    run_simulation()
