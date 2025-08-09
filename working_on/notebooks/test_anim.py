# ------------------- Imports ------------------- #
import sys
import math
import random
import pygame
import imageio
import numpy as np


# ------------------- Configuration ------------------- #
# --- Grille / affichage --- #
VISIBLE_CASES   = 30          # nb de cases visibles (fenêtre)
TOTAL_CASES     = 30 * 5      # longueur totale de la chaîne
OBSTACLE_COUNT  = 40          # nb de cases inaccessibles
CASE_SIZE       = 40          # taille d'une case (px)

WIDTH           = VISIBLE_CASES * CASE_SIZE
HEIGHT          = 500

# --- Kymographe (en bas) --- #
KYM_WIDTH       = WIDTH
KYM_HEIGHT      = 200
KYM_Y_OFFSET    = HEIGHT - KYM_HEIGHT
KYM_ZOOM        = 5           # échelle verticale (px par case)
KYM_WINDOW      = 300         # nb de points récents à tracer

# --- Animation / logique --- #
FPS             = 60
JUMP_MIN        = 1           # taille de saut minimale (en cases)
JUMP_MAX        = 3           # taille de saut maximale (en cases)
SECS_PER_JUMP   = 0.60        # durée d'un saut (sec)
WAIT_MIN        = 0.20        # attente minimale entre sauts (sec)
WAIT_MAX        = 0.80        # attente maximale entre sauts (sec)
ARC_BASE        = 0.20        # hauteur de base de l'arc du saut
ARC_PER_CELL    = 0.10        # bonus de hauteur par case sautée

# --- Enregistrement GIF --- #
RECORD_GIF      = True
GIF_PATH        = "animation.gif"
GIF_FPS         = 30          # fps du GIF de sortie


# ------------------- Couleurs ------------------- #
WHITE     = (255, 255, 255)
BLACK     = (0,   0,   0)
GRAY      = (180, 180, 180)   # accessible
DARK_BLUE = (0,   0, 139)     # inaccessible
RED       = (200, 30,  30)


# ------------------- Outils ------------------- #
def clamp(val, a, b):
    return max(a, min(b, val))


# ------------------- Initialisation ------------------- #
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chaîne 1D : sauts avec obstacles + kymographe")
clock = pygame.time.Clock()

# Grille d'accessibilité
access = [1] * TOTAL_CASES
access[0] = 1
access[-1] = 1
rng = random.Random(42)  # seed pour reproductibilité
obstacle_pool = list(range(1, TOTAL_CASES - 1))
rng.shuffle(obstacle_pool)
for idx in obstacle_pool[:min(OBSTACLE_COUNT, len(obstacle_pool))]:
    access[idx] = 0


# ------------------- État de la simulation ------------------- #
pos = 0                    # position (index de case)
jumping = False
jump_target = None
jump_progress = 0.0        # [0..1] pendant un saut
frames_per_jump = max(2, int(SECS_PER_JUMP * FPS))
jump_step = 1.0 / frames_per_jump

wait_timer = 0.0           # secondes restantes avant prochain saut

trajectory = []            # pour le kymographe (positions dans le temps)
gif_frames = []            # frames capturées si RECORD_GIF


# ------------------- Choix du prochain saut ------------------- #
def choose_next_target(cur_pos):
    """
    Choisit aléatoirement une case cible accessible dans [JUMP_MIN, JUMP_MAX].
    Si aucune dispo dans cette fenêtre, tente de trouver la prochaine accessible plus loin.
    """
    # Candidats dans la fenêtre autorisée
    candidates = []
    for step in range(JUMP_MIN, JUMP_MAX + 1):
        idx = cur_pos + step
        if idx < TOTAL_CASES and access[idx] == 1:
            candidates.append(idx)

    if candidates:
        return rng.choice(candidates)

    # Fallback : cherche la prochaine accessible plus loin (évite blocage)
    idx = cur_pos + JUMP_MIN
    while idx < TOTAL_CASES and access[idx] == 0:
        idx += 1
    return idx if idx < TOTAL_CASES else cur_pos


# ------------------- Dessin ------------------- #
def draw_scene():
    screen.fill(WHITE)

    # 1) Bande des cases en haut (centrée verticalement autour de y=100)
    start_case = clamp(pos - VISIBLE_CASES // 2, 0, TOTAL_CASES - VISIBLE_CASES)
    # Dessine la fenêtre visible
    for i in range(VISIBLE_CASES):
        idx = start_case + i
        rect = pygame.Rect(i * CASE_SIZE, 90, CASE_SIZE, 20)
        color = GRAY if access[idx] else DARK_BLUE
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, BLACK, rect, 1)

    # 2) Bille (avec interpolation si en saut)
    if jumping and jump_target is not None:
        # easing cos pour un mouvement plus doux
        eased = 0.5 - 0.5 * math.cos(math.pi * jump_progress)
        # interpolation horizontale en pixels absolus (centres de cases)
        x0 = pos * CASE_SIZE + CASE_SIZE // 2
        x1 = jump_target * CASE_SIZE + CASE_SIZE // 2
        x_abs = (1 - eased) * x0 + eased * x1

        # arc vertical (en pixels, relatif à une "ligne" à y=100)
        length = max(1, abs(jump_target - pos))
        arc_amp = (ARC_BASE + ARC_PER_CELL * length) * CASE_SIZE  # échelle en px
        y = 100 - arc_amp * (4 * jump_progress * (1 - jump_progress))

        # conversion coord. absolues -> fenêtre visible
        x_screen = int(x_abs - start_case * CASE_SIZE)
        pygame.draw.circle(screen, RED, (x_screen, int(y)), 10)
    else:
        # bille posée dans sa case
        x_screen = (pos - start_case) * CASE_SIZE + CASE_SIZE // 2
        pygame.draw.circle(screen, RED, (x_screen, 100), 10)

    # 3) Kymographe (trace pos(t) sur la fenêtre glissante)
    if len(trajectory) > 1:
        start_idx = max(0, len(trajectory) - KYM_WINDOW)
        points = []
        scale_x = KYM_WIDTH / KYM_WINDOW
        for i in range(start_idx, len(trajectory)):
            x = (i - start_idx) * scale_x
            y = KYM_Y_OFFSET + KYM_HEIGHT - trajectory[i] * KYM_ZOOM
            points.append((x, y))
        if len(points) > 1:
            pygame.draw.lines(screen, RED, False, points, 2)

    pygame.display.flip()


# ------------------- Boucle principale ------------------- #
def main():
    global pos, jumping, jump_target, jump_progress, wait_timer

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # secondes écoulées
        # ---- Gestion des événements ---- #
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False

        # ---- Logique de saut / attente ---- #
        if jumping:
            # Avancement du saut
            jump_progress += jump_step
            if jump_progress >= 1.0:
                # Atterrissage
                pos = jump_target
                jumping = False
                jump_target = None
                jump_progress = 0.0
                wait_timer = rng.uniform(WAIT_MIN, WAIT_MAX)
        else:
            if wait_timer > 0.0:
                wait_timer -= dt
            else:
                # Décider du prochain saut (si pas à la fin)
                if pos < TOTAL_CASES - 1:
                    nxt = choose_next_target(pos)
                    if nxt == pos:
                        # Aucune issue → fin
                        running = False
                    else:
                        jumping = True
                        jump_target = nxt
                        jump_progress = 0.0
                else:
                    # Arrivé au bout
                    running = False

        # ---- Mise à jour kymographe ---- #
        trajectory.append(pos)

        # ---- Rendu ---- #
        draw_scene()

        # ---- Capture GIF ---- #
        if RECORD_GIF:
            # Pygame: (w,h,3) avec axes (x,y). imageio attend (y,x,3).
            frame = pygame.surfarray.array3d(screen)
            frame = np.transpose(frame, (1, 0, 2))
            gif_frames.append(frame)

    # Sortie propre
    pygame.quit()

    if RECORD_GIF and len(gif_frames) > 0:
        print("Enregistrement du GIF...")
        imageio.mimsave(GIF_PATH, gif_frames, fps=GIF_FPS)
        print(f"GIF enregistré sous '{GIF_PATH}'")

    sys.exit(0)


# ------------------- Lancement ------------------- #
if __name__ == "__main__":
    main()