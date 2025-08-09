import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, patches
from matplotlib.animation import PillowWriter

# ------------------- Paramètres ------------------- #
N_TOTAL_CELLS = 300     # longueur totale de la chaîne
N_VIEW = 30             # nombre de cases visibles à l'écran (fenêtre)
FPS = 30

# Sauts
JUMP_MIN = 1            # taille minimale du saut (en nb de cases)
JUMP_MAX = 3            # taille maximale du saut (en nb de cases)
SECS_PER_JUMP = 0.6     # durée d'un saut (fixe), augmenter pour ralentir le déplacement

# Attente (dwell) entre sauts
WAIT_MIN = 0.2          # temps d'attente min (sec)
WAIT_MAX = 0.8          # temps d'attente max (sec)

BALL_RADIUS = 0.28
ARC_BASE = 0.20         # hauteur de base de l'arc
ARC_PER_CELL = 0.10     # bonus de hauteur par case sautée

# Accessibilité : 1 = accessible (gris clair), 0 = inaccessible (gris foncé)
# Exemple : quelques obstacles épars
accessibles = np.ones(N_TOTAL_CELLS, dtype=int)
rng = np.random.default_rng(42)
blocked_idx = rng.choice(np.arange(5, N_TOTAL_CELLS-5), size=40, replace=False)
accessibles[blocked_idx] = 0

# Option : forcer la première case accessible
accessibles[0] = 1

# ------------------- Génération du chemin ------------------- #
def pick_next_target(cur):
    """Choisit une destination en respectant JUMP_MIN..JUMP_MAX et l'accessibilité."""
    if cur >= N_TOTAL_CELLS - 1:
        return cur
    tries = 0
    while tries < 100:
        L = rng.integers(JUMP_MIN, JUMP_MAX + 1)  # longueur du saut
        tgt = min(N_TOTAL_CELLS - 1, cur + L)
        # si la case cible est inaccessible, on retente un autre L
        if accessibles[tgt] == 1:
            return tgt
        tries += 1
    # si vraiment pas trouvé (rare), on cherche la prochaine accessible vers la droite
    nxt = cur + 1
    while nxt < N_TOTAL_CELLS and accessibles[nxt] == 0:
        nxt += 1
    return min(nxt, N_TOTAL_CELLS - 1)

# Construit les frames (positions balle + offset de défilement)
frames_x, frames_y, frames_s = [], [], []   # positions relatives (x,y) et scroll s

def add_jump_segment(a, b):
    """Ajoute les frames d'un saut a->b (coord absolues), puis l'attente aléatoire."""
    length = abs(b - a)
    arc_amp = ARC_BASE + ARC_PER_CELL * length
    frames_per_jump = max(2, int(SECS_PER_JUMP * FPS))
    wait_secs = float(rng.uniform(WAIT_MIN, WAIT_MAX))
    wait_frames = int(wait_secs * FPS)

    # interpolation horizontale et arc vertical
    for t in np.linspace(0, 1, frames_per_jump, endpoint=False):
        x_abs = (1 - t) * (a + 0.5) + t * (b + 0.5)
        y = 0.5 + arc_amp * (4 * t * (1 - t))
        s = compute_scroll(x_abs)  # scroll pour “caméra”
        frames_x.append(x_abs - s)
        frames_y.append(y)
        frames_s.append(s)

    # attente à l'arrivée
    x_abs = b + 0.5
    for _ in range(wait_frames):
        s = compute_scroll(x_abs)
        frames_x.append(x_abs - s)
        frames_y.append(0.5)
        frames_s.append(s)

def compute_scroll(x_abs):
    """
    Centre la 'caméra' autour de la bille autant que possible.
    x_abs est la position absolue (centre) de la bille.
    s = index flottant du bord gauche de la fenêtre.
    """
    desired_left = x_abs - N_VIEW / 2
    s = max(-0.5, min(N_TOTAL_CELLS - 0.5 - N_VIEW, desired_left))
    # bornages pour rester dans [ -0.5 .. N_TOTAL_CELLS - 0.5 - N_VIEW ]
    return s

# Simule jusqu'à la fin (ou un nombre max de sauts pour éviter les boucles)
cur = 0
max_jumps = 2000
jump_count = 0
while cur < N_TOTAL_CELLS - 1 and jump_count < max_jumps:
    nxt = pick_next_target(cur)
    if nxt == cur:
        break
    add_jump_segment(cur, nxt)
    cur = nxt
    jump_count += 1

# ------------------- Figure & artistes ------------------- #
fig, ax = plt.subplots(figsize=(8, 2.6))
ax.set_aspect('equal')
ax.set_ylim(0, 1.6)
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

# Précrée un pool de rectangles réutilisés (performance)
# On garde N_VIEW+2 pour couvrir les bords
rects = []
for _ in range(N_VIEW + 2):
    r = patches.Rectangle((0, 0), 1, 1, edgecolor='black', facecolor=(0.85, 0.85, 0.85), lw=1.2)
    ax.add_patch(r)
    rects.append(r)

# Bille
ball = plt.Circle((0.5, 0.5), BALL_RADIUS, color='red')
ax.add_patch(ball)

def draw_window(s):
    """
    Met à jour la position/couleur des rectangles visibles en fonction du scroll s.
    Chaque rectangle i représente la case d'index idx = floor(s) + i - buffer.
    """
    left_idx = int(np.floor(s + 0.5))  # approx aligne centres, reste fluide
    x_offset = left_idx - s  # décalage fractionnaire

    for i, r in enumerate(rects):
        idx = left_idx + i
        x = (i + x_offset)          # position relative dans la fenêtre
        r.set_xy((x, 0))
        # couleur selon accessibilité
        if 0 <= idx < N_TOTAL_CELLS:
            if accessibles[idx] == 1:
                r.set_facecolor((0.85, 0.85, 0.85))   # accessible
            else:
                r.set_facecolor((0.55, 0.55, 0.55))   # inaccessible
            r.set_visible(True)
        else:
            r.set_visible(False)

    ax.set_xlim(0, N_VIEW)

def init():
    # première frame
    draw_window(frames_s[0])
    ball.center = (frames_x[0], frames_y[0])
    return (ball, *rects)

def animate(i):
    draw_window(frames_s[i])
    ball.center = (frames_x[i], frames_y[i])
    return (ball, *rects)

ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=len(frames_x), interval=1000/FPS, blit=True)

ani.save("anim.gif", writer=PillowWriter(fps=FPS))
plt.close(fig)
print("GIF écrit : bille_rouge_longue_chaine.gif")
