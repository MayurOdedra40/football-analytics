"""
draw_pitch(ax): a from-scratch 105x68m pitch outline in centre-origin
coordinates (matches events.csv's x_centred/y_centred and positions'
native x/y -- see src/parse.py).

No external pitch-drawing library (mplsoccer etc.) on purpose -- every line
below is a real FIFA-standard pitch dimension, and being able to point at
any one of them and say what it is (not "the library drew it") is the
point, per the project's brief. Shared by notebooks/01_eda.ipynb and
notebooks/02_pass_visualisation.ipynb so the pitch is defined once.
"""

import matplotlib.patches as patches

# Standard pitch markings, in metres. Length/width are configurable (they
# come from matches.csv's pitch_x/pitch_y) but the markings below use the
# fixed IFAB dimensions, which don't scale with pitch size in real pitches
# either -- a 105x68m and a 100x64m pitch both have an 18-yard box.
PENALTY_AREA_LENGTH = 16.5
PENALTY_AREA_WIDTH = 40.32
SIX_YARD_LENGTH = 5.5
SIX_YARD_WIDTH = 18.32
PENALTY_SPOT_DIST = 11.0
CENTRE_CIRCLE_RADIUS = 9.15
GOAL_WIDTH = 7.32
GOAL_DEPTH = 2.0


def draw_pitch(ax, pitch_length=105.0, pitch_width=68.0, line_color="black", face_color="none", linewidth=1.2):
    """Draws pitch markings on `ax` in centre-origin coordinates:
    x in [-pitch_length/2, +pitch_length/2], y in [-pitch_width/2, +pitch_width/2].
    Returns ax for chaining."""
    half_l, half_w = pitch_length / 2, pitch_width / 2

    # Outer boundary
    ax.add_patch(patches.Rectangle((-half_l, -half_w), pitch_length, pitch_width,
                                    fill=(face_color != "none"), facecolor=face_color,
                                    edgecolor=line_color, linewidth=linewidth, zorder=1))

    # Halfway line
    ax.plot([0, 0], [-half_w, half_w], color=line_color, linewidth=linewidth, zorder=1)

    # Centre circle + spot
    ax.add_patch(patches.Circle((0, 0), CENTRE_CIRCLE_RADIUS, fill=False,
                                 edgecolor=line_color, linewidth=linewidth, zorder=1))
    ax.plot(0, 0, marker="o", markersize=2, color=line_color, zorder=1)

    # Left (x=-half_l) and right (x=+half_l) ends: penalty area, six-yard
    # box, penalty spot, goal -- mirrored.
    for side in (-1, 1):
        goal_x = side * half_l

        # Penalty area
        pen_x = goal_x - side * PENALTY_AREA_LENGTH
        ax.add_patch(patches.Rectangle(
            (min(goal_x, pen_x), -PENALTY_AREA_WIDTH / 2),
            PENALTY_AREA_LENGTH, PENALTY_AREA_WIDTH,
            fill=False, edgecolor=line_color, linewidth=linewidth, zorder=1))

        # Six-yard box
        six_x = goal_x - side * SIX_YARD_LENGTH
        ax.add_patch(patches.Rectangle(
            (min(goal_x, six_x), -SIX_YARD_WIDTH / 2),
            SIX_YARD_LENGTH, SIX_YARD_WIDTH,
            fill=False, edgecolor=line_color, linewidth=linewidth, zorder=1))

        # Penalty spot
        spot_x = goal_x - side * PENALTY_SPOT_DIST
        ax.plot(spot_x, 0, marker="o", markersize=2, color=line_color, zorder=1)

        # Penalty arc (the part of the circle around the spot outside the box)
        angle = side * -90  # orient the arc to open away from the goal
        theta1, theta2 = (308, 52) if side == -1 else (128, 232)
        ax.add_patch(patches.Arc((spot_x, 0), CENTRE_CIRCLE_RADIUS * 2, CENTRE_CIRCLE_RADIUS * 2,
                                  theta1=theta1, theta2=theta2, edgecolor=line_color, linewidth=linewidth, zorder=1))

        # Goal (small rectangle projecting outside the pitch)
        goal_outer_x = goal_x + side * GOAL_DEPTH
        ax.add_patch(patches.Rectangle(
            (min(goal_x, goal_outer_x), -GOAL_WIDTH / 2),
            GOAL_DEPTH, GOAL_WIDTH,
            fill=False, edgecolor=line_color, linewidth=linewidth, zorder=1))

    ax.set_xlim(-half_l - GOAL_DEPTH - 2, half_l + GOAL_DEPTH + 2)
    ax.set_ylim(-half_w - 2, half_w + 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax
