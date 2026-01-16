import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# GLOBAL VISUALIZATION SETTINGS
# ==============================================================================

# 1. GRAPHICS FORMAT CONFIGURATION
# Saving as PDF is preferred for high-resolution vector graphics in reports.
SAVE_FORMAT = 'pdf'

# Matplotlib global configuration for consistent styling
plt.rcParams.update({
    'figure.figsize': (12, 8),      # Default figure size
    'figure.dpi': 300,              # High resolution for display
    'axes.titlesize': 18,           # Title font size
    'axes.labelsize': 14,           # Axis label font size
    'xtick.labelsize': 12,          # X-axis tick font size
    'ytick.labelsize': 12,          # Y-axis tick font size
    'legend.fontsize': 12,          # Legend font size
    'lines.markersize': 11,          # Marker size for better visibility
    'lines.linewidth': 2.5,
    'font.family': 'sans-serif',    # Clean font family
    'pdf.fonttype': 42              # Ensure fonts are embedded as TrueType (editable in vector soft)
})

# ==============================================================================
# COLOR PALETTE & ACCESSIBILITY
# ==============================================================================

# AVAILABLE PALETTES (Viridis family - Colorblind friendly & Perceptually Uniform)
# Change the 'SELECTED_PALETTE' variable below to switch the theme for all plots.
# Options:
#   'viridis'  : The default. Blue -> Green -> Yellow. High contrast.
#   'plasma'   : Blue -> Red -> Yellow. Higher contrast, very vibrant.
#   'inferno'  : Black -> Red -> Yellow. Good for dark backgrounds or high intensity.
#   'magma'    : Black -> Purple -> Peach. Similar to inferno but softer.
#   'cividis'  : Specifically designed for color vision deficiency (CVD). Blue -> Yellow.
#   'mako'     : Dark Blue -> Green. Ocean-like.
#   'rocket'   : Dark Purple -> Red -> White.
#   'turbo'    : Rainbow alternative (use with caution, but better than Jet).

SELECTED_PALETTE = 'rocket'  # <--- MODIFY THIS to test different variants

def getPalette():
    return self.SELECTED_PALETTE

# Generate a list of discrete colors from the continuous colormap.
# For binary classes (CTRL vs CHD), the first two or specific indices will be used.
_cmap =  plt.get_cmap(SELECTED_PALETTE)
contrast_indices = [0.0, 0.55, 0.95, 0.25, 0.75, 0.10, 0.65, 0.40, 0.85, 0.05, 0.50, 0.90, 0.20, 0.70, 0.30]
DISCRETE_COLORS = [_cmap(i) for i in contrast_indices]

#DISCRETE_COLORS = [_cmap(i) for i in np.linspace(0, 1, 10)]

# Set Seaborn default palette to match
sns.set_palette(DISCRETE_COLORS)

# ==============================================================================
# MARKERS FOR BLACK & WHITE COMPATIBILITY
# ==============================================================================
# To ensure accessibility when printed in grayscale, we map classes to specific markers.
MARKERS = [
    'o',  # Circle
    's',  # Square
    '^',  # Triangle Up
    'D',  # Diamond
    'v',  # Triangle Down
    'X',  # X (filled)
    'P',  # Plus (filled)
    '*',  # Star
    'p',  # Pentagon
    'h',  # Hexagon 1
    'H',  # Hexagon 2
    '<',  # Triangle Left
    '>',  # Triangle Right
    'd',  # Thin Diamond
    '8'   # Octagon
]
