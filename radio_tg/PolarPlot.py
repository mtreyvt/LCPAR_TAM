import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

def plot_patterns(deg: np.ndarray,
                  traces: list[tuple[str, np.ndarray]],
                  title: str = "Radiation Pattern: Anechoic vs Time-Gated",
                  save_path: Optional[str] = None) -> None:
    import matplotlib.pyplot as plt  # local import to keep base deps light
    import numpy as np

    plt.figure(figsize=(9, 5))
    for label, y in traces:
        plt.plot(deg, y, label=label, linewidth=1.5)
    plt.grid(True)
    plt.xlabel("Angle (deg)")
    plt.ylabel("Relative Level (dB)")
    plt.title(title)
    plt.legend(loc="best")
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot -> {save_path}")
    plt.show()

def plot_polar_patterns(deg: np.ndarray,
                        traces: list[tuple[str, np.ndarray]],
                        rmin: float = -60.0,
                        rmax: float = 0.0,
                        rticks: tuple = (-60, -40, -20, 0),
                        title: str = "Radiation Pattern (Polar)",
                        save_path: Optional[str] = None) -> None:
    """
    Polar plot similar to MATLAB 'polarpattern'.
      - deg: angles in degrees (0..360). Does not need to include 360.
      - traces: list of (label, values_dB) arrays; each will be normalized already.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Ensure angles are radians and monotonic 0..2π
    theta = np.deg2rad(np.mod(deg, 360.0))

    # If the first and last angle aren't the same, close the loop for prettier lines
    def _closed(x):
        return np.concatenate([x, x[:1]])

    theta_c = _closed(theta)

    fig = plt.figure(figsize=(7.0, 7.0))
    ax = fig.add_subplot(111, projection='polar')

    # Set radial limits (dB)
    ax.set_rlim(rmin, rmax)
    ax.set_rticks(list(rticks))
    ax.set_rlabel_position(90)  # put radial labels at the top for readability
    ax.grid(True)

    # Plot each trace; close the loop
    for label, y in traces:
        y = np.asarray(y, dtype=float)
        y_c = _closed(y)
        ax.plot(theta_c, y_c, linewidth=1.6, label=label)

    # Angle ticks every 15° like MATLAB look
    ax.set_thetagrids(np.arange(0, 360, 15))

    ax.set_title(title, va='bottom')
    # Legend below the plot
    leg = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=True)
    leg.get_frame().set_alpha(0.9)

    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved polar plot -> {save_path}")

    plt.show()
