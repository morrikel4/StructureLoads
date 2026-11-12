"""
live_variance_plot.py  (utils/)

Live load-cell diagnostics for the angle sweep. One figure, redrawn after
every motor angle:
    top:    mean force  (Fx, Fy, Fz)  vs motor angle
    middle: mean moment (Mx, My, Mz)  vs motor angle
    bottom: variance of the force magnitude Var(|F|) vs motor angle

Usage in main.py:
    plotter = LivePlotter()
    plotter.add_point(motor_angle_deg, mean_load, load_variance)   # mean_load = [Fx..Mz]
    plotter.pump()   # during long blocking operations
    plotter.save(OUTPUT_DIR); plotter.close()
"""

import os

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

FORCE_LABELS = ["Fx", "Fy", "Fz"]
MOMENT_LABELS = ["Mx", "My", "Mz"]
COLORS = ["tab:blue", "tab:orange", "tab:green"]


class LivePlotter:
    def __init__(self, enabled=True, block=False, figsize=(8, 9), dpi=100):
        self.enabled = enabled and _HAS_MPL
        if not self.enabled:
            if enabled and not _HAS_MPL:
                print("  WARN: matplotlib not available, live plotting disabled")
            return
        plt.ion()
        self.fig, (self.ax_f, self.ax_m, self.ax_v) = plt.subplots(
            3, 1, figsize=figsize, dpi=dpi, num="Live diagnostics", sharex=True)

        self.lines_f = [self.ax_f.plot([], [], "o-", color=c, label=l)[0]
                        for c, l in zip(COLORS, FORCE_LABELS)]
        self.ax_f.set_ylabel("Mean force (N)")
        self.ax_f.legend(loc="upper right", ncol=3, fontsize=8)

        self.lines_m = [self.ax_m.plot([], [], "o-", color=c, label=l)[0]
                        for c, l in zip(COLORS, MOMENT_LABELS)]
        self.ax_m.set_ylabel("Mean moment (N·m)")
        self.ax_m.legend(loc="upper right", ncol=3, fontsize=8)

        (self.line_v,) = self.ax_v.plot([], [], "s-", color="tab:red")
        self.ax_v.set_ylabel("Var(|F|) (N$^2$)")
        self.ax_v.set_xlabel("Motor angle (deg)")

        for ax in (self.ax_f, self.ax_m, self.ax_v):
            ax.grid(alpha=0.3)
        self.fig.tight_layout()

        self._angles, self._mean, self._var = [], [], []
        plt.show(block=block)
        plt.pause(0.1)

    def pump(self):
        """Service the GUI event loop without adding data."""
        if not self.enabled:
            return
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def add_point(self, motor_angle_deg, mean_load, load_variance):
        """Append one angle's result (mean_load = [Fx, Fy, Fz, Mx, My, Mz]) and redraw."""
        if not self.enabled or motor_angle_deg is None:
            return
        self._angles.append(motor_angle_deg)
        self._mean.append(np.asarray(mean_load, dtype=float))
        self._var.append(np.nan if load_variance is None else load_variance)

        x = np.array(self._angles)
        o = np.argsort(x)
        mean = np.array(self._mean)[o]
        for k, line in enumerate(self.lines_f):
            line.set_data(x[o], mean[:, k])
        for k, line in enumerate(self.lines_m):
            line.set_data(x[o], mean[:, 3 + k])
        self.line_v.set_data(x[o], np.array(self._var)[o])
        for ax in (self.ax_f, self.ax_m, self.ax_v):
            ax.relim()
            ax.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def save(self, output_dir):
        if self.enabled:
            self.fig.savefig(os.path.join(output_dir, "live_diagnostics.png"), dpi=150)

    def close(self):
        if self.enabled:
            plt.close(self.fig)
