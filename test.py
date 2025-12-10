import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal

from seiswiggle import wiggle_plot


def _ricker(frequency: float, length: float, dt: float) -> np.ndarray:
    """Ricker wavelet for synthetic data."""
    t = np.arange(-length / 2, length / 2 + dt, dt)
    a = (np.pi * frequency * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def _synthetic_gather(
    n_traces: int = 30,
    n_samples: int = 500,
    dt: float = 0.002,
    offset_max: float = 550.0,
    f0: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Creates a seismogram with direct wave, reflection hyperbola, and harmonic noise."""
    offsets = np.linspace(0, offset_max, n_traces)
    time = np.arange(n_samples) * dt
    data = np.zeros((n_traces, n_samples), dtype=float)

    wavelet = _ricker(f0, length=0.08, dt=dt)
    hw = len(wavelet) // 2

    t0_direct = 0.05
    v_direct = 1800.0
    t0_reflect = 0.52
    v_rms = 2500.0

    def _add_event(trace_idx: int, arrival_time: float, amp: float) -> None:
        center = int(arrival_time / dt)
        start = max(center - hw, 0)
        end = min(center + hw + 1, n_samples)
        w_start = hw - (center - start)
        w_end = w_start + (end - start)
        data[trace_idx, start:end] += amp * wavelet[w_start:w_end]

    def _add_harmonic_noise(trace_idx: int) -> None:
        # Multiple harmonics with different phases to degrade the frequency band
        freqs = [8.0, 70.0, 120.0]
        phases = np.random.rand(len(freqs)) * 2 * np.pi
        amp = 0.05  # relative noise amplitude
        noise = sum(
            np.sin(2 * np.pi * f * time + ph) for f, ph in zip(freqs, phases)
        )
        data[trace_idx] += amp * noise

    for i, x in enumerate(offsets):
        t_direct = t0_direct + abs(x) / v_direct
        t_hyper = np.sqrt(t0_reflect**2 + (x / v_rms) ** 2)
        _add_event(i, t_direct, amp=1.0)
        _add_event(i, t_hyper, amp=0.7)
        _add_harmonic_noise(i)

    return data, time, offsets


def _zero_phase_bandpass(
    traces: np.ndarray,
    dt: float,
    lowcut: float = 12.0,
    highcut: float = 40.0,
    order: int = 4
) -> np.ndarray:
    """Zero-phase bandpass filter using scipy.signal."""
    nyquist = 0.5 / dt
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Create Butterworth bandpass filter
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    
    # Apply zero-phase filtering (filtfilt) to each trace
    filtered = np.zeros_like(traces)
    for i in range(traces.shape[0]):
        filtered[i] = signal.sosfiltfilt(sos, traces[i])
    
    return filtered


def test_wiggle_plot_synthetic():
    """Tests synthetic seismogram display in two ways."""
    data, time, offsets = _synthetic_gather()
    out_dir = Path.cwd()
    pics_dir = out_dir / "pics"
    pics_dir.mkdir(exist_ok=True)
    print(f"Figures saved to {pics_dir}")  # so results can be opened manually

    # Option with automatic figure/axis creation. All default parameters.
    fig_auto, ax_auto = wiggle_plot(
        data
    )    

    assert fig_auto is not None
    assert ax_auto is not None  

    ax_auto.set_title("Solo Wiggle Plot. All defaults")
    ax_auto.set_ylabel("Time (s)")
    ax_auto.set_xlabel("Offset (m)")
    auto_path = pics_dir / "solo_auto.png"
    fig_auto.savefig(auto_path, dpi=150, bbox_inches='tight')
    plt.close(fig_auto)
    assert auto_path.exists()

    # Option with automatic figure/axis creation. Horizontal time axis.
    # Brown fill color.
    fig_horizontal, ax_horizontal = wiggle_plot(
        data,
        time_values=time,
        trace_values=offsets,
        vertical_time=False,
        gain=2.0,
        clip=1.0,
        fill_color="brown"    
    )    

    assert fig_horizontal is not None
    assert ax_auto is not None  

    ax_auto.set_title("Solo Wiggle Plot. Horizontal time (VSP-style). Fill color: brown. Gain: 2.0")
    ax_auto.set_ylabel("Offset (m)")
    ax_auto.set_xlabel("Time (s)")
    horizontal_path = pics_dir / "solo_horizontal.png"
    fig_horizontal.savefig(horizontal_path, dpi=150, bbox_inches='tight')
    plt.close(fig_horizontal)

    assert horizontal_path.exists()

    # Option in provided subplot grid
    fig, axes = plt.subplots(1, 3, figsize=(15, 10), tight_layout=True)
    fig.suptitle("Wiggle plots inside Pyplot Subplots")
    dt = time[1] - time[0]  # time step
    filtered = _zero_phase_bandpass(data, dt=dt, lowcut=12.0, highcut=40.0)
    residual = data - filtered

    wiggle_plot(data, time_values=time, trace_values=offsets, axis=axes[0], gain=1.0)
    wiggle_plot(filtered, time_values=time, trace_values=offsets, axis=axes[1], gain=1.0)
    wiggle_plot(residual, time_values=time, trace_values=offsets, axis=axes[2], gain=1.0)

    axes[0].set_title("Original", pad=20)
    axes[1].set_title("Filtered", pad=20)
    axes[2].set_title("Residual", pad=20)
    axes[0].set_ylabel("Time (s)")
    axes[1].set_ylabel("Time (s)")
    axes[2].set_ylabel("Time (s)")
    axes[0].set_xlabel("Offset (m)")
    axes[1].set_xlabel("Offset (m)")
    axes[2].set_xlabel("Offset (m)")

    for ax in axes:
        # Check that lines are added
        assert len(ax.lines) > 0

    grid_path = pics_dir / "subplots.png"
    fig.savefig(grid_path, dpi=150, bbox_inches='tight')
    assert grid_path.exists()
    print(f"Saved: {auto_path}")
    print(f"Saved: {grid_path}")

    
    plt.close(fig)


if __name__ == "__main__":
    test_wiggle_plot_synthetic()
    print("Test completed successfully!")

