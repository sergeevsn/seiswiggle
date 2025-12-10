import numpy as np
import matplotlib.pyplot as plt


def wiggle_plot(
    data, # data in format (n_traces, n_samples)
    time_values=None, # time samples in format (n_samples)
    trace_values=None, # trace coordinates in format (n_traces)
    axis=None, # axis for plotting, if plot is created in context of a more general pyplot figure    
    gain=1.0, # gain factor for data normalization
    vertical_time=True, # time axis orientation, default is vertical
    fill_color='black', # fill color
    fill_alpha=1.0, # fill transparency
    line_color='black', # line color
    line_width=0.5, # line width
    clip=1.0, # clipping threshold for data normalization
    t_lim=None, # time limits
    x_lim=None, # trace coordinate limits
    figsize=(8, 6), # figure size
    time_grid=True # enable grid along time axis
    ):
    """
    Fast and universal wiggle plot.
    If axis=None — automatically creates a new figure.
    If time_values=None → uses np.arange(n_samples)
    If trace_values=None → uses np.arange(n_traces)
    If time_grid=True → adds grid lines along the time axis.
    """

    # -----------------------------------------
    # Create axis if axis is not provided
    # -----------------------------------------
    created_axis = False
    if axis is None:
        fig, axis = plt.subplots(figsize=figsize)
        created_axis = True
    else:
        fig = axis.figure

    # -----------------------------------------
    # Original dimensions
    # -----------------------------------------
    n_traces, n_samples = data.shape

    # -----------------------------------------
    # Fallback for time and trace coordinates
    # -----------------------------------------
    if time_values is None:
        time_values = np.arange(n_samples)

    if trace_values is None:
        trace_values = np.arange(n_traces)

    t = time_values

    # -----------------------------------------
    # Spacing between traces
    # -----------------------------------------
    if n_traces > 1:
        trace_spacing = np.median(np.diff(trace_values))
    else:
        trace_spacing = 1.0

    # -----------------------------------------
    # Normalization and clipping
    # -----------------------------------------
    max_val = np.abs(data).max() or 1.0

    if clip is not None:
        data = np.clip(data, -clip * max_val, clip * max_val)

    scalar = (trace_spacing * gain) / max_val
    data_scaled = data * scalar

    base = trace_values[:, None]   # for vectorization

    # -----------------------------------------
    # Data orientation
    # -----------------------------------------
    if vertical_time:
        x = base + data_scaled
        y = np.broadcast_to(t, (n_traces, n_samples))
        fill_func = axis.fill_betweenx
    else:
        x = np.broadcast_to(t, (n_traces, n_samples))
        y = base + data_scaled
        fill_func = axis.fill_between

    # -----------------------------------------
    # Plot all traces with a single plot()
    # -----------------------------------------
    axis.plot(x.T, y.T, color=line_color, linewidth=line_width)

    # -----------------------------------------
    # Fill
    # -----------------------------------------
    for i in range(n_traces):
        fill_func(
            t,
            trace_values[i],
            trace_values[i] + data_scaled[i],
            where=(data_scaled[i] > 0),
            color=fill_color,
            alpha=fill_alpha
        )

    # -----------------------------------------
    # Axis configuration
    # -----------------------------------------
    if vertical_time:
        axis.set_ylabel("Time")
        axis.set_xlabel("Trace")
        axis.invert_yaxis()

        if t_lim:
            axis.set_ylim(t_lim[1], t_lim[0])
        if x_lim:
            axis.set_xlim(x_lim)

        # Add grid along time axis (vertical lines)
        if time_grid:
            axis.grid(True, axis='y', alpha=0.5)

    else:
        axis.set_xlabel("Time")
        axis.set_ylabel("Trace")
        axis.invert_yaxis()

        if t_lim:
            axis.set_xlim(t_lim)
        if x_lim:
            axis.set_ylim(x_lim[1], x_lim[0])

        # Add grid along time axis (horizontal lines)
        if time_grid:
            axis.grid(True, axis='x', alpha=0.5)

    # Move horizontal axis ticks and labels to the top
    axis.xaxis.tick_top()
    axis.xaxis.set_label_position('top')
    # Increase padding for axis label to create gap between x_label and title
    axis.xaxis.labelpad = 10

    # Adjust figure layout if axis was created to provide more space at the top
    if created_axis:
        fig.subplots_adjust(top=0.92)

    # -----------------------------------------
    # Return
    # -----------------------------------------
    if created_axis:
        return fig, axis
    else:
        return axis
