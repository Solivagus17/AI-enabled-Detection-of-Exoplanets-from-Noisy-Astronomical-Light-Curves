"""
ExoHunter Visualization Module
Author: Sarthakk Anjariya

This module generates interactive Plotly visualizations of TESS light curves and
transit detections for the Streamlit dashboard.

Astrophysical reasoning:
- Time-series plots: Essential to inspect the detrending and verify that transit dips
  are not instrumental artifacts or stellar flares.
- Folded curves: Folding at the correct period piles up all transits on top of each other,
  boosting the signal-to-noise ratio and revealing the characteristic U-shape of planetary transits
  or V-shape of stellar eclipses.
"""

import plotly.graph_objects as go
import numpy as np

# Premium Color Palette (HSL tailormade styles)
COLOR_RAW = "#636EFA"       # Deep Purple/Blue
COLOR_TRANSIT = "#EF553B"   # Vibrant Coral
COLOR_BINNED = "#00CC96"    # Emerald Green
COLOR_FIT = "#FF7F0E"       # Energetic Orange
COLOR_BG = "#111726"        # Deep Navy Slate
COLOR_GRID = "#2A3F5F"      # Gridlines

def plot_raw_lc(time, flux, tic_id, period=None, epoch=None, duration_days=None):
    """
    Generate an interactive Plotly plot of the raw detrended light curve.
    Highlights transit regions if transit parameters are provided.
    """
    fig = go.Figure()
    
    # Base scatter plot
    fig.add_trace(go.Scatter(
        x=time,
        y=flux,
        mode='markers',
        marker=dict(size=3, color=COLOR_RAW, opacity=0.6),
        name='Detrended Flux',
        hovertemplate='Time: %{x:.4f} days<br>Flux: %{y:.5f}<extra></extra>'
    ))
    
    # Highlight transits
    if period is not None and epoch is not None and duration_days is not None:
        # Calculate transit midpoints within the time range
        t_start = np.min(time)
        t_end = np.max(time)
        
        # N_transits back and forward
        n_min = int(np.floor((t_start - epoch) / period))
        n_max = int(np.ceil((t_end - epoch) / period))
        
        shapes = []
        for n in range(n_min, n_max + 1):
            t_mid = epoch + n * period
            t1 = t_mid - duration_days / 2.0
            t2 = t_mid + duration_days / 2.0
            
            # Check overlap with data range
            if t2 >= t_start and t1 <= t_end:
                # Add highlighting rectangles
                shapes.append(dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=t1,
                    y0=0,
                    x1=t2,
                    y1=1,
                    fillcolor=COLOR_TRANSIT,
                    opacity=0.15,
                    layer="below",
                    line_width=0,
                ))
        fig.update_layout(shapes=shapes)
        
        # Add a dummy trace for legend
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=COLOR_TRANSIT, opacity=0.3, symbol='square'),
            name='Detected Transit Window'
        ))
        
    fig.update_layout(
        title=dict(
            text=f"Detrended Light Curve - {tic_id}",
            font=dict(size=18, color="#FFFFFF")
        ),
        xaxis=dict(
            title="Time (BJD - 2457000, days)",
            gridcolor=COLOR_GRID,
            zeroline=False,
            color="#FFFFFF"
        ),
        yaxis=dict(
            title="Normalized Flux",
            gridcolor=COLOR_GRID,
            zeroline=False,
            color="#FFFFFF",
            tickformat=".4f"
        ),
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        legend=dict(
            font=dict(color="#FFFFFF"),
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=60, r=30, t=50, b=50),
        hovermode='closest'
    )
    return fig

def plot_folded_lc(phases, fluxes, tic_id, period, depth_ppm, duration_hrs):
    """
    Generate an interactive Plotly plot of the phase-folded light curve.
    Overlays the binned points and the best-fit box model representation.
    """
    fig = go.Figure()
    
    # 1. Add all points as light background markers
    # For phase folding, phases are from -0.5 to 0.5
    fig.add_trace(go.Scatter(
        x=phases,
        y=fluxes,
        mode='markers',
        marker=dict(size=4, color=COLOR_RAW, opacity=0.3),
        name='Folded Data',
        hovertemplate='Phase: %{x:.4f}<br>Flux: %{y:.5f}<extra></extra>'
    ))
    
    # 2. Add median binned points
    # Binned points are passed in directly
    fig.add_trace(go.Scatter(
        x=phases,
        y=fluxes,
        mode='lines+markers',
        line=dict(color=COLOR_BINNED, width=2.5),
        marker=dict(size=6, color=COLOR_BINNED),
        name='Phase Binned Median',
        hovertemplate='Phase: %{x:.4f}<br>Binned Flux: %{y:.5f}<extra></extra>'
    ))
    
    # 3. Overlay best-fit BLS box model
    # Convert duration in hours to phase width
    duration_days = duration_hrs / 24.0
    phase_width = duration_days / period
    depth_val = depth_ppm / 1e6
    
    # Define box coordinates
    # Flat before transit -> drop at -phase_width/2 -> flat at 1 - depth -> rise at phase_width/2 -> flat after
    box_phases = [-0.5, -phase_width/2.0, -phase_width/2.0, phase_width/2.0, phase_width/2.0, 0.5]
    box_fluxes = [1.0, 1.0, 1.0 - depth_val, 1.0 - depth_val, 1.0, 1.0]
    
    fig.add_trace(go.Scatter(
        x=box_phases,
        y=box_fluxes,
        mode='lines',
        line=dict(color=COLOR_FIT, width=3, dash='dash'),
        name='BLS Box Model Fit'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Phase Folded Transit Model - {tic_id}",
            font=dict(size=18, color="#FFFFFF")
        ),
        xaxis=dict(
            title="Phase (Transit Centered at 0.0)",
            gridcolor=COLOR_GRID,
            zeroline=False,
            color="#FFFFFF",
            range=[-0.5, 0.5]
        ),
        yaxis=dict(
            title="Normalized Flux",
            gridcolor=COLOR_GRID,
            zeroline=False,
            color="#FFFFFF",
            tickformat=".4f"
        ),
        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,
        legend=dict(
            font=dict(color="#FFFFFF"),
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=60, r=30, t=50, b=50),
        hovermode='closest'
    )
    return fig
