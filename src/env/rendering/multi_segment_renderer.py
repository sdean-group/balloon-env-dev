"""Multi-segment renderer for visualizing sequential navigation segments.

Extends NavigationRenderer to draw multiple trajectory segments — each with
its own color, start marker, target marker and vicinity — on a single figure.

All export formats (animated HTML, GIF, MP4) work through the same
``_create_figure`` override — no format-specific rendering logic needed.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from .navigation_renderer import NavigationRenderer
from .rendering_utils import get_layout_config_2d, get_layout_config_3d
from ..utils.types import GridConfig

if TYPE_CHECKING:
    from ..field.flow_field import FlowField


_SEGMENT_COLORS = [
    "royalblue",
    "crimson",
    "forestgreen",
    "darkorange",
    "mediumpurple",
    "deeppink",
    "teal",
    "goldenrod",
]

_COLOR_TO_RGBA = {
    "royalblue": "65,105,225",
    "crimson": "220,20,60",
    "forestgreen": "34,139,34",
    "darkorange": "255,140,0",
    "mediumpurple": "147,112,219",
    "deeppink": "255,20,147",
    "teal": "0,128,128",
    "goldenrod": "218,165,32",
}


def _rgba(color: str, alpha: float) -> str:
    rgb = _COLOR_TO_RGBA.get(color, "128,128,128")
    return f"rgba({rgb},{alpha})"


class MultiSegmentRenderer(NavigationRenderer):
    """Renderer that accumulates multiple trajectory segments.

    Each segment is drawn with a unique color.  Completed segments retain
    their start marker, target + vicinity, and full trajectory.  The
    current (in-progress) segment is drawn alongside them.

    The renderer only collects data during ``step()``.  All rendering
    happens in ``_create_figure()``, which every exporter (animated HTML,
    GIF, MP4) calls via ``render()`` — no format-specific logic needed.
    """

    def __init__(
        self,
        config: GridConfig,
        width: int = 1024,
        height: int = 768,
        show_grid_points: bool = True,
        grid_subsample: Optional[int] = None,
        backend: str = "plotly",
        camera_eye: Optional[dict] = None,
        field: Optional["FlowField"] = None,
        show_field: bool = False,
    ):
        super().__init__(
            config=config,
            width=width,
            height=height,
            show_grid_points=show_grid_points,
            grid_subsample=grid_subsample,
            backend=backend,
            camera_eye=camera_eye,
            field=field,
            show_field=show_field,
        )
        self.segments: list[list] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        super().reset()
        self.segments = []

    def new_segment(self) -> None:
        """Finalize the current segment and start collecting a new one."""
        if self.states:
            self.segments.append(list(self.states))
            self.states = []

    # ------------------------------------------------------------------
    # Figure creation (override) — single source of truth
    # ------------------------------------------------------------------

    def _create_figure(self) -> go.Figure:
        """Build a Plotly figure showing all completed segments and the
        current in-progress segment.

        The GIF / MP4 exporters call this via ``render('rgb_array')``
        after temporarily slicing ``self.states``.  The animated-HTML
        exporter does the same with ``self.segments`` / ``self.states``.
        """
        fig = go.Figure()

        all_segs = self._all_segments()
        if not all_segs or not any(all_segs):
            return fig

        # Static background
        if self.show_grid_points:
            self._add_grid_points(fig)
        if self.show_field:
            self._add_field(fig)

        # Per-segment traces
        for idx, seg in enumerate(all_segs):
            if not seg:
                continue
            color = _SEGMENT_COLORS[idx % len(_SEGMENT_COLORS)]
            label = f"Seg {idx + 1}"
            self._add_segment(fig, seg, color, label)

        # Actor marker on the latest position
        latest = self._latest_state(all_segs)
        if latest is not None:
            self._add_actor(fig, latest)
            self._configure_layout(fig, latest)

        return fig

    # ------------------------------------------------------------------
    # Segment drawing
    # ------------------------------------------------------------------

    def _add_segment(self, fig: go.Figure, seg_states: list,
                     color: str, label: str) -> None:
        """Add vicinity, target, start, and trajectory traces for one segment."""
        first, last = seg_states[0], seg_states[-1]

        # Vicinity circle
        tp = last.target_position
        r = last.vicinity_radius
        theta = np.linspace(0, 2 * np.pi, 60)
        fig.add_trace(go.Scatter(
            x=tp.i + r * np.cos(theta), y=tp.j + r * np.sin(theta),
            mode="lines", fill="toself",
            fillcolor=_rgba(color, 0.15),
            line=dict(color=color, width=1.5, dash="dot"),
            name=f"{label} vicinity", showlegend=False, hoverinfo="skip",
        ))

        # Target marker
        fig.add_trace(go.Scatter(
            x=[tp.i], y=[tp.j], mode="markers",
            marker=dict(size=self.target_size, color=color, symbol="x",
                        line=dict(color="black", width=1.5)),
            name=f"{label} target", showlegend=True,
        ))

        # Start marker — use the actual position of the first state
        sp = first.position
        fig.add_trace(go.Scatter(
            x=[sp.i], y=[sp.j], mode="markers",
            marker=dict(size=self.target_size * 0.8, color=color,
                        symbol="circle",
                        line=dict(color="black", width=1.5), opacity=0.85),
            name=f"{label} start", showlegend=True,
        ))

        # Trajectory
        traj = np.array([[s.position.i, s.position.j] for s in seg_states])
        fig.add_trace(go.Scatter(
            x=traj[:, 0], y=traj[:, 1], mode="lines+markers",
            line=dict(color=color, width=self.trajectory_width),
            marker=dict(size=self.trajectory_width * 1.5, color=color,
                        opacity=0.7),
            name=label, showlegend=True, opacity=0.85,
        ))

    # ------------------------------------------------------------------
    # Layout (override) — shorter title
    # ------------------------------------------------------------------

    def _configure_layout(self, fig: go.Figure, state) -> None:
        seg_idx = getattr(state, "segment_index", 0)
        seg_step = getattr(state, "segment_step_count", state.step_count)
        seg_rew = getattr(state, "segment_cumulative_reward", 0.0)
        g_step = getattr(state, "global_step_count", state.step_count)
        g_rew = getattr(state, "global_cumulative_reward", 0.0)

        title_text = (
            f"<b>Seg {seg_idx + 1} · "
            f"step {g_step} ({seg_step}) · "
            f"seg {seg_rew:+.1f} · "
            f"total {g_rew:+.1f}</b>"
        )

        if self.ndim == 3:
            cfg = get_layout_config_3d(
                self.config, title_text, self.width, self.height,
                self.camera_eye,
            )
        else:
            cfg = get_layout_config_2d(
                self.config, title_text, self.width, self.height,
            )
        fig.update_layout(**cfg)

    # ------------------------------------------------------------------
    # Animated-HTML export (override)
    #
    # The GIF / MP4 exporters already work via the inherited
    # save_gif / save_mp4 → render('rgb_array') → _create_figure()
    # path.  For animated HTML we need Plotly frames with a fixed
    # trace count, so we override save_animated_html.
    # ------------------------------------------------------------------

    def save_animated_html(self, output_path: str, fps: int = 10) -> None:
        """Export animated HTML spanning all segments.

        Each frame is built by temporarily setting ``self.segments`` and
        ``self.states`` to the appropriate slice, then calling
        ``_create_figure()``.  This guarantees visual consistency with
        the static render and with GIF / MP4 exports.

        Plotly requires every frame to have the same trace count, so we
        pad short frames with invisible placeholders.
        """
        import os

        all_segs = self._all_segments()
        flat = [s for seg in all_segs for s in seg]
        if not flat:
            print("WARNING: No states recorded.")
            return

        n_segs = len(all_segs)
        boundaries = self._segment_boundaries(all_segs)
        frame_duration = int(1000 / fps)

        print(f"Creating animated HTML with {len(flat)} frames "
              f"across {n_segs} segments …")

        # Save originals
        orig_segments = self.segments
        orig_states = self.states

        # Build every frame via _create_figure with sliced data
        raw_figures: list[go.Figure] = []
        for global_idx in range(len(flat)):
            seg_idx, local_idx = self._locate_frame(global_idx, boundaries)
            # Completed segments stay in full
            self.segments = [list(all_segs[s]) for s in range(seg_idx)]
            # Active segment sliced up to current frame
            self.states = list(all_segs[seg_idx][: local_idx + 1])
            raw_figures.append(self._create_figure())

        # Restore originals
        self.segments = orig_segments
        self.states = orig_states

        # Determine the max trace count across all frames and pad
        max_traces = max(len(f.data) for f in raw_figures)
        _empty = go.Scatter(x=[], y=[], mode="markers",
                            showlegend=False, hoverinfo="skip")

        frames: list[go.Frame] = []
        for i, fig in enumerate(raw_figures):
            data = list(fig.data)
            while len(data) < max_traces:
                data.append(_empty)
            # Only pass the title as the per-frame layout; axis ranges
            # and all other layout properties are fixed on the base
            # layout so the viewport doesn't jump between frames.
            frame_layout = go.Layout(title=fig.layout.title)
            frames.append(go.Frame(
                data=data, name=f"frame{i}", layout=frame_layout,
            ))

        # Assemble animated figure
        anim_fig = go.Figure(
            data=frames[0].data,
            layout=self._get_animated_layout(),
            frames=frames,
        )
        anim_fig.update_layout(
            updatemenus=[dict(
                type="buttons", showactive=False,
                x=0.1, y=0.0, xanchor="left", yanchor="bottom",
                buttons=[
                    dict(label="Play", method="animate", args=[
                        None, {"frame": {"duration": frame_duration,
                                         "redraw": True},
                               "fromcurrent": True, "mode": "immediate",
                               "transition": {"duration": 0}}]),
                    dict(label="Pause", method="animate", args=[
                        [None], {"frame": {"duration": 0, "redraw": False},
                                 "mode": "immediate",
                                 "transition": {"duration": 0}}]),
                ],
            )],
            sliders=[{
                "active": 0, "yanchor": "top", "y": 0.0,
                "xanchor": "left", "x": 0.25,
                "currentvalue": {"prefix": "Frame: ", "visible": True,
                                 "xanchor": "left"},
                "pad": {"b": 10, "t": 10}, "len": 0.7,
                "steps": [
                    {"args": [[f"frame{i}"],
                              {"frame": {"duration": 0, "redraw": True},
                               "mode": "immediate",
                               "transition": {"duration": 0}}],
                     "method": "animate", "label": str(i)}
                    for i in range(len(frames))
                ],
            }],
        )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        anim_fig.write_html(output_path)
        print(f"Animated HTML saved to: {output_path}")
        print(f"   Frames: {len(frames)}")
        print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")

    # ------------------------------------------------------------------
    # GIF / MP4 exports (override)
    #
    # The parent exporters only slice self.states.  We need to slice
    # across self.segments + self.states so every frame sees the right
    # completed-segment / active-segment split.
    # ------------------------------------------------------------------

    def _render_frame_sequence(self, subsample: int = 1):
        """Yield RGB arrays for every frame (or every *subsample*-th)."""
        all_segs = self._all_segments()
        flat = [s for seg in all_segs for s in seg]
        if not flat:
            return

        boundaries = self._segment_boundaries(all_segs)
        orig_segments, orig_states = self.segments, self.states

        for global_idx in range(0, len(flat), subsample):
            seg_idx, local_idx = self._locate_frame(global_idx, boundaries)
            self.segments = [list(all_segs[s]) for s in range(seg_idx)]
            self.states = list(all_segs[seg_idx][: local_idx + 1])
            frame = self.render(mode="rgb_array")
            if frame is not None:
                yield frame

        self.segments = orig_segments
        self.states = orig_states

    def save_gif(self, output_path: str, fps: int = 10,
                 subsample: int = 1) -> None:
        import os
        try:
            import imageio.v2 as imageio
        except ImportError:
            print("ERROR: GIF export requires 'imageio' package")
            return

        frames = list(self._render_frame_sequence(subsample))
        if not frames:
            print("ERROR: No frames captured.")
            return

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        imageio.mimsave(output_path, frames, fps=fps, loop=0)
        print(f"GIF saved to: {output_path}  ({len(frames)} frames, "
              f"{os.path.getsize(output_path) / 1024:.1f} KB)")

    def save_mp4(self, output_path: str, fps: int = 15,
                 subsample: int = 1) -> None:
        import os
        try:
            import imageio.v2 as imageio
        except ImportError:
            print("ERROR: MP4 export requires 'imageio' package")
            return

        frames = list(self._render_frame_sequence(subsample))
        if not frames:
            print("ERROR: No frames captured.")
            return

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        try:
            imageio.mimwrite(output_path, frames, fps=fps, codec="libx264")
            print(f"MP4 saved to: {output_path}  ({len(frames)} frames, "
                  f"{os.path.getsize(output_path) / 1024:.1f} KB)")
        except Exception as e:
            print(f"ERROR: Could not save video: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _all_segments(self) -> list[list]:
        result = list(self.segments)
        if self.states:
            result.append(list(self.states))
        return result

    @staticmethod
    def _latest_state(all_segments):
        for seg in reversed(all_segments):
            if seg:
                return seg[-1]
        return None

    @staticmethod
    def _segment_boundaries(all_segments):
        boundaries, offset = [], 0
        for seg in all_segments:
            boundaries.append((offset, offset + len(seg)))
            offset += len(seg)
        return boundaries

    @staticmethod
    def _locate_frame(global_idx, boundaries):
        for seg_idx, (start, end) in enumerate(boundaries):
            if global_idx < end:
                return seg_idx, global_idx - start
        last = len(boundaries) - 1
        return last, global_idx - boundaries[last][0]
