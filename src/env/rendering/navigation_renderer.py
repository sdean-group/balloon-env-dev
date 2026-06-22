from typing import List, Union, Optional, TYPE_CHECKING
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff

from .renderer import Renderer
from .rendering_utils import (
    compute_scaling,
    get_layout_config_2d,
    get_layout_config_3d,
    get_animated_layout_2d,
    get_animated_layout_3d,
    fig_to_array,
)
from ..utils.types import GridConfig, NavigationArenaState, GridPosition

if TYPE_CHECKING:
    from ..field.flow_field import FlowField


class NavigationRenderer(Renderer):
    """Navigation arena renderer using Plotly backend.

    Supports both 2D and 3D navigation tasks:
    - 2D: Uses Scatter plots, vicinity shown as circle
    - 3D: Uses Scatter3d plots, vicinity shown as cylinder

    Core visualization logic for navigation tasks. Supports:
    - Standard Gymnasium modes: 'human' (interactive), 'rgb_array' (numpy array)
    - Export methods: save_gif(), save_mp4(), save_html(), save_animated_html()
    """
    
    def __init__(
        self,
        config: GridConfig,
        width: int = 1024,
        height: int = 768,
        show_grid_points: bool = True,
        grid_subsample: Optional[int] = None,
        backend: str = 'plotly',
        camera_eye: Optional[dict] = None,
        field: Optional['FlowField'] = None,
        show_field: bool = False
    ):
        """Initialize navigation renderer.
        
        Args:
            config: Grid configuration.
            width: Figure width in pixels.
            height: Figure height in pixels.
            show_grid_points: Whether to show grid points.
            grid_subsample: Subsample factor for grid points and field arrows (auto if None).
            backend: Rendering backend ('plotly').
            camera_eye: Camera position for 3D (ignored for 2D).
                       Defaults to (1.5, -1.5, 1.0) for 45 degree perspective.
                       Examples:
                         - Top-down: {'x': 0, 'y': 0, 'z': 2.5}
                         - Side view: {'x': 2.5, 'y': 0, 'z': 0}
                         - Isometric: {'x': 1.5, 'y': -1.5, 'z': 1.0}
            field: Optional FlowField for visualizing the velocity field.
            show_field: Whether to show field arrows (requires field with velocity_field/velocity_at).
        """
        self.config = config
        self.width = width
        self.height = height
        self.show_grid_points = show_grid_points
        self.backend = backend
        self.camera_eye = camera_eye or {'x': 1.5, 'y': -1.5, 'z': 1.0}
        self.field = field
        self.show_field = show_field
        
        self._compute_scaling()
        
        # Determine grid subsampling
        if grid_subsample is None:
            if self.ndim == 3:
                total = config.n_x * config.n_y * config.n_z
                grid_subsample = max(2, int(np.cbrt(total / 1000))) if total > 10000 else (2 if total > 1000 else 1)
            else:
                total = config.n_x * config.n_y
                grid_subsample = max(2, int(np.sqrt(total / 500))) if total > 1000 else (2 if total > 200 else 1)
        self.grid_subsample = grid_subsample
        
        # Episode data
        self.states: List[NavigationArenaState] = []
        
        # Cache for static traces (field doesn't change during episode)
        self._cached_field_trace = None
        self._cached_grid_trace = None
    
    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return self.config.ndim
    
    def reset(self) -> None:
        """Reset renderer for new episode."""
        self.states = []
        self._cached_field_trace = None
        self._cached_grid_trace = None
    
    def step(self, state: NavigationArenaState) -> None:
        """Record navigation arena state for visualization."""
        self.states.append(state)
    
    def render(self, mode: str) -> Union[None, np.ndarray, str]:
        """Render the visualization.
        
        Args:
            mode: 'human' (show in browser) or 'rgb_array' (return numpy array).
            
        Returns:
            None for 'human', numpy array (H, W, 3) for 'rgb_array'.
        """
        if mode not in self.render_modes:
            raise ValueError(f"Unsupported render mode: {mode}. Use one of {self.render_modes}")
        
        fig = self._create_figure()
        
        if mode == 'human':
            fig.show()
            return None
        elif mode == 'rgb_array':
            return fig_to_array(fig, self.width, self.height)

    @property
    def render_modes(self) -> List[str]:
        """Supported render modes."""
        return ['human', 'rgb_array']
    
    # ========================================================================
    # Export Methods
    # ========================================================================
    
    def save_gif(self, output_path: str, fps: int = 10, subsample: int = 1) -> None:
        from .exporters import save_gif
        save_gif(self, output_path, fps, subsample)
    
    def save_mp4(self, output_path: str, fps: int = 15, subsample: int = 1) -> None:
        from .exporters import save_mp4
        save_mp4(self, output_path, fps, subsample=subsample)
    
    def save_html(self, output_path: str) -> None:
        from .exporters import save_html
        save_html(self, output_path)
    
    def save_animated_html(self, output_path: str, fps: int = 10) -> None:
        from .exporters import save_animated_html
        save_animated_html(self, output_path, fps)
    
    # ========================================================================
    # Internal visualization methods
    # ========================================================================
    
    def _create_figure(self) -> go.Figure:
        """Create Plotly figure with all visualization elements."""
        fig = go.Figure()
        
        if not self.states:
            return fig
        
        current_state = self.states[-1]
        
        if self.show_grid_points:
            self._add_grid_points(fig)
        if self.show_field:
            self._add_field(fig)
        
        self._add_target_vicinity(fig, current_state)
        self._add_target(fig, current_state)
        self._add_initial_position(fig, current_state)
        self._add_trajectory(fig)
        self._add_actor(fig, current_state)
        self._configure_layout(fig, current_state)
        
        return fig

    def _configure_layout(self, fig: go.Figure, state: NavigationArenaState):
        """Configure figure layout."""
        action_names = ['DEC', 'STAY', 'INC']
        action_name = action_names[state.last_action] if state.last_action is not None else 'N/A'
        
        title_text = (
            f"<b>Step: {state.step_count} | "
            f"Action: {action_name} | "
            f"Reward: {state.last_reward:+.2f} | "
            f"Cumulative: {state.cumulative_reward:+.2f}</b>"
        )
        
        if self.ndim == 3:
            layout_config = get_layout_config_3d(
                self.config, title_text, self.width, self.height, self.camera_eye
            )
        else:
            layout_config = get_layout_config_2d(
                self.config, title_text, self.width, self.height
            )
        fig.update_layout(**layout_config)
    
    def _compute_scaling(self):
        """Compute marker sizes based on grid dimensions."""
        scaling = compute_scaling(self.config)
        self.grid_point_size = scaling['grid_point_size']
        self.actor_size = scaling['actor_size']
        self.target_size = scaling['target_size']
        self.trajectory_width = scaling['trajectory_width']
    
    # ========================================================================
    # Visualization elements (dimension-aware)
    # ========================================================================
    
    def _add_grid_points(self, fig: go.Figure):
        """Add subsampled grid points."""
        fig.add_trace(self._get_grid_points_trace())
    
    def _add_field(self, fig: go.Figure):
        """Add field mean displacement arrows."""
        trace = self._get_field_trace()
        if trace is not None:
            if isinstance(trace, list):
                for t in trace:
                    fig.add_trace(t)
            else:
                fig.add_trace(trace)
    
    def _add_target_vicinity(self, fig: go.Figure, state: NavigationArenaState):
        """Add target vicinity region (cylinder for 3D, circle for 2D)."""
        fig.add_trace(self._get_target_vicinity_trace(state))
    
    def _add_target(self, fig: go.Figure, state: NavigationArenaState):
        """Add target marker."""
        fig.add_trace(self._get_target_trace(state))
    
    def _add_initial_position(self, fig: go.Figure, state: NavigationArenaState):
        """Add initial position marker."""
        fig.add_trace(self._get_initial_position_trace(state))
    
    def _add_trajectory(self, fig: go.Figure):
        """Add trajectory line."""
        fig.add_trace(self._get_trajectory_trace())
    
    def _add_actor(self, fig: go.Figure, state: NavigationArenaState):
        """Add current actor position."""
        fig.add_trace(self._get_actor_trace(state))
    
    # ========================================================================
    # Trace data methods (for use by exporters)
    # ========================================================================
    
    def _get_grid_points_trace(self):
        """Get grid points as Plotly trace (cached)."""
        if self._cached_grid_trace is not None:
            return self._cached_grid_trace
        
        s = self.grid_subsample
        xs = np.arange(1, self.config.n_x + 1, s)
        ys = np.arange(1, self.config.n_y + 1, s)
        marker = dict(size=self.grid_point_size, color='gray', opacity=0.4)
        
        if self.ndim == 3:
            zs = np.arange(1, self.config.n_z + 1, s)
            gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
            trace = go.Scatter3d(
                x=gx.ravel(), y=gy.ravel(), z=gz.ravel(),
                mode='markers', marker=marker,
                name='Grid', showlegend=False, hoverinfo='skip'
            )
        else:
            gx, gy = np.meshgrid(xs, ys, indexing='ij')
            trace = go.Scatter(
                x=gx.ravel(), y=gy.ravel(),
                mode='markers', marker=marker,
                name='Grid', showlegend=False, hoverinfo='skip'
            )
        
        self._cached_grid_trace = trace
        return trace
    
    def _get_field_trace(self):
        """Get field mean displacement as Plotly trace(s) (cached).
        
        Returns:
            - 2D: List of Scatter traces from ff.create_quiver
            - 3D: Single Scatter3d trace of NaN-separated line-segment arrows
            - None if field not available or all arrows are zero
        """
        if self._cached_field_trace is not None:
            return self._cached_field_trace
        
        if self.field is None:
            return None
        
        result = self._build_field_trace()
        self._cached_field_trace = result
        return result
    
    def _build_field_trace(self):
        """Build field quiver arrows for 2D or 3D.
        
        Shared pipeline: subsample grid -> get (u, v) vectors -> check magnitude -> render.
        Uses bulk field array when available, pointwise query as fallback.
        
        2D fields have v=0 (field only affects ambient axis x).
        3D fields have (u, v) at each (x, y, z) level, rendered as flat arrows at each z.
        """
        s = self.grid_subsample
        is_3d = self.ndim == 3
        
        # Build subsampled grid
        xs = np.arange(1, self.config.n_x + 1, s, dtype=float)
        ys = np.arange(1, self.config.n_y + 1, s, dtype=float)
        if is_3d:
            zs = np.arange(1, self.config.n_z + 1, s, dtype=float)
            gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
        else:
            gx, gy = np.meshgrid(xs, ys, indexing='ij')
        
        # Get velocity vectors: prefer bulk array, fall back to pointwise
        field_arr = self.field.velocity_field() if hasattr(self.field, 'velocity_field') else None
        if field_arr is not None:
            if is_3d:
                u = field_arr[::s, ::s, ::s, 0].ravel()
                v = field_arr[::s, ::s, ::s, 1].ravel()
            else:
                u = field_arr[::s, ::s, 0].ravel()
                v = np.zeros_like(u)
        elif hasattr(self.field, 'velocity_at'):
            flat_coords = np.column_stack(
                [gx.ravel(), gy.ravel()] + ([gz.ravel()] if is_3d else [])
            )
            u = np.zeros(len(flat_coords))
            v = np.zeros(len(flat_coords))
            for idx, coords in enumerate(flat_coords):
                pos = GridPosition(int(coords[0]), int(coords[1]),
                                   int(coords[2]) if is_3d else None)
                vel = self.field.velocity_at(pos)
                if vel is not None:
                    u[idx] = vel[0]
                    v[idx] = vel[1] if (len(vel) > 1 and vel[1] is not None) else 0.0
        else:
            return None
        
        speed = np.sqrt(u**2 + v**2)
        max_mag = speed.max()
        if max_mag < 1e-6:
            return None
        
        arrow_scale = 0.7 * s / max_mag
        
        if is_3d:
            return self._quiver_scatter3d(gx, gy, gz, u, v, speed, max_mag, arrow_scale)
        else:
            return self._quiver_scatter2d(gx, gy, u, v, arrow_scale)
    
    @staticmethod
    def _quiver_scatter3d(gx, gy, gz, u, v, speed, max_mag, arrow_scale):
        """Render quiver arrows as NaN-separated Scatter3d line segments."""
        x0, y0, z0 = gx.ravel(), gy.ravel(), gz.ravel()
        x1 = x0 + arrow_scale * u
        y1 = y0 + arrow_scale * v
        
        n = len(x0)
        px = np.empty(3 * n);  px[0::3] = x0;  px[1::3] = x1;  px[2::3] = np.nan
        py = np.empty(3 * n);  py[0::3] = y0;  py[1::3] = y1;  py[2::3] = np.nan
        pz = np.empty(3 * n);  pz[0::3] = z0;  pz[1::3] = z0;  pz[2::3] = np.nan
        
        return go.Scatter3d(
            x=px, y=py, z=pz,
            mode='lines',
            line=dict(color=np.repeat(speed, 3), colorscale='Blues',
                      width=3, cmin=0.0, cmax=float(max_mag)),
            opacity=0.7, name='Field', showlegend=True, hoverinfo='skip',
        )
    
    @staticmethod
    def _quiver_scatter2d(gx, gy, u, v, arrow_scale):
        """Render quiver arrows using ff.create_quiver for 2D."""
        quiver_fig = ff.create_quiver(
            gx.ravel().tolist(), gy.ravel().tolist(),
            u.tolist(), v.tolist(),
            scale=arrow_scale,
            arrow_scale=0.4,
            line=dict(color='steelblue', width=1.5),
            name='Field'
        )
        traces = list(quiver_fig.data)
        for i, trace in enumerate(traces):
            trace.showlegend = (i == 0)
            trace.name = 'Field' if i == 0 else None
        return traces
    
    def _get_target_vicinity_trace(self, state: NavigationArenaState):
        """Get target vicinity region as Plotly trace."""
        tp = state.target_position
        r = state.vicinity_radius
        
        if self.ndim == 3:
            theta = np.linspace(0, 2 * np.pi, 40)
            z_levels = np.linspace(1, self.config.n_z, 30)
            theta_grid, z_grid = np.meshgrid(theta, z_levels)
            
            return go.Surface(
                x=tp.i + r * np.cos(theta_grid),
                y=tp.j + r * np.sin(theta_grid),
                z=z_grid,
                colorscale=[[0, 'lightgreen'], [1, 'lightgreen']],
                opacity=0.25, showscale=False, showlegend=False,
                hoverinfo='skip', name='Vicinity'
            )
        else:
            theta = np.linspace(0, 2 * np.pi, 60)
            return go.Scatter(
                x=tp.i + r * np.cos(theta),
                y=tp.j + r * np.sin(theta),
                mode='lines', fill='toself',
                fillcolor='rgba(144, 238, 144, 0.3)',
                line=dict(color='lightgreen', width=2),
                name='Vicinity', showlegend=False, hoverinfo='skip'
            )
    
    def _get_target_trace(self, state: NavigationArenaState):
        """Get target marker as Plotly trace."""
        tp = state.target_position
        if self.ndim == 3:
            return go.Scatter3d(
                x=[tp.i], y=[tp.j], z=[tp.k],
                mode='markers',
                marker=dict(size=self.target_size * 0.5, color='green',
                            symbol='x', line=dict(color='darkgreen', width=2)),
                name='Target', showlegend=True
            )
        else:
            return go.Scatter(
                x=[tp.i], y=[tp.j],
                mode='markers',
                marker=dict(size=self.target_size, color='green',
                            symbol='x', line=dict(color='darkgreen', width=2)),
                name='Target', showlegend=True
            )
    
    def _get_initial_position_trace(self, state: NavigationArenaState):
        """Get initial position marker as Plotly trace."""
        ip = state.initial_position
        marker = dict(size=self.target_size * 0.8, color='orange',
                      symbol='circle', line=dict(color='darkorange', width=2), opacity=0.8)
        if self.ndim == 3:
            return go.Scatter3d(
                x=[ip.i], y=[ip.j], z=[ip.k],
                mode='markers', marker=marker, name='Start', showlegend=True
            )
        else:
            return go.Scatter(
                x=[ip.i], y=[ip.j],
                mode='markers', marker=marker, name='Start', showlegend=True
            )
    
    def _get_trajectory_trace(self, up_to_idx: int = None):
        """Get trajectory as Plotly trace.
        
        Args:
            up_to_idx: If provided, only include states up to this index.
        """
        positions = [s.position for s in (self.states[:up_to_idx + 1] if up_to_idx is not None else self.states)]
        
        if self.ndim == 3:
            traj = np.array([[p.i, p.j, p.k] for p in positions])
            return go.Scatter3d(
                x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                mode='lines+markers',
                line=dict(color='royalblue', width=self.trajectory_width * 2.5),
                marker=dict(size=self.trajectory_width * 3, color='steelblue', opacity=0.7),
                name='Trajectory', showlegend=True, opacity=0.8
            )
        else:
            traj = np.array([[p.i, p.j] for p in positions])
            return go.Scatter(
                x=traj[:, 0], y=traj[:, 1],
                mode='lines+markers',
                line=dict(color='royalblue', width=self.trajectory_width),
                marker=dict(size=self.trajectory_width * 1.5, color='steelblue', opacity=0.7),
                name='Trajectory', showlegend=True, opacity=0.8
            )
    
    def _get_actor_trace(self, state: NavigationArenaState):
        """Get actor marker as Plotly trace."""
        pos = state.position
        if self.ndim == 3:
            return go.Scatter3d(
                x=[pos.i], y=[pos.j], z=[pos.k],
                mode='markers',
                marker=dict(size=self.actor_size * 0.8, color='red',
                            symbol='diamond', line=dict(color='darkred', width=2.5)),
                name='Actor', showlegend=True
            )
        else:
            return go.Scatter(
                x=[pos.i], y=[pos.j],
                mode='markers',
                marker=dict(size=self.actor_size, color='red',
                            symbol='diamond', line=dict(color='darkred', width=2)),
                name='Actor', showlegend=True
            )
    
    def _get_animated_layout(self) -> go.Layout:
        """Get layout for animated figure (used by html exporter)."""
        if self.ndim == 3:
            return get_animated_layout_3d(
                self.config, self.width, self.height, self.camera_eye
            )
        else:
            return get_animated_layout_2d(self.config, self.width, self.height)
