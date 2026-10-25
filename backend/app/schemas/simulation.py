"""Simulation state and control schemas."""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class SimulationConfigRequest(BaseModel):
    """User configuration for running simulation/experiment."""
    mode: str = Field(default="demo", description="'demo' (mock with controlled pacing) or 'real' (actual scientific pipeline)")
    model_name: str = Field(default="QuantumMeta-Sat (4Q)", description="Model architecture: 'Classical MLP', 'Classical MAML', 'Fixed VQC', 'QuantumMeta-Sat (4Q)', 'QuantumMeta-Sat (8Q)'")
    elevation_deg: float = Field(default=45.0, ge=0.0, le=90.0, description="Elevation angle in degrees")
    snr_db: float = Field(default=15.0, description="Signal-to-Noise ratio in dB")
    pilot_count: int = Field(default=10, description="Number of pilot symbols K (1, 5, 10, 20)")
    adaptation_steps: int = Field(default=5, description="Number of gradient adaptation steps (0, 1, 3, 5, 10)")
    environment: str = Field(default="average", description="Shadowing condition: 'infrequent_light', 'average', 'frequent_heavy'")
    doppler_freq_hz: Optional[float] = Field(default=None, description="Doppler frequency shift in Hz")
    speed: float = Field(default=1.0, description="Playback speed multiplier (0.5x, 1x, 2x)")
    step_by_step: bool = Field(default=False, description="Whether to pause after each stage")

class SimulationStateResponse(BaseModel):
    """Overall state summary of the Digital Twin simulation."""
    status: str = Field(default="ready", description="'ready', 'running', 'paused', 'completed', 'error'")
    current_stage: str = Field(default="idle", description="Current active stage")
    current_step: int = Field(default=0, description="Step index (0 to 10)")
    total_steps: int = Field(default=10, description="Total steps in pipeline")
    is_demo_mode: bool = Field(default=True, description="Whether currently running in Demo/Mock Mode")
    config: SimulationConfigRequest
    latest_event: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    system_health: Dict[str, Any] = Field(default_factory=dict)
