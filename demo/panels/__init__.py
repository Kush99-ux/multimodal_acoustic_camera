"""demo.panels package

Exporting all 10 presentation mode panels and the persistent telemetry sidebar.
"""

from demo.panels.telemetry_panel import TelemetrySidebar
from demo.panels.multimodal_panel import MultimodalPanel
from demo.panels.acoustic_panel import AcousticPanel
from demo.panels.vision_panel import VisionPanel
from demo.panels.horizontal_panel import HorizontalTestPanel
from demo.panels.vertical_panel import VerticalTestPanel
from demo.panels.sync_panel import SynchronizationPanel
from demo.panels.diagnostics_panel import DiagnosticsPanel
from demo.panels.validation_panel import ValidationPanel
from demo.panels.architecture_panel import ArchitecturePanel
from demo.panels.roadmap_panel import RoadmapPanel

__all__ = [
    "TelemetrySidebar",
    "MultimodalPanel",
    "AcousticPanel",
    "VisionPanel",
    "HorizontalTestPanel",
    "VerticalTestPanel",
    "SynchronizationPanel",
    "DiagnosticsPanel",
    "ValidationPanel",
    "ArchitecturePanel",
    "RoadmapPanel",
]
