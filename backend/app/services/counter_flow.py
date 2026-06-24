"""CounterFlow — PLC flow strategy for Component Counter mode (placeholder).

TODO: Implement when Counter mode handshake is designed.
For now this is a stub that logs warnings and does nothing.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from backend.app.services.plc_flow_strategy import PlcFlowStrategy

if TYPE_CHECKING:
    from backend.app.services.plc_adapter import PlcAdapter
    from backend.app.models.machine_settings import CounterModeConfig

logger = logging.getLogger(__name__)


class CounterFlow(PlcFlowStrategy):
    """Component Counter mode — PLACEHOLDER.

    CounterFlow is activated when template validator_mode == "component_count".
    The actual handshake (sensor → clamp → count → accept/reject) is not yet
    designed. This stub logs warnings so the system doesn't crash.
    """

    def __init__(
        self,
        adapter: PlcAdapter,
        settings: CounterModeConfig,
        num_channels: int = 4,
    ):
        super().__init__(adapter, settings, num_channels)
        self._counter_settings = settings
        logger.warning(
            "[counter-flow] CounterFlow is a STUB — no PLC actions will be performed. "
            "Implement the counter handshake before using this mode in production."
        )

    @property
    def flow_name(self) -> str:
        return "counter-flow"

    def get_input_release_address(self) -> int:
        return self._counter_settings.input_release_address

    def get_input_template_address(self) -> int:
        return self._counter_settings.input_template_address

    def get_input_clamp_engaged_address(self) -> int:
        return self._counter_settings.input_clamp_engaged_address

    def on_part_ready(self, worker) -> None:
        logger.warning("[counter-flow] on_part_ready — STUB, no action")

    def on_accept(self, worker) -> None:
        logger.warning("[counter-flow] on_accept — STUB, no action")

    def on_reject(self, worker) -> None:
        logger.warning("[counter-flow] on_reject — STUB, no action")

    def handle_input_release(self, worker, inputs: list[bool]) -> bool:
        return False

    def handle_input_template_cycle(self, worker, inputs: list[bool]) -> bool:
        return False

    def handle_clamp_feedback(self, worker, inputs: list[bool]) -> None:
        pass
