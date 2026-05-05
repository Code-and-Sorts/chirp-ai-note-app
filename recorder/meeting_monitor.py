import threading
import time
from collections.abc import Callable
from datetime import datetime

from config.settings import MonitoringConfig
from utils.time_utils import should_warn_user


class MeetingMonitor:
    def __init__(
        self,
        config: MonitoringConfig,
        start_time: datetime,
        warning_callback: Callable[[int], None],
        should_stop_callback: Callable[[], bool],
    ):
        self.config = config
        self.start_time = start_time
        self.warning_callback = warning_callback
        self.should_stop_callback = should_stop_callback

        self.is_monitoring = False
        self.monitor_thread = None
        self.last_warning = None

    def start(self):
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)

    def _monitor_loop(self):
        while self.is_monitoring:
            try:
                elapsed_minutes = int(
                    (datetime.now() - self.start_time).total_seconds() / 60
                )

                if should_warn_user(
                    self.start_time,
                    self.config.warning_minutes,
                    self.last_warning,
                    self.config.warning_interval,
                ):
                    self.warning_callback(elapsed_minutes)
                    self.last_warning = datetime.now()

                if self.should_stop_callback():
                    from utils.popup_manager import PopupManager

                    popup_manager = PopupManager()

                    max_hours = self.config.max_recording_hours
                    popup_manager.show_error(
                        f"Recording stopped automatically after {max_hours} hours for safety"
                    )
                    break

                time.sleep(60)

            except Exception as exc:  # noqa: BLE001 - monitor loop must survive callback/popup errors
                import logging

                logging.getLogger(__name__).debug("Meeting monitor loop error: %s", exc)
                time.sleep(60)
                continue

    def get_elapsed_time(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()

    def get_elapsed_minutes(self) -> int:
        return int(self.get_elapsed_time() / 60)

    def is_over_warning_threshold(self) -> bool:
        return self.get_elapsed_minutes() >= self.config.warning_minutes
