# ============================================
# SEISMIC TRIGGER MODULE — CLEAN VERSION
# ============================================

from enum import Enum

class SeismicTriggerType(Enum):
    SCALE = "scale_change"
    STRUCTURE = "wave_structure_change"
    MODEL_CONFLICT = "model_conflict"
    CONTINUITY = "signal_discontinuity"
    NONE = "none"

class SeismicTriggerResult:
    def __init__(self, triggered=False, trigger_type=SeismicTriggerType.NONE,
                 location=None, message=""):
        self.triggered = triggered
        self.trigger_type = trigger_type
        self.location = location
        self.message = message

    def as_dict(self):
        return {
            "triggered": self.triggered,
            "type": self.trigger_type.value,
            "location": self.location,
            "message": self.message
        }

class SeismicTriggerModule:
    """
    MODULE: Seismic Trigger
    Detects critical changes in seismic signal behavior.
    """

    def __init__(self):
        self.last_result = SeismicTriggerResult()

    def analyze(self, seismic_steps):
        """
        Main entry point.
        seismic_steps: list of dicts describing each processed segment of the signal.
        """
        self.last_result = self._detect(seismic_steps)
        return self.last_result

    def _detect(self, steps):
        for step_id, step in enumerate(steps):

            # --- TRIGGER: SCALE CHANGE ---
            if step.get("amplitude_local") and step.get("amplitude_global"):
                if step["amplitude_global"] > step["amplitude_local"] * 10:
                    return SeismicTriggerResult(
                        True,
                        SeismicTriggerType.SCALE,
                        step_id,
                        "Seismic amplitude changed scale significantly."
                    )

            # --- TRIGGER: WAVE STRUCTURE CHANGE ---
            if step.get("wave_type_prev") and step.get("wave_type_curr"):
                if step["wave_type_prev"] != step["wave_type_curr"]:
                    return SeismicTriggerResult(
                        True,
                        SeismicTriggerType.STRUCTURE,
                        step_id,
                        "Wave structure changed (P → S or S → surface)."
                    )

            # --- TRIGGER: MODEL CONFLICT ---
            if step.get("model_prediction") and step.get("measured_value"):
                if abs(step["model_prediction"] - step["measured_value"]) > step.get("tolerance", 0.0):
                    return SeismicTriggerResult(
                        True,
                        SeismicTriggerType.MODEL_CONFLICT,
                        step_id,
                        "Model prediction does not match measured seismic data."
                    )

            # --- TRIGGER: SIGNAL DISCONTINUITY ---
            if step.get("gap_detected"):
                return SeismicTriggerResult(
                    True,
                    SeismicTriggerType.CONTINUITY,
                    step_id,
                    "Signal continuity break detected."
                )

        return SeismicTriggerResult(False, SeismicTriggerType.NONE, None, "No seismic trigger detected.")

    def get_last(self):
        return self.last_result
