"""
Constants and definitions for cognitive state classifications and pedagogical directives.
"""

from typing import Dict

STATE_FOCUSED = "Focused / Attentive"
STATE_CONFUSED = "Confused / High Cognitive Load"
STATE_DISTRACTED = "Distracted / Looking Away"
STATE_DROWSY = "Drowsy / Fatigued"

STATE_SCORE_MAPPING: Dict[str, float] = {
    STATE_FOCUSED: 1.0,
    STATE_CONFUSED: 0.5,
    STATE_DISTRACTED: 0.0,
    STATE_DROWSY: 0.1,
}
