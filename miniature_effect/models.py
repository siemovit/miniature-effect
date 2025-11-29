from dataclasses import dataclass


@dataclass
class MiniatureSettings:
    line1_height: float = 0.25
    margin1_height: float = 0.35
    margin2_height: float = 0.65
    line2_height: float = 0.75


def clamp_settings(settings: "MiniatureSettings") -> "MiniatureSettings":
    """Ensure guide lines remain ordered from top to bottom."""
    ordered = sorted(
        [
            ("line1_height", settings.line1_height),
            ("margin1_height", settings.margin1_height),
            ("margin2_height", settings.margin2_height),
            ("line2_height", settings.line2_height),
        ],
        key=lambda item: item[1],
    )
    values = {name: value for name, value in ordered}
    return MiniatureSettings(
        line1_height=values["line1_height"],
        margin1_height=values["margin1_height"],
        margin2_height=values["margin2_height"],
        line2_height=values["line2_height"],
    )
