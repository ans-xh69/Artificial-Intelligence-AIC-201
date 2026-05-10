"""
main.py
Entry point for AIDRA — Adaptive Intelligent Disaster Response Agent.
"""

from environment import build_default_scenario, EventSimulator
from gui import launch_gui


def main():
    # Build scenario
    grid_map, victims, resources = build_default_scenario()

    # Dynamic event simulator
    event_sim = EventSimulator(grid_map, seed=7)

    # Launch GUI
    launch_gui(
        grid_map,
        victims,
        resources,
        event_sim
    )


if __name__ == "__main__":
    main()