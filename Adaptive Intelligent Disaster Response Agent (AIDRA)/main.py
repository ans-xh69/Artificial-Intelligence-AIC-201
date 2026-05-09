"""
main.py
Entry point for AIDRA — Adaptive Intelligent Disaster Response Agent.
Run: python main.py
"""
 
from environment import build_default_scenario, EventSimulator
from agent import AIDRAAgent
 
 
def main():
    # Build the scenario defined in the CCP
    grid_map, victims, resources = build_default_scenario()
 
    # Set up dynamic event simulator (seed for reproducibility)
    event_sim = EventSimulator(grid_map, seed=7)
 
    # Instantiate and run the agent
    agent = AIDRAAgent(
        grid_map=grid_map,
        victims=victims,
        resources=resources,
        event_sim=event_sim,
    )
    agent.run()
 
 
if __name__ == "__main__":
    main()