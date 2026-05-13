AIDRA: Adaptive Intelligent Disaster Response Agent
AIDRA is a hybrid AI disaster-response simulator that combines search, constraint satisfaction, machine learning, and fuzzy reasoning to plan rescue operations in a dynamic emergency grid.

Overview
This project was built for a Complex Computing Problem (CCP) assignment. It demonstrates a complete AI system for disaster response planning with:


route planning using multiple search algorithms

resource allocation with CSP

KTAS-based victim severity prediction from a real triage dataset

fuzzy logic for uncertainty handling

dynamic replanning when roads or hazards change

KPI tracking and visual reporting


Dataset
The ML models are trained on the cleaned triage dataset:


dataset_clean.csv


The raw file is kept for reference:


data.csv


Training features
The starter feature set used for KTAS prediction is:


Age

Injury

NRS_pain

SBP

DBP

HR

RR

BT


Target label
The project predicts KTAS severity and maps it to the simulator classes:


1, 2 -> critical

3, 4 -> moderate

5 -> minor


System Workflow
During a simulation run, AIDRA:

samples 5 real rows from dataset_clean.csv
assigns their KTAS-based criticality to 5 victims
places victims on different cells of the disaster grid
trains or reuses the ML models
prioritizes victims using the learned urgency score
allocates limited rescue resources using CSP
plans routes using search algorithms
replans if roads block or hazards change
drops victims at the nearest medical center
exports figures and KPI reports

Main Modules

main.py - launches the GUI

gui.py - interactive simulator and training controls

environment.py - grid world, victims, resources, and dynamic events

ml_model.py - KTAS dataset loading, KNN, and Naive Bayes

search.py - BFS, DFS, Greedy Best-First, A*, Hill Climbing

csp.py - resource allocation via CSP

uncertainty.py - fuzzy logic risk estimation

metrics.py - KPI tracking and CSV export

visualizer.py - figure generation

agent.py - command-line style integrated agent workflow

make_presentation.py - generates the slide deck


How to Run
Make sure dataset_clean.csv is in the project folder, then run:

bash



python main.py



In the GUI:


click TRAIN ML to train the KTAS models

click RUN to start the simulation

click PDFs to generate the full figure set

click Plot Graph to generate the GUI comparison chart


Generated Outputs
The visualizer saves both .png and .pdf versions in figures/.

Common outputs include:


grid_initial

grid_final

search_comparison

cm_kNN_test

cm_NB_test

ml_comparison

ml_survival_scatter

rescue_timeline

kpi_dashboard

fuzzy_surface

risk_heatmap


Requirements
The project uses standard Python libraries plus:


numpy

matplotlib

Pillow


Notes:


tkinter is used for the GUI and is included with most Python installations.

pandas is not required in the current version.


Presentation Deck
The repository includes a generated slide deck:


AIDRA_Presentation.pptx


You can regenerate it with:

bash



python make_presentation.py



Project Structure
Add to chat
AIDRA/
├── agent.py
├── csp.py
├── data.csv
├── dataset_clean.csv
├── environment.py
├── figures/
├── gui.py
├── logger.py
├── main.py
├── make_presentation.py
├── metrics.py
├── ml_model.py
├── search.py
├── uncertainty.py
├── visualizer.py
└── README.md



Notes

Victims are sampled from the real dataset on each simulation run.

Victim positions change between runs, so the scenario is not static.

Medical centers are fixed at opposite sides of the grid.

Ambulances carry at most 2 victims before returning to a medical center.

The project keeps the hybrid AI structure required by the CCP: search, CSP, ML, uncertainty handling, and dynamic replanning.