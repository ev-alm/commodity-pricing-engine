 Commodity Valuation & Energy Storage Arbitrage Pricing Engine

An algorithmic time-series asset valuation engine designed to model commodity price trajectories and evaluate physical storage contracts for seasonal arbitrage opportunities.

System Architecture & Features
* Time-Series Regression Engine:** Models natural gas spot prices using multi-variable linear trend analysis alongside seasonal monthly dummy variables.
* Operational Simulation Framework:** Models continuous injection and withdrawal cycles under strict operational constraints (maximum volume storage capacities, physical extraction rates).
* Friction and Overhead Matrix:** Factors in transactional transport overhead, injection/withdrawal execution fees, and monthly storage maintenance costs.
* Arbitrage Optimization:** Evaluates financial contract worth by identifying profitable spreads between low-cost injection periods (summer) and high-cost withdrawal windows (winter).

Tech Stack
* Python
* NumPy, Pandas
* Scikit-learn
