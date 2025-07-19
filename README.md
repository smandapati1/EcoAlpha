EcoAlpha is a command-line tool that combines natural language processing (NLP) and quantitative finance to build ESG-aware investment portfolios.

Instead of relying on proprietary or opaque ESG vendor scores, ESGOptima extracts ESG sentiment directly from unstructured financial text—like 10-K filings or sustainability reports—using transformer-based NLP models. These scores are then integrated into a portfolio optimization framework to help investors allocate capital in a way that balances returns, risk, and environmental/social responsibility.

🔍 Key Features
🔎 ESG Signal Extraction
Uses FinBERT (or similar models) to analyze financial documents for sentiment related to Environmental, Social, and Governance factors.

📊 ESG Scoring Engine
Aggregates sentiment into interpretable, numerical ESG scores for each company.

💰 Portfolio Optimization
Incorporates ESG scores into a risk-return optimization model using techniques like mean-variance optimization (Markowitz) or the Black-Litterman model.

🖥 Command-Line Interface (CLI)
Allows users to run the full pipeline with customizable inputs:

