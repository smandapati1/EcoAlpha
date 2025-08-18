🌱 EcoAlpha – ESG-Based Portfolio Optimization

EcoAlpha is a financial analytics platform that combines Environmental, Social, and Governance (ESG) factors with traditional quantitative portfolio optimization. The goal is to help investors align portfolios with sustainable investing principles while maintaining strong risk-adjusted returns.

✨ Features

Live ESG Data Integration – Fetches sustainability scores and company data using Yahoo Finance’s APIs.

Quantitative Portfolio Optimization – Uses PyPortfolioOpt and cvxpy to build efficient frontiers and optimize portfolios under ESG constraints.

Interactive Visualizations – Explore ESG scores and portfolio allocations in real time with an intuitive Streamlit dashboard.

Cloud Deployment – Hosted on Render for easy access and scalability.

⚙️ Tech Stack

Backend & Data: Python, Pandas, NumPy, yfinance

Optimization: PyPortfolioOpt, cvxpy

Frontend: Streamlit

Deployment: Render

🚀 Getting Started
Installation
git clone https://github.com/yourusername/EcoAlpha.git
cd EcoAlpha
pip install -r requirements.txt

Run Locally
streamlit run main.py

Deploy to Render

The app is fully configured to run on Render. Just push the repository and connect it with your Render account for deployment.

📊 Example Output
ESG-based portfolio optimization for: ['AAPL', 'MSFT', 'TSLA']

ESG Scores: {'AAPL': {'E': 0.73, 'S': 0.71, 'G': 0.65}, 
             'MSFT': {'E': 0.80, 'S': 0.76, 'G': 0.70}, 
             'TSLA': {'E': 0.60, 'S': 0.68, 'G': 0.55}}

Optimized Portfolio Weights: 
{'AAPL': 0.42, 'MSFT': 0.38, 'TSLA': 0.20}

📌 Use Cases

Sustainable investing research

ESG-focused portfolio construction

Educational projects for financial engineering and data science

🤝 Contributing

Contributions are welcome! Please open issues and submit pull requests to improve data integration, optimization strategies, or visualization.
