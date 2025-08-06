import matplotlib.pyplot as plt

def plot_portfolio_weights(weights, title="Optimized Portfolio Allocation"):
    """
    Plots a pie chart of portfolio weights.
    """
    labels = list(weights.keys())
    values = list(weights.values())

    plt.figure(figsize=(6,6))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
