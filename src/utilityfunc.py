# src/utilityfunc.py

import os
import json
from datetime import datetime


def ensure_dir(path):
    
    if not os.path.exists(path):
        os.makedirs(path)


def log(message, prefix="INFO"):
    
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{prefix}] {time_str} - {message}")


def save_json(data, filepath):
   
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def pretty_print_portfolio(weights, performance):
    
    print("\n📊 Portfolio Allocation:")
    for asset, weight in weights.items():
        if weight > 0:
            print(f"  {asset}: {weight*100:.2f}%")

    exp_return, volatility, sharpe = performance
    print("\n📈 Performance:")
    print(f"  Expected Annual Return: {exp_return*100:.2f}%")
    print(f"  Annual Volatility: {volatility*100:.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")


def pretty_print_allocation(allocation, leftover):
    
    print("\n🧮 Discrete Allocation:")
    for ticker, shares in allocation.items():
        print(f"  {ticker}: {shares} share(s)")
    print(f"\n💰 Funds remaining (unallocated): ${leftover:.2f}")
