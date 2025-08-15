#!/bin/bash

# Install or upgrade specific Python packages
pip install --upgrade yfinance pandas numpy matplotlib seaborn scikit-learn

# Upgrade all installed Python packages
pip freeze | cut -d= -f1 | xargs pip install --upgrade
