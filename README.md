# Quantitative Backtesting Platform

A web-based platform for backtesting various trading strategies on stocks and cryptocurrencies. The platform provides an interactive interface for testing and visualizing the performance of different trading strategies.

## Features

- **Multiple Trading Strategies:**
  - Simple Moving Average (SMA) Strategy
  - Momentum (Mom) Strategy
  - Mean Reversion (MR) Strategy
  - Linear Regression (LR) Strategy
  - Scikit-Learn (ML) Strategy
  - Deep Learning (DL) Strategy

- **Asset Classes:**
  - Stocks 
  - Cryptocurrencies

- **Strategy Features:**
  - Dynamic date range selection
  - Customizable parameters for each strategy
  - Transaction costs consideration
  - Adjustable initial capital
  - Performance comparison with buy-and-hold strategy and S&P500

- **Visualization:**
  - Interactive price charts
  - Performance metrics
  - Returns distribution
  - Strategy-specific indicators

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd QB
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:8050/
```

3. Select a strategy from the navigation menu and configure:
   - Choose an asset (stock or cryptocurrency)
   - Set the date range for testing
   - Adjust strategy-specific parameters
   - Set transaction costs and initial capital
   - View the results in real-time

## Strategy Descriptions

### SMA Strategy
Based on golden cross and death cross patterns. Generates buy/sell signals when a short-term SMA crosses a long-term SMA. Features parameter optimization to find the most profitable SMA combinations.

### Momentum Strategy
Tracks the momentum of price movements to identify potential trading opportunities. Uses historical price data to determine the strength and direction of price trends.

### Mean Reversion Strategy
Implements mean reversion trading based on the hypothesis that asset prices tend to return to their mean levels after significant deviations. Uses threshold values to generate trading signals.

### Linear Regression Strategy
Uses linear regression to predict price movement direction. Features separate training and testing periods with adjustable lag parameters for robust predictions.

### Scikit-Learn Strategy
Implements machine learning-based trading using scikit-learn. Features:
- Both regression and logistic regression models
- Separate training and testing periods
- Adjustable lag parameters
- Real-time performance monitoring

### Deep Learning Strategy
Employs a neural network to predict price movement direction. Includes:
- Separate training and testing periods
- Real-time visualization of model performance
- Adjustable network parameters
- Loss and accuracy monitoring

## Project Structure

- `app.py` - Main application file
- `assets/` - Contains backtester implementations and static files
- `pages/` - Individual strategy page implementations
- `requirements.txt` - Project dependencies
- `Procfile` - Deployment configuration
- `runtime.txt` - Python runtime specification

## Dependencies

Main dependencies include:
- Dash
- Plotly
- Pandas
- NumPy
- TensorFlow
- yfinance
- scikit-learn

For a complete list, see `requirements.txt`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data provided by Yahoo Finance
- Built with Dash by Plotly
- Inspired by "Python for Algorithmic Trading" by Dr. Yves J. Hilpisch 
