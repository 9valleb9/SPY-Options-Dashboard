# 📊 SPY 0DTE Options Analysis Dashboard

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

A professional web-based dashboard for analyzing SPY 0DTE (zero days to expiration) options trading signals. This application provides real-time market analysis, technical indicators, and trading alerts specifically designed for intraday options scalping strategies.

##  Features

###  **Market Analysis**
- **Real-time Data**: Live SPY market data with configurable intervals (1m, 5m, 15m, 30m, 1h)
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, Stochastic, OBV
- **Cross-Asset Analysis**: VIX, QQQ, IWM, TLT correlations and divergence signals
- **Market Regime Detection**: Volatility regimes, credit spreads, sector rotation

###  **Trading Signals**
- **15+ Scalping Signals**: Momentum, mean reversion, and volatility-based signals
- **Time-based Alerts**: Opening hour, power hour, and final 30-minute warnings
- **Composite Strategies**: Multi-signal alignment for high-conviction trades
- **Statistical Validation**: Win rates, Sharpe ratios, and significance testing

###  **Interactive Dashboard**
- **Responsive Design**: Bootstrap-powered interface for desktop and mobile
- **Real-time Charts**: Plotly visualizations with zoom, pan, and export
- **Performance Analytics**: Signal backtesting and performance tracking
- **Alert System**: Priority-based trading alerts with actionable recommendations

###  **Export & Integration**
- **Excel Export**: Complete analysis data in spreadsheet format
- **API Endpoints**: RESTful API for integration with other systems
- **Copy Functions**: Quick copy of signals and alerts for trading platforms

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Internet connection for market data

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/spy-0dte-dashboard.git
   cd spy-0dte-dashboard
   ```

2. **Set up virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Activate virtual environment
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python run.py
   ```

5. **Access the dashboard**
   ```
   Open your browser and navigate to: http://localhost:5000
   ```

## 📁 Project Structure

```
spy-0dte-dashboard/
├── app.py                          # Main Flask application
├── spy_0dte_options_analysis.py    # Core analysis framework
├── run.py                          # Application runner script
├── requirements.txt                # Python dependencies
├── templates/
│   └── index.html                  # Main dashboard template
├── static/                         # Static assets (auto-created)
├── screenshots/                    # Dashboard screenshots
├── docs/                          # Additional documentation
├── tests/                         # Unit tests
├── CONTRIBUTING.md                # Contribution guidelines
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## 🎛️ Usage

### Configuration Options

#### **Usage Patterns**
- **Full Analysis**: 30 days of 5-minute data for comprehensive analysis
- **Quick Check**: 5 days of 5-minute data for faster processing  
- **Intraday Focus**: 1 day of 1-minute data for current session
- **Performance Review**: 30 days for historical performance analysis
- **Custom**: User-defined period and interval settings

#### **Display Options**
- Toggle performance charts, market overview, and signal heatmaps
- Enable auto-refresh for real-time updates (5-minute intervals)
- Customize alert priorities and notification settings

### Trading Signal Categories

#### ** Momentum Signals**
- **MACD Crossovers**: Bullish/bearish momentum shifts
- **Volatility Expansion**: Short-term vol exceeding long-term
- **Breakout Signals**: Price action beyond key levels

#### ** Mean Reversion Signals**  
- **RSI Extremes**: Overbought (>70) and oversold (<30) conditions
- **Bollinger Band Bounces**: Price rejection from upper/lower bands
- **VIX Spikes**: Extreme fear levels indicating potential reversals

#### **Time-based Opportunities**
- **Opening Hour** (9:30-10:30 AM): High volatility gap trading
- **Power Hour** (3:00-4:00 PM): Institutional activity period
- **Final 30 Minutes** (3:30-4:00 PM): 0DTE gamma effects warning

#### **Composite Strategies**
- **Scalp Long**: Multiple bullish signals aligned (≥2 confirmations)
- **Scalp Short**: Multiple bearish signals aligned (≥2 confirmations)
- **Signal Strength Gauge**: Overall market signal intensity meter

## API Documentation

### Core Endpoints

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/` | GET | Main dashboard interface |
| `/api/load_data` | GET | Load market data with parameters |
| `/api/alerts` | GET | Current trading alerts |
| `/api/active_signals` | GET | Active signals and strength |

### Chart Endpoints

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/api/charts/performance` | GET | Signal performance analysis |
| `/api/charts/heatmap` | GET | Signal activity heatmap |
| `/api/charts/market_overview` | GET | SPY price, volume, RSI |
| `/api/charts/signal_gauge` | GET | Signal strength gauge |

### Data Endpoints

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/api/performance_table` | GET | Performance data table |
| `/api/performance_stats` | GET | Summary statistics |
| `/api/export/excel` | GET | Export analysis to Excel |
| `/api/summary` | GET | Daily summary report |

### Example API Usage

```python
import requests

# Load 5 days of 5-minute data
response = requests.get('http://localhost:5000/api/load_data?period=5d&interval=5m')
data = response.json()

# Get current alerts
alerts = requests.get('http://localhost:5000/api/alerts').json()

# Get active signals
signals = requests.get('http://localhost:5000/api/active_signals').json()
```

##  Configuration

### Environment Variables

```bash
# Flask Configuration
export FLASK_APP=app.py
export FLASK_ENV=development  # or production
export SECRET_KEY=your-secure-secret-key

# Optional: Custom settings
export DATA_CACHE_TIMEOUT=300  # 5 minutes
export AUTO_REFRESH_INTERVAL=300000  # 5 minutes in milliseconds
```

### Custom Settings

Edit `app.py` to modify default settings:

```python
# Data cache timeout (seconds)
CACHE_TIMEOUT = 300

# Default analysis parameters
DEFAULT_PERIOD = '30d'
DEFAULT_INTERVAL = '5m'

# Chart update intervals
AUTO_REFRESH_INTERVAL = 300000  # milliseconds
```

## Screenshots

### Main Dashboard

### Signal Analysis

### Performance Charts

### Trading Alerts

## Deployment

### Development
```bash
python run.py
# Access at http://localhost:5000
```

### Production with Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

```bash
# Build and run
docker build -t spy-dashboard .
docker run -p 5000:5000 spy-dashboard
```

### Cloud Deployment

#### Heroku
```bash
# Create Procfile
echo "web: gunicorn app:app" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

#### AWS Elastic Beanstalk
```bash
# Install EB CLI
pip install awsebcli

# Initialize and deploy
eb init
eb create spy-dashboard-env
eb deploy
```

## Security Considerations

### Production Checklist
- [ ] Change default secret key in `app.py`
- [ ] Enable HTTPS/SSL certificates
- [ ] Implement rate limiting for API endpoints
- [ ] Add input validation and sanitization
- [ ] Set up proper error logging
- [ ] Configure firewall rules
- [ ] Regular dependency updates

### Environment Security
```python
# Use environment variables for sensitive data
import os
app.secret_key = os.environ.get('SECRET_KEY', 'fallback-key')

# Rate limiting example
from flask_limiter import Limiter
limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/api/load_data')
@limiter.limit("10 per minute")
def load_data():
    # API logic here
```

##  Contributing

We welcome contributions! 

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes and add tests
4. Ensure all tests pass
5. Submit a pull request

### Code Standards
- **Python**: Follow PEP 8 style guidelines
- **JavaScript**: Use ES6+ features and proper formatting
- **HTML/CSS**: Follow Bootstrap conventions
- **Documentation**: Update README and docstrings

### Issue Reporting
Please use the [GitHub Issues](https://github.com/yourusername/spy-0dte-dashboard/issues) page to report bugs or request features.

## Documentation

### Additional Resources
- [Technical Analysis Guide](docs/technical-analysis.md)
- [API Reference](docs/api-reference.md)
- [Trading Strategies](docs/trading-strategies.md)
- [Troubleshooting Guide](docs/troubleshooting.md)

### External Documentation
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Plotly Python Documentation](https://plotly.com/python/)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [Bootstrap Documentation](https://getbootstrap.com/docs/)

## Educational Resources

### Options Trading Basics
- Understanding 0DTE options and their risks
- Time decay (theta) and its impact on pricing
- Greeks and their importance in options trading
- Risk management strategies for short-term trading

### Technical Analysis
- Moving averages and trend identification
- Oscillators (RSI, Stochastic) for momentum analysis
- Bollinger Bands for volatility and mean reversion
- Volume analysis and market structure

## ⚠Risk Disclaimer

**This software is for educational and informational purposes only. It is not financial advice.**

### Important Warnings
- **0DTE Options Trading** involves extremely high risk and may result in total loss
- **Past Performance** does not guarantee future results
- **Technical Signals** may generate false positives and should not be relied upon exclusively
- **Market Volatility** can cause rapid and significant losses

### Recommended Practices
- **Paper Trade First**: Test all strategies with virtual money before risking capital
- **Position Sizing**: Never risk more than you can afford to lose completely
- **Risk Management**: Always use stop losses and position limits
- **Professional Advice**: Consult qualified financial advisors for personalized guidance
- **Continuous Education**: Stay informed about market conditions and trading risks

### Legal Notice
The authors and contributors of this software assume no responsibility for any financial losses incurred through the use of this application. Users are solely responsible for their trading decisions and outcomes.

## Support

### Getting Help
- **Documentation**: Check the [docs](docs/) folder for detailed guides
- **Issues**: Report bugs on [GitHub Issues](https://github.com/yourusername/spy-0dte-dashboard/issues)
- **Discussions**: Join community discussions in [GitHub Discussions](https://github.com/yourusername/spy-0dte-dashboard/discussions)


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **yfinance**: Yahoo Finance data provider
- **Plotly**: Interactive charting library
- **Flask**: Web framework foundation
- **Bootstrap**: UI framework
- **Options Trading Community**: Inspiration and feedback

## Project Status

- **Current Version**: v1.0.0
- **Development Status**: Active
- **Last Updated**: January 2025
- **Python Support**: 3.8+
- **Tested Platforms**: Windows, macOS, Linux

---

### 🌟 Star this repository if you find it useful!

**Made with ❤️ for the options trading community**
