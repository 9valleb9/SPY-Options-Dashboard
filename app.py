from flask import Flask, render_template, request, jsonify, send_file, session
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly
import json
import datetime
import time
from io import BytesIO
import base64
from functools import wraps
import os
import logging

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Import your analysis framework
    from spy_0dte_options_analysis import MarketStructureAnalyzer, generate_trading_alerts, create_signal_documentation
    ANALYSIS_AVAILABLE = True
    logger.info("Analysis framework loaded successfully")
except ImportError as e:
    logger.warning(f"Analysis framework not available: {e}")
    ANALYSIS_AVAILABLE = False

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'
app.json_encoder = NumpyEncoder  # Use custom encoder for JSON responses

# Global cache for data
data_cache = {
    'analyzer': None,
    'signals': None,
    'performance': None,
    'alerts': None,
    'last_update': None,
    'dashboard_data': None,
    'mock_mode': False
}

def create_mock_data(period='30d', interval='5m'):
    """Create realistic mock data for testing"""
    logger.info(f"Creating mock data for {period}, {interval}")
    
    # Calculate number of periods
    period_map = {'1d': 1, '5d': 5, '10d': 10, '30d': 30, '60d': 60}
    interval_map = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60}
    
    days = period_map.get(period, 30)
    minutes = interval_map.get(interval, 5)
    
    # Create date range (only business days, market hours)
    end_date = datetime.datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
    start_date = end_date - datetime.timedelta(days=days)
    
    # Generate market hours only (9:30 AM - 4:00 PM)
    dates = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Weekdays only
            # Add market hours
            market_open = current_date.replace(hour=9, minute=30)
            market_close = current_date.replace(hour=16, minute=0)
            
            current_time = market_open
            while current_time <= market_close:
                dates.append(current_time)
                current_time += datetime.timedelta(minutes=minutes)
        
        current_date += datetime.timedelta(days=1)
    
    dates = dates[-1000:]  # Limit to last 1000 points for performance
    
    # Generate realistic SPY price data
    base_price = 445.0
    prices = []
    current_price = base_price
    
    for i, date in enumerate(dates):
        # Add realistic price movement with volatility
        hour = date.hour
        minute = date.minute
        
        # Higher volatility during opening/closing hours
        if (hour == 9 and minute < 60) or hour >= 15:
            volatility = 0.002  # 0.2%
        else:
            volatility = 0.001  # 0.1%
        
        # Random walk with slight upward bias
        change = np.random.normal(0.0001, volatility)
        current_price *= (1 + change)
        
        # Create OHLC data
        high = current_price * (1 + abs(np.random.normal(0, 0.0005)))
        low = current_price * (1 - abs(np.random.normal(0, 0.0005)))
        open_price = current_price + np.random.normal(0, current_price * 0.0002)
        
        prices.append({
            'Open': open_price,
            'High': max(open_price, high, current_price),
            'Low': min(open_price, low, current_price),
            'Close': current_price,
            'Volume': np.random.randint(1000000, 5000000)
        })
    
    # Create DataFrame
    mock_data = pd.DataFrame(prices, index=dates)
    logger.info(f"Created mock SPY data: {len(mock_data)} rows")
    
    return mock_data

def create_mock_analyzer(period='30d', interval='5m'):
    """Create a mock analyzer with realistic data"""
    
    class MockAnalyzer:
        def __init__(self):
            self.data = {}
            self.features = {}
            
        def fetch_data(self, period, interval):
            logger.info(f"Mock: Fetching data for {period}, {interval}")
            
            # Create mock SPY data
            spy_data = create_mock_data(period, interval)
            self.data['SPY'] = spy_data
            
            # Create mock data for other tickers (simplified)
            for ticker in ['^VIX', 'QQQ', 'IWM']:
                mock_ticker_data = spy_data.copy()
                # Modify slightly for different assets
                if ticker == '^VIX':
                    mock_ticker_data *= 0.05  # VIX typically 15-25
                    mock_ticker_data += 15
                elif ticker == 'QQQ':
                    mock_ticker_data *= 0.8  # QQQ typically lower than SPY
                else:
                    mock_ticker_data *= 1.1  # IWM varies
                
                self.data[ticker] = mock_ticker_data
            
            logger.info(f"Mock: Created data for {len(self.data)} tickers")
        
        def calculate_market_structure_features(self):
            logger.info("Mock: Calculating market structure features")
            
            if 'SPY' not in self.data:
                return
            
            spy_data = self.data['SPY']
            self.features['SPY'] = {}
            
            # Calculate basic features
            self.features['SPY']['returns'] = spy_data['Close'].pct_change()
            
            # RSI
            delta = spy_data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            self.features['SPY']['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = spy_data['Close'].ewm(span=12).mean()
            ema26 = spy_data['Close'].ewm(span=26).mean()
            self.features['SPY']['macd'] = ema12 - ema26
            self.features['SPY']['macd_signal'] = self.features['SPY']['macd'].ewm(span=9).mean()
            self.features['SPY']['macd_histogram'] = self.features['SPY']['macd'] - self.features['SPY']['macd_signal']
            
            # Bollinger Bands
            bb_window = 20
            bb_middle = spy_data['Close'].rolling(window=bb_window).mean()
            bb_std = spy_data['Close'].rolling(window=bb_window).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            self.features['SPY']['bb_width'] = ((bb_upper - bb_lower) / bb_middle.replace(0, np.nan)) * 100
            self.features['SPY']['bb_position'] = (spy_data['Close'] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
            
            # Volatility measures
            self.features['SPY']['volatility_5'] = self.features['SPY']['returns'].rolling(window=12).std()
            self.features['SPY']['volatility_15'] = self.features['SPY']['returns'].rolling(window=36).std()
            self.features['SPY']['volatility_30'] = self.features['SPY']['returns'].rolling(window=72).std()
            
            logger.info(f"Mock: Calculated {len(self.features['SPY'])} features")
        
        def calculate_scalping_signals(self):
            logger.info("Mock: Calculating scalping signals")
            
            if 'SPY' not in self.features:
                return pd.DataFrame()
            
            signals = pd.DataFrame(index=self.data['SPY'].index)
            spy_features = self.features['SPY']
            
            # RSI signals
            if 'rsi' in spy_features:
                rsi = spy_features['rsi']
                signals['rsi_oversold'] = (rsi < 30).astype(int)
                signals['rsi_overbought'] = (rsi > 70).astype(int)
            
            # MACD signals
            if 'macd_histogram' in spy_features:
                macd_hist = spy_features['macd_histogram']
                macd_bullish = (macd_hist > 0) & (macd_hist.shift(1) <= 0)
                macd_bearish = (macd_hist < 0) & (macd_hist.shift(1) >= 0)
                signals['macd_bullish'] = macd_bullish.astype(int)
                signals['macd_bearish'] = macd_bearish.astype(int)
            
            # Bollinger Band signals
            if 'bb_width' in spy_features and 'bb_position' in spy_features:
                bb_width = spy_features['bb_width']
                bb_pos = spy_features['bb_position']
                
                bb_compressed = bb_width < bb_width.rolling(window=48).quantile(0.2)
                signals['bb_compression'] = bb_compressed.astype(int)
                
                bb_mean_revert_bull = (bb_pos < 0.1) & (bb_pos.shift(1) >= 0.1)
                bb_mean_revert_bear = (bb_pos > 0.9) & (bb_pos.shift(1) <= 0.9)
                signals['bb_mean_revert_bull'] = bb_mean_revert_bull.astype(int)
                signals['bb_mean_revert_bear'] = bb_mean_revert_bear.astype(int)
            
            # Volatility signals
            if all(k in spy_features for k in ['volatility_5', 'volatility_15', 'volatility_30']):
                vol_5 = spy_features['volatility_5']
                vol_15 = spy_features['volatility_15']
                vol_30 = spy_features['volatility_30']
                
                vol_expansion = (vol_5 > vol_15) & (vol_15 > vol_30)
                vol_contraction = (vol_5 < vol_15) & (vol_15 < vol_30)
                signals['vol_expansion'] = vol_expansion.astype(int)
                signals['vol_contraction'] = vol_contraction.astype(int)
            
            # Time-based signals
            time_index = self.data['SPY'].index
            time_hour = time_index.hour
            time_minute = time_index.minute
            
            opening_condition = ((time_hour == 9) & (time_minute >= 30)) | ((time_hour == 10) & (time_minute < 30))
            power_condition = (time_hour == 15)
            final_condition = (time_hour == 15) & (time_minute >= 30)
            
            signals['opening_hour'] = opening_condition.astype(int)
            signals['power_hour'] = power_condition.astype(int)
            signals['final_30min'] = final_condition.astype(int)
            
            # Composite signals
            signals['total_signals'] = signals.sum(axis=1, numeric_only=True)
            
            # Scalp signals
            long_signals = ['rsi_oversold', 'bb_mean_revert_bull', 'macd_bullish']
            available_long = [col for col in long_signals if col in signals.columns]
            if len(available_long) >= 2:
                signals['scalp_long'] = (signals[available_long].sum(axis=1) >= 2).astype(int)
            else:
                signals['scalp_long'] = 0
            
            short_signals = ['rsi_overbought', 'bb_mean_revert_bear', 'macd_bearish']
            available_short = [col for col in short_signals if col in signals.columns]
            if len(available_short) >= 2:
                signals['scalp_short'] = (signals[available_short].sum(axis=1) >= 2).astype(int)
            else:
                signals['scalp_short'] = 0
            
            logger.info(f"Mock: Generated {len(signals.columns)} signal types")
            return signals
        
        def analyze_signal_performance(self):
            logger.info("Mock: Analyzing signal performance")
            
            # Create mock performance data
            signals = ['rsi_oversold', 'rsi_overbought', 'macd_bullish', 'macd_bearish', 
                      'bb_mean_revert_bull', 'bb_mean_revert_bear', 'scalp_long', 'scalp_short']
            
            performance_data = {}
            for signal in signals:
                for period in ['5min', '15min', '30min']:
                    key = f"{signal}_{period}"
                    performance_data[key] = {
                        'win_rate': np.random.uniform(0.45, 0.75),
                        'signal_sharpe': np.random.uniform(-0.5, 2.0),
                        'count': np.random.randint(20, 200),
                        'excess_return': np.random.uniform(-0.001, 0.002),
                        'signal_mean': np.random.uniform(-0.001, 0.002),
                        'baseline_mean': np.random.uniform(-0.0005, 0.0005)
                    }
            
            return pd.DataFrame(performance_data).T
        
        def create_dashboard_data(self):
            if 'SPY' in self.data:
                return self.data['SPY']
            return None
    
    def mock_generate_trading_alerts(analyzer):
        """Generate mock trading alerts"""
        current_time = datetime.datetime.now()
        hour = current_time.hour
        
        alerts = []
        
        # Time-based alerts
        if 9 <= hour <= 10:
            alerts.append({
                'priority': 'MEDIUM',
                'type': 'TIME_OPPORTUNITY',
                'message': 'OPENING HOUR - High volatility period active',
                'action': 'Gap fade/fill strategies, volatility plays - increased opportunity',
                'timeframe': 'Next 30-60 minutes',
                'confidence': '70%+'
            })
        
        if hour == 15:
            alerts.append({
                'priority': 'MEDIUM',
                'type': 'TIME_OPPORTUNITY',
                'message': 'POWER HOUR - Institutional activity period',
                'action': 'Trend continuation plays, momentum trades',
                'timeframe': 'Next 60 minutes',
                'confidence': '70%+'
            })
            
            if current_time.minute >= 30:
                alerts.append({
                    'priority': 'HIGH',
                    'type': 'GAMMA_RISK',
                    'message': 'FINAL 30 MINUTES - 0DTE gamma effects in play',
                    'action': 'EXTREME CAUTION - Quick scalps only, tight stops, gamma pin risk',
                    'timeframe': 'Until market close',
                    'confidence': 'High risk/reward'
                })
        
        # Random signal-based alerts
        signal_alerts = [
            {
                'priority': 'HIGH',
                'type': 'SCALP_LONG',
                'message': 'MULTIPLE BULLISH SIGNALS ALIGNED - High-conviction long setup',
                'action': 'Quick call spreads with tight stops - scalping opportunity',
                'timeframe': '5-15 minutes',
                'confidence': '85%+'
            },
            {
                'priority': 'HIGH',
                'type': 'MEAN_REVERSION',
                'message': 'BOLLINGER BAND BOUNCE - Strong mean reversion signal upward',
                'action': 'High-conviction call spreads targeting middle band',
                'timeframe': '5-20 minutes',
                'confidence': '80%+'
            },
            {
                'priority': 'MEDIUM',
                'type': 'MOMENTUM_BULLISH',
                'message': 'MACD BULLISH CROSSOVER - Momentum shifting upward',
                'action': 'Consider ATM/ITM call spreads for momentum continuation',
                'timeframe': '15-45 minutes',
                'confidence': '75%+'
            }
        ]
        
        # Add 1-2 random signal alerts
        import random
        alerts.extend(random.sample(signal_alerts, random.randint(1, 2)))
        
        return alerts
    
    def mock_create_signal_documentation():
        return {
            "MOMENTUM_SIGNALS": {
                "macd_bullish": "MACD histogram crosses positive - momentum shift up",
                "macd_bearish": "MACD histogram crosses negative - momentum shift down"
            },
            "MEAN_REVERSION": {
                "rsi_oversold": "RSI < 30 - oversold condition, potential bounce",
                "bb_mean_revert_bull": "Price bounces from lower Bollinger Band",
                "bb_mean_revert_bear": "Price rejects upper Bollinger Band"
            }
        }
    
    # Create and return mock analyzer
    analyzer = MockAnalyzer()
    return analyzer, mock_generate_trading_alerts, mock_create_signal_documentation

def cache_timeout(minutes=5):
    """Check if cache has expired"""
    if data_cache['last_update'] is None:
        return True
    return (datetime.datetime.now() - data_cache['last_update']).seconds > minutes * 60

def load_market_data(period='30d', interval='5m', force_mock=False):
    """Load and cache market data with fallback to mock data"""
    try:
        if ANALYSIS_AVAILABLE and not force_mock:
            logger.info("Attempting to load real market data")
            analyzer = MarketStructureAnalyzer()
            analyzer.fetch_data(period=period, interval=interval)
            
            # Check if we got any real data
            if hasattr(analyzer, 'data') and analyzer.data and 'SPY' in analyzer.data and not analyzer.data['SPY'].empty:
                logger.info("Real market data loaded successfully")
                analyzer.calculate_market_structure_features()
                signals = analyzer.calculate_scalping_signals()
                performance = analyzer.analyze_signal_performance()
                alerts = generate_trading_alerts(analyzer)
                dashboard_data = analyzer.create_dashboard_data()
                
                data_cache['mock_mode'] = False
                
            else:
                logger.warning("No real data available, switching to mock mode")
                raise Exception("No real market data available")
        else:
            raise Exception("Analysis framework not available or mock mode forced")
            
    except Exception as e:
        logger.warning(f"Real data failed ({e}), using mock data")
        
        # Use mock data
        analyzer, mock_alerts_func, mock_docs_func = create_mock_analyzer(period, interval)
        analyzer.fetch_data(period, interval)
        analyzer.calculate_market_structure_features()
        signals = analyzer.calculate_scalping_signals()
        performance = analyzer.analyze_signal_performance()
        alerts = mock_alerts_func(analyzer)
        dashboard_data = analyzer.create_dashboard_data()
        
        data_cache['mock_mode'] = True
    
    # Update cache
    data_cache['analyzer'] = analyzer
    data_cache['signals'] = signals
    data_cache['performance'] = performance
    data_cache['alerts'] = alerts
    data_cache['last_update'] = datetime.datetime.now()
    data_cache['dashboard_data'] = dashboard_data
    
    return True, f"Data loaded successfully ({'Mock Mode' if data_cache['mock_mode'] else 'Live Data'})"

# ... (keep all the chart creation functions from the original app.py) ...

def create_performance_chart(performance):
    """Create performance visualization"""
    if performance.empty:
        return None
    
    # Extract 5min performance data
    perf_5min = performance[performance.index.str.contains('5min')].copy()
    if perf_5min.empty:
        return None
        
    perf_5min['signal_name'] = perf_5min.index.str.replace('_5min', '')
    
    # Sort by win rate
    perf_5min = perf_5min.sort_values('win_rate', ascending=True)
    
    fig = go.Figure()
    
    # Win rate bars
    fig.add_trace(go.Bar(
        x=perf_5min['win_rate'],
        y=perf_5min['signal_name'],
        orientation='h',
        name='Win Rate',
        text=[f"{x:.1%}" for x in perf_5min['win_rate']],
        textposition='auto',
        marker_color=['#2ecc71' if x > 0.5 else '#e74c3c' for x in perf_5min['win_rate']]
    ))
    
    fig.update_layout(
        title="Signal Performance - Win Rates (5min horizon)",
        xaxis_title="Win Rate",
        yaxis_title="Signal",
        height=600,
        showlegend=False
    )
    
    # Add 50% line
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="50% (Random)")
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_signals_heatmap(signals):
    """Create signals activity heatmap"""
    if signals.empty:
        return None
    
    # Get recent signals (last 100 periods)
    recent_signals = signals.tail(100)
    
    # Create time-based grouping
    recent_signals['hour'] = recent_signals.index.hour
    recent_signals['minute_group'] = (recent_signals.index.minute // 15) * 15  # 15-min groups
    
    # Aggregate signals by time
    signal_cols = [col for col in recent_signals.columns if col not in ['total_signals', 'scalp_long', 'scalp_short', 'hour', 'minute_group']]
    
    if not signal_cols:
        return None
    
    heatmap_data = recent_signals.groupby(['hour', 'minute_group'])[signal_cols].sum()
    
    if heatmap_data.empty:
        return None
    
    fig = px.imshow(
        heatmap_data.T,
        labels=dict(x="Time (Hour:Minute)", y="Signal Type", color="Frequency"),
        title="Signal Activity Heatmap - Last 100 Periods",
        color_continuous_scale="YlOrRd"
    )
    
    # Update x-axis labels
    x_labels = [f"{h}:{m:02d}" for h, m in heatmap_data.index]
    fig.update_xaxes(tickvals=list(range(len(x_labels))), ticktext=x_labels[::5])  # Show every 5th label
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_market_overview(analyzer):
    """Create market overview charts with signal trigger points"""
    if not hasattr(analyzer, 'data') or 'SPY' not in analyzer.data:
        return None
    
    spy_data = analyzer.data['SPY']
    
    # Handle MultiIndex columns
    if isinstance(spy_data.columns, pd.MultiIndex):
        close_col = ('Close', 'SPY')
        volume_col = ('Volume', 'SPY')
    else:
        close_col = 'Close'
        volume_col = 'Volume'
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=['SPY Price Action with Signal Triggers', 'Volume', 'RSI with Signal Levels'],
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(x=spy_data.index, y=spy_data[close_col], 
                  name='SPY Close', line=dict(color='#2E86AB', width=2)),
        row=1, col=1
    )
    
    # Add signal trigger points if available
    if hasattr(analyzer, 'features') and 'SPY' in analyzer.features and 'signals' in analyzer.features['SPY']:
        signals = analyzer.features['SPY']['signals']
        spy_close = spy_data[close_col]
        
        # Add buy signals (green triangles up)
        buy_signals = ['rsi_oversold', 'bb_mean_revert_bull', 'macd_bullish', 'scalp_long']
        for signal_name in buy_signals:
            if signal_name in signals.columns:
                signal_points = signals[signals[signal_name] == 1]
                if not signal_points.empty:
                    # Get corresponding prices
                    signal_prices = spy_close.reindex(signal_points.index)
                    fig.add_trace(
                        go.Scatter(
                            x=signal_points.index,
                            y=signal_prices,
                            mode='markers',
                            name=f'Buy: {signal_name.replace("_", " ").title()}',
                            marker=dict(
                                symbol='triangle-up',
                                size=8,
                                color='#27AE60',
                                line=dict(width=1, color='white')
                            ),
                            showlegend=True
                        ),
                        row=1, col=1
                    )
        
        # Add sell signals (red triangles down)
        sell_signals = ['rsi_overbought', 'bb_mean_revert_bear', 'macd_bearish', 'scalp_short']
        for signal_name in sell_signals:
            if signal_name in signals.columns:
                signal_points = signals[signals[signal_name] == 1]
                if not signal_points.empty:
                    # Get corresponding prices
                    signal_prices = spy_close.reindex(signal_points.index)
                    fig.add_trace(
                        go.Scatter(
                            x=signal_points.index,
                            y=signal_prices,
                            mode='markers',
                            name=f'Sell: {signal_name.replace("_", " ").title()}',
                            marker=dict(
                                symbol='triangle-down',
                                size=8,
                                color='#E74C3C',
                                line=dict(width=1, color='white')
                            ),
                            showlegend=True
                        ),
                        row=1, col=1
                    )
        
        # Add high-conviction signals (larger markers)
        if 'scalp_long' in signals.columns:
            scalp_long_points = signals[signals['scalp_long'] == 1]
            if not scalp_long_points.empty:
                signal_prices = spy_close.reindex(scalp_long_points.index)
                fig.add_trace(
                    go.Scatter(
                        x=scalp_long_points.index,
                        y=signal_prices,
                        mode='markers',
                        name='HIGH CONVICTION LONG',
                        marker=dict(
                            symbol='star',
                            size=12,
                            color='#F39C12',
                            line=dict(width=2, color='white')
                        ),
                        showlegend=True
                    ),
                    row=1, col=1
                )
        
        if 'scalp_short' in signals.columns:
            scalp_short_points = signals[signals['scalp_short'] == 1]
            if not scalp_short_points.empty:
                signal_prices = spy_close.reindex(scalp_short_points.index)
                fig.add_trace(
                    go.Scatter(
                        x=scalp_short_points.index,
                        y=signal_prices,
                        mode='markers',
                        name='HIGH CONVICTION SHORT',
                        marker=dict(
                            symbol='star',
                            size=12,
                            color='#8E44AD',
                            line=dict(width=2, color='white')
                        ),
                        showlegend=True
                    ),
                    row=1, col=1
                )
    
    # Volume chart
    fig.add_trace(
        go.Bar(x=spy_data.index, y=spy_data[volume_col], 
               name='Volume', marker_color='rgba(46, 134, 171, 0.6)'),
        row=2, col=1
    )
    
    # RSI chart with signal levels
    if hasattr(analyzer, 'features') and 'SPY' in analyzer.features and 'rsi' in analyzer.features['SPY']:
        rsi_data = analyzer.features['SPY']['rsi']
        fig.add_trace(
            go.Scatter(x=rsi_data.index, y=rsi_data, 
                      name='RSI', line=dict(color='#9B59B6', width=2)),
            row=3, col=1
        )
        
        # Add RSI trigger levels
        fig.add_hline(y=70, line_dash="dash", line_color="#E74C3C", 
                     annotation_text="Overbought (70)", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#27AE60", 
                     annotation_text="Oversold (30)", row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="#7F8C8D", 
                     annotation_text="Neutral (50)", row=3, col=1)
        
        # Highlight RSI extreme zones
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(231, 76, 60, 0.1)", 
                     layer="below", line_width=0, row=3, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(39, 174, 96, 0.1)", 
                     layer="below", line_width=0, row=3, col=1)
    
    # Update layout
    fig.update_layout(
        height=800, 
        showlegend=True,
        title_text="Live Market Analysis with Signal Triggers",
        title_font_size=16,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update x-axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_signal_strength_gauge(signals):
    """Create signal strength gauge"""
    if signals.empty:
        return None
    
    current_signals = signals.iloc[-1]
    active_signals = current_signals[current_signals > 0]
    signal_strength = len(active_signals) / len(current_signals) * 100 if len(current_signals) > 0 else 0
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = signal_strength,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Signal Strength"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# ... (keep all the Flask routes from the original app.py) ...

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/load_data')
def load_data():
    """API endpoint to load market data"""
    period = request.args.get('period', '30d')
    interval = request.args.get('interval', '5m')
    mock_mode = request.args.get('mock', 'false').lower() == 'true'
    
    logger.info(f"Loading data: period={period}, interval={interval}, mock={mock_mode}")
    
    success, message = load_market_data(period, interval, force_mock=mock_mode)
    
    if success:
        # Get current metrics
        analyzer = data_cache['analyzer']
        signals = data_cache['signals']
        
        metrics = {
            'data_points': int(len(analyzer.data['SPY']) if hasattr(analyzer, 'data') and 'SPY' in analyzer.data else 0),
            'signal_types': int(len(signals.columns) if not signals.empty else 0),
            'active_signals': 0,
            'features': int(len(analyzer.features.get('SPY', {})) if hasattr(analyzer, 'features') else 0),
            'last_update': data_cache['last_update'].strftime('%Y-%m-%d %H:%M:%S'),
            'mock_mode': data_cache['mock_mode']
        }
        
        if not signals.empty:
            current_signals = signals.iloc[-1]
            metrics['active_signals'] = int((current_signals > 0).sum())
        
        return jsonify({
            'success': True,
            'message': message,
            'metrics': metrics
        })
    else:
        return jsonify({
            'success': False,
            'message': message
        }), 500

@app.route('/api/alerts')
def get_alerts():
    """Get current trading alerts"""
    if data_cache['alerts'] is None:
        return jsonify({'alerts': []})
    
    return jsonify({'alerts': data_cache['alerts']})

@app.route('/api/active_signals')
def get_active_signals():
    """Get currently active signals"""
    if data_cache['signals'] is None or data_cache['signals'].empty:
        return jsonify({'active_signals': [], 'signal_strength': 0})
    
    signals = data_cache['signals']
    current_signals = signals.iloc[-1]
    active_signals = current_signals[current_signals > 0]
    
    active_list = []
    for signal_name in active_signals.index:
        if signal_name != 'total_signals':
            active_list.append({
                'name': signal_name.replace('_', ' ').title(),
                'value': int(active_signals[signal_name])
            })
    
    signal_strength = float(len(active_signals) / len(current_signals) * 100) if len(current_signals) > 0 else 0.0
    
    return jsonify({
        'active_signals': active_list,
        'signal_strength': signal_strength
    })

@app.route('/api/charts/performance')
def performance_chart():
    """Get performance chart data"""
    if data_cache['performance'] is None or data_cache['performance'].empty:
        return jsonify({'chart': None})
    
    chart_json = create_performance_chart(data_cache['performance'])
    return jsonify({'chart': chart_json})

@app.route('/api/charts/heatmap')
def heatmap_chart():
    """Get signals heatmap chart data"""
    if data_cache['signals'] is None or data_cache['signals'].empty:
        return jsonify({'chart': None})
    
    chart_json = create_signals_heatmap(data_cache['signals'])
    return jsonify({'chart': chart_json})

@app.route('/api/charts/market_overview')
def market_overview_chart():
    """Get market overview chart data"""
    if data_cache['analyzer'] is None:
        return jsonify({'chart': None})
    
    chart_json = create_market_overview(data_cache['analyzer'])
    return jsonify({'chart': chart_json})

@app.route('/api/charts/signal_gauge')
def signal_gauge_chart():
    """Get signal strength gauge chart data"""
    if data_cache['signals'] is None or data_cache['signals'].empty:
        return jsonify({'chart': None})
    
    chart_json = create_signal_strength_gauge(data_cache['signals'])
    return jsonify({'chart': chart_json})

@app.route('/api/performance_table')
def performance_table():
    """Get performance data as table"""
    if data_cache['performance'] is None or data_cache['performance'].empty:
        return jsonify({'data': []})
    
    performance = data_cache['performance'].copy()
    performance['win_rate'] = performance['win_rate'].apply(lambda x: f"{x:.2%}")
    performance['signal_sharpe'] = performance['signal_sharpe'].round(4)
    performance['excess_return'] = performance['excess_return'].apply(lambda x: f"{x:.4f}")
    
    # Convert to records for JSON serialization
    table_data = performance[['win_rate', 'signal_sharpe', 'count', 'excess_return']].reset_index()
    return jsonify({'data': table_data.to_dict('records')})

@app.route('/api/performance_stats')
def performance_stats():
    """Get performance statistics"""
    if data_cache['performance'] is None or data_cache['performance'].empty:
        return jsonify({'stats': {}})
    
    performance = data_cache['performance']
    perf_5min = performance[performance.index.str.contains('5min')]
    
    if perf_5min.empty:
        return jsonify({'stats': {}})
    
    stats = {
        'avg_win_rate': f"{perf_5min['win_rate'].mean():.2%}",
        'best_signal': perf_5min.loc[perf_5min['win_rate'].idxmax()].name.replace('_5min', ''),
        'total_signals': f"{perf_5min['count'].sum():,}"
    }
    
    return jsonify({'stats': stats})

@app.route('/api/export/excel')
def export_excel():
    """Export analysis to Excel"""
    if data_cache['analyzer'] is None:
        return jsonify({'error': 'No data available'}), 400
    
    try:
        analyzer = data_cache['analyzer']
        signals = data_cache['signals']
        performance = data_cache['performance']
        alerts = data_cache['alerts']
        
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Export signals
            if not signals.empty:
                signals.to_excel(writer, sheet_name='Current_Signals')
            
            # Export performance
            if not performance.empty:
                performance.to_excel(writer, sheet_name='Performance_Analysis')
            
            # Export alerts
            if alerts:
                alerts_df = pd.DataFrame(alerts)
                alerts_df.to_excel(writer, sheet_name='Trading_Alerts', index=False)
            
            # Export recent price data
            if hasattr(analyzer, 'data') and 'SPY' in analyzer.data:
                spy_recent = analyzer.data['SPY'].tail(100)
                spy_recent.to_excel(writer, sheet_name='Recent_SPY_Data')
            
            # Export features summary
            if hasattr(analyzer, 'features') and 'SPY' in analyzer.features:
                features_summary = pd.DataFrame({
                    'Feature': list(analyzer.features['SPY'].keys()),
                    'Type': ['Technical Indicator'] * len(analyzer.features['SPY'])
                })
                features_summary.to_excel(writer, sheet_name='Features_Summary', index=False)
        
        output.seek(0)
        
        filename = f"spy_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        
        return send_file(
            output,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/summary')
def get_summary():
    """Get daily summary data"""
    if data_cache['analyzer'] is None:
        return jsonify({'summary': 'No data available'})
    
    analyzer = data_cache['analyzer']
    signals = data_cache['signals']
    performance = data_cache['performance']
    alerts = data_cache['alerts']
    
    summary = {
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        'data_points': int(len(analyzer.data['SPY']) if hasattr(analyzer, 'data') and 'SPY' in analyzer.data else 0),
        'signal_types': int(len(signals.columns) if not signals.empty else 0),
        'features_calculated': int(len(analyzer.features.get('SPY', {})) if hasattr(analyzer, 'features') else 0),
        'current_alerts': int(len(alerts) if alerts else 0),
        'performance_available': bool(not performance.empty),
        'mock_mode': bool(data_cache.get('mock_mode', False))
    }
    
    if not signals.empty:
        current_signals = signals.iloc[-1]
        summary['active_signals'] = int((current_signals > 0).sum())
    else:
        summary['active_signals'] = 0
    
    if not performance.empty:
        perf_5min = performance[performance.index.str.contains('5min')]
        if not perf_5min.empty:
            best_signal = perf_5min.loc[perf_5min['win_rate'].idxmax()]
            summary['best_signal'] = {
                'name': best_signal.name.replace('_5min', ''),
                'win_rate': f"{best_signal['win_rate']:.2%}",
                'sharpe_ratio': f"{best_signal['signal_sharpe']:.4f}"
            }
    
    return jsonify({'summary': summary})

@app.route('/api/mock_data')
def enable_mock_data():
    """Enable mock data mode for testing"""
    success, message = load_market_data('30d', '5m', force_mock=True)
    return jsonify({'success': success, 'message': message, 'mock_mode': True})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'data_age': (datetime.datetime.now() - data_cache['last_update']).seconds if data_cache['last_update'] else None,
        'mock_mode': data_cache.get('mock_mode', False),
        'analysis_framework': ANALYSIS_AVAILABLE
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    logger.info("Starting SPY 0DTE Options Analysis Dashboard")
    logger.info(f"Analysis framework available: {ANALYSIS_AVAILABLE}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
