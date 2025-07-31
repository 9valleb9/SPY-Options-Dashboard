import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Set display options for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

class MarketStructureAnalyzer:
    def __init__(self):
        # Market-wide tickers for comprehensive analysis
        self.tickers = {
            'SPY': 'S&P 500 ETF (Primary)',
            '^VIX': 'Volatility Index',
            'QQQ': 'NASDAQ 100 ETF',
            'IWM': 'Russell 2000 ETF',
            'DIA': 'Dow Jones ETF',
            'XLF': 'Financial Sector',
            'XLK': 'Technology Sector',
            'HYG': 'High Yield Credit',
            'TLT': 'Long-term Treasury',
            'UUP': 'US Dollar Index',
        }
        
        self.data = {}
        self.features = {}
        
    def fetch_data(self, period='60d', interval='5m'):
        """Fetch intraday data for all tickers"""
        print("Fetching market data...")
        
        for ticker in self.tickers.keys():
            try:
                data = yf.download(ticker, period=period, interval=interval, progress=False)
                if not data.empty:
                    self.data[ticker] = data
                    print(f"✓ {ticker}: {len(data)} data points")
                else:
                    print(f"✗ {ticker}: No data available")
            except Exception as e:
                print(f"✗ {ticker}: Error - {e}")
                
    def calculate_returns_and_volatility(self, data):
        """Calculate returns, log returns, and volatility metrics"""
        features = {}
        
        try:
            # Handle MultiIndex columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                ticker = data.columns[0][1]  # Get ticker name
                close = data[('Close', ticker)]
                high = data[('High', ticker)]
                low = data[('Low', ticker)]
            else:
                close = data['Close']
                high = data['High'] 
                low = data['Low']
            
            # Returns
            features['returns'] = close.pct_change()
            features['log_returns'] = np.log(close / close.shift(1))
            
            # Volatility measures
            features['volatility_5'] = features['returns'].rolling(window=12).std()  # 1-hour rolling (5min * 12)
            features['volatility_15'] = features['returns'].rolling(window=36).std()  # 3-hour rolling
            features['volatility_30'] = features['returns'].rolling(window=72).std()  # 6-hour rolling
            
            # Intraday range volatility
            hl_diff = high - low
            hc_diff = (high - close.shift(1)).abs()
            lc_diff = (low - close.shift(1)).abs()
            
            # Use pandas concat and max instead of numpy maximum
            tr_components = pd.concat([hl_diff, hc_diff, lc_diff], axis=1)
            features['true_range'] = tr_components.max(axis=1)
            
        except Exception as e:
            print(f"Error in returns calculation: {e}")
            # Create empty series with proper index
            features['returns'] = pd.Series(index=data.index, dtype=float)
            features['volatility_5'] = pd.Series(index=data.index, dtype=float)
            features['true_range'] = pd.Series(index=data.index, dtype=float)
        
        return features
    
    def calculate_technical_indicators(self, data):
        """Calculate RSI, MACD, Stochastic, ATR, Bollinger Bands"""
        features = {}
        
        try:
            # Handle MultiIndex columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                # Get the ticker name from the first column
                ticker = data.columns[0][1]  # e.g., 'SPY' from ('Close', 'SPY')
                close = data[('Close', ticker)]
                high = data[('High', ticker)]
                low = data[('Low', ticker)]
                volume = data.get(('Volume', ticker), pd.Series(index=data.index, dtype=float))
            else:
                # Handle regular column structure
                close = data['Close']
                high = data['High']
                low = data['Low']
                volume = data.get('Volume', pd.Series(index=data.index, dtype=float))
            
            # RSI (14-period)
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            
            # Avoid division by zero with safe division
            rs = gain / loss.replace(0, np.nan)
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD (12, 26, 9)
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            features['macd'] = ema12 - ema26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            
            # Stochastic (14, 3, 3)
            lowest_low = low.rolling(window=14).min()
            highest_high = high.rolling(window=14).max()
            range_diff = highest_high - lowest_low
            
            # Safe division for stochastic
            k_percent = 100 * ((close - lowest_low) / range_diff.replace(0, np.nan))
            features['stoch_k'] = k_percent.rolling(window=3).mean()
            features['stoch_d'] = features['stoch_k'].rolling(window=3).mean()
            
            # ATR (14-period) - Fixed approach
            high_low = high - low
            high_close = (high - close.shift(1)).abs()
            low_close = (low - close.shift(1)).abs()
            
            tr_components = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = tr_components.max(axis=1)
            features['atr'] = true_range.rolling(window=14).mean()
            
            # Bollinger Bands
            bb_window = 20
            bb_std = 2
            bb_middle = close.rolling(window=bb_window).mean()
            bb_std_dev = close.rolling(window=bb_window).std()
            bb_upper = bb_middle + (bb_std_dev * bb_std)
            bb_lower = bb_middle - (bb_std_dev * bb_std)
            
            # Safe division for BB
            features['bb_width'] = ((bb_upper - bb_lower) / bb_middle.replace(0, np.nan)) * 100
            features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
            
            # OBV (On-Balance Volume) - Fixed for MultiIndex
            if not volume.empty and len(volume) > 1:
                try:
                    price_change = close.diff()
                    
                    # Create volume direction series with same index as close
                    volume_direction = pd.Series(0, index=close.index)
                    volume_direction[price_change > 0] = 1
                    volume_direction[price_change < 0] = -1
                    
                    obv_changes = volume * volume_direction
                    features['obv'] = obv_changes.cumsum()
                except Exception as obv_error:
                    print(f"OBV calculation error: {obv_error}")
                    features['obv'] = pd.Series(0, index=close.index)
            else:
                features['obv'] = pd.Series(0, index=close.index)
                
        except Exception as e:
            print(f"Error in technical indicators: {e}")
            # Create empty series for failed calculations
            for indicator in ['rsi', 'macd', 'atr', 'bb_width', 'obv']:
                features[indicator] = pd.Series(index=data.index, dtype=float)
        
        return features
    
    def calculate_market_structure_features(self):
        """Calculate advanced market structure features"""
        print("Calculating market structure features...")
        
        for ticker, data in self.data.items():
            try:
                print(f"Processing {ticker}...")
                
                # Basic features
                basic_features = self.calculate_returns_and_volatility(data)
                technical_features = self.calculate_technical_indicators(data)
                
                # Combine all features
                self.features[ticker] = {**basic_features, **technical_features}
                
            except Exception as e:
                print(f"Warning: Error processing {ticker}: {e}")
                # Initialize empty features for this ticker
                self.features[ticker] = {
                    'returns': pd.Series(index=data.index),
                    'rsi': pd.Series(index=data.index),
                    'macd': pd.Series(index=data.index)
                }
                continue
            
        # Calculate cross-asset relationships
        try:
            self._calculate_correlations()
        except Exception as e:
            print(f"Warning: Error calculating correlations: {e}")
            
        try:
            self._calculate_divergence_signals()
        except Exception as e:
            print(f"Warning: Error calculating divergence signals: {e}")
            
        try:
            self._calculate_regime_indicators()
        except Exception as e:
            print(f"Warning: Error calculating regime indicators: {e}")
    
    def _calculate_correlations(self):
        """Calculate rolling correlations between key assets"""
        print("Calculating cross-asset correlations...")
        
        # Key correlation pairs for 0DTE strategy
        correlation_pairs = [
            ('SPY', '^VIX', 'vix_spy_corr', 12),  # 1-hour rolling
            ('SPY', 'QQQ', 'spy_qqq_corr', 12),
            ('SPY', 'IWM', 'spy_iwm_corr', 12),
            ('SPY', 'TLT', 'spy_tlt_corr', 36),   # 3-hour rolling
            ('SPY', 'HYG', 'spy_hyg_corr', 36),
            ('HYG', 'TLT', 'hyg_tlt_corr', 36),
        ]
        
        for ticker1, ticker2, name, window in correlation_pairs:
            if ticker1 in self.data and ticker2 in self.data:
                try:
                    data1 = self.data[ticker1]
                    data2 = self.data[ticker2]
                    
                    returns1 = data1['Close'].pct_change()
                    returns2 = data2['Close'].pct_change()
                    
                    # Align data
                    aligned_data = pd.concat([returns1, returns2], axis=1, keys=[ticker1, ticker2]).dropna()
                    
                    if len(aligned_data) > window:
                        correlation = aligned_data[ticker1].rolling(window=window).corr(aligned_data[ticker2])
                        
                        # Store in SPY features as primary reference
                        if 'correlations' not in self.features.get('SPY', {}):
                            self.features['SPY']['correlations'] = {}
                        self.features['SPY']['correlations'][name] = correlation
                        
                except Exception as e:
                    print(f"Warning: Error calculating correlation {name}: {e}")
    
    def _calculate_divergence_signals(self):
        """Calculate divergence signals between SPY and other assets"""
        print("Calculating divergence signals...")
        
        if 'SPY' not in self.data or 'QQQ' not in self.data:
            return
            
        try:
            spy_data = self.data['SPY']
            qqq_data = self.data['QQQ']
            
            spy_returns = spy_data['Close'].pct_change()
            qqq_returns = qqq_data['Close'].pct_change()
            
            # Calculate performance divergence
            spy_cum_returns = (1 + spy_returns).cumprod()
            qqq_cum_returns = (1 + qqq_returns).cumprod()
            
            # Normalize to same starting point
            spy_norm = spy_cum_returns / spy_cum_returns.iloc[0] if len(spy_cum_returns) > 0 else spy_cum_returns
            qqq_norm = qqq_cum_returns / qqq_cum_returns.iloc[0] if len(qqq_cum_returns) > 0 else qqq_cum_returns
            
            # Divergence signal (positive = QQQ outperforming)
            divergence = qqq_norm - spy_norm
            divergence_ma = divergence.rolling(window=12).mean()
            
            self.features['SPY']['qqq_spy_divergence'] = divergence
            self.features['SPY']['qqq_spy_divergence_ma'] = divergence_ma
            
            # Divergence momentum (rate of change of divergence)
            self.features['SPY']['divergence_momentum'] = divergence.diff()
            
        except Exception as e:
            print(f"Warning: Error calculating divergence signals: {e}")
    
    def _calculate_regime_indicators(self):
        """Calculate market regime indicators"""
        print("Calculating market regime indicators...")
        
        if 'SPY' not in self.features:
            return
            
        # VIX regime classification
        if '^VIX' in self.data:
            try:
                vix_data = self.data['^VIX']
                
                # Handle MultiIndex columns
                if isinstance(vix_data.columns, pd.MultiIndex):
                    vix_close = vix_data[('Close', '^VIX')]
                else:
                    vix_close = vix_data['Close']
                
                vix_ma = vix_close.rolling(window=60).mean()  # 5-hour average
                
                # VIX regime (0=low vol, 1=medium vol, 2=high vol) - simplified approach
                vix_regime = pd.Series(0, index=vix_close.index, dtype=int)
                vix_regime.loc[vix_close >= 25] = 2  # High vol
                vix_regime.loc[(vix_close >= 15) & (vix_close < 25)] = 1  # Medium vol
                # Low vol (< 15) stays 0
                
                self.features['SPY']['vix_regime'] = vix_regime
                
                # VIX term structure (current vs MA)
                self.features['SPY']['vix_term_structure'] = (vix_close / vix_ma) - 1
                
                # VIX spike detection (2+ standard deviations above mean)
                vix_mean = vix_close.rolling(window=96).mean()
                vix_std = vix_close.rolling(window=96).std()
                self.features['SPY']['vix_spike'] = (vix_close - vix_mean) / vix_std.replace(0, np.nan)
                
            except Exception as e:
                print(f"Warning: Error processing VIX regime: {e}")
        
        # Credit-Treasury spread (HYG vs TLT)
        if 'HYG' in self.data and 'TLT' in self.data:
            try:
                hyg_data = self.data['HYG']
                tlt_data = self.data['TLT']
                hyg_returns = hyg_data['Close'].pct_change()
                tlt_returns = tlt_data['Close'].pct_change()
                
                # Credit spread proxy (inverse relationship)
                credit_spread_proxy = tlt_returns - hyg_returns
                self.features['SPY']['credit_spread_proxy'] = credit_spread_proxy
                self.features['SPY']['credit_spread_ma'] = credit_spread_proxy.rolling(window=36).mean()
            except Exception as e:
                print(f"Warning: Error processing credit spread: {e}")
        
        # Dollar strength regime
        if 'UUP' in self.data:
            try:
                uup_data = self.data['UUP']
                uup_returns = uup_data['Close'].pct_change()
                uup_ma_fast = uup_returns.rolling(window=12).mean()
                uup_ma_slow = uup_returns.rolling(window=48).mean()
                self.features['SPY']['dollar_momentum'] = uup_ma_fast - uup_ma_slow
            except Exception as e:
                print(f"Warning: Error processing dollar momentum: {e}")
        
        # Sector rotation analysis
        if 'XLF' in self.data and 'XLK' in self.data:
            try:
                xlf_data = self.data['XLF']
                xlk_data = self.data['XLK']
                xlf_returns = xlf_data['Close'].pct_change()
                xlk_returns = xlk_data['Close'].pct_change()
                
                # Financial vs Tech rotation (risk-on vs risk-off proxy)
                sector_rotation = xlf_returns - xlk_returns
                self.features['SPY']['fin_tech_rotation'] = sector_rotation.rolling(window=24).mean()
            except Exception as e:
                print(f"Warning: Error processing sector rotation: {e}")
        
        # Market breadth - skip since ^NYAD is not available
        # if '^NYAD' in self.data:
        #     try:
        #         nyad_close = self.data['^NYAD']['Close']
        #         nyad_ma = nyad_close.rolling(window=48).mean()
        #         self.features['SPY']['breadth_momentum'] = (nyad_close - nyad_ma) / nyad_ma
        #     except Exception as e:
        #         print(f"Warning: Error processing breadth: {e}")
    
    def calculate_scalping_signals(self):
        """Calculate comprehensive signals for 0DTE scalping strategy with detailed descriptions"""
        print("Generating comprehensive 0DTE scalping signals...")
        
        if 'SPY' not in self.features:
            return pd.DataFrame()
            
        spy_features = self.features['SPY']
        signals = {}
        
        # VIX-SPY Correlation Breakdown
        if 'correlations' in spy_features and 'vix_spy_corr' in spy_features['correlations']:
            try:
                vix_spy_corr = spy_features['correlations']['vix_spy_corr']
                signals['vix_breakdown'] = (vix_spy_corr > -0.3).astype(int)
            except Exception as e:
                print(f"Warning: Error creating vix_breakdown signal: {e}")
        
        # VIX Spike Detection
        if 'vix_spike' in spy_features:
            try:
                vix_spike = spy_features['vix_spike']
                signals['vix_spike_bullish'] = (vix_spike > 2.0).astype(int)
                signals['vix_spike_bearish'] = (vix_spike < -1.5).astype(int)
            except Exception as e:
                print(f"Warning: Error creating vix_spike signals: {e}")
        
        # MACD signals
        if 'macd_histogram' in spy_features:
            try:
                macd_hist = spy_features['macd_histogram']
                macd_bullish = (macd_hist > 0) & (macd_hist.shift(1) <= 0)
                macd_bearish = (macd_hist < 0) & (macd_hist.shift(1) >= 0)
                signals['macd_bullish'] = macd_bullish.astype(int)
                signals['macd_bearish'] = macd_bearish.astype(int)
            except Exception as e:
                print(f"Warning: Error creating MACD signals: {e}")
        
        # RSI signals
        if 'rsi' in spy_features:
            try:
                rsi = spy_features['rsi']
                signals['rsi_oversold'] = (rsi < 30).astype(int)
                signals['rsi_overbought'] = (rsi > 70).astype(int)
            except Exception as e:
                print(f"Warning: Error creating RSI signals: {e}")
        
        # Bollinger Band signals
        if 'bb_width' in spy_features and 'bb_position' in spy_features:
            try:
                bb_width = spy_features['bb_width']
                bb_pos = spy_features['bb_position']
                
                bb_compressed = bb_width < bb_width.rolling(window=48).quantile(0.2)
                signals['bb_compression'] = bb_compressed.astype(int)
                
                bb_mean_revert_bull = (bb_pos < 0.1) & (bb_pos.shift(1) >= 0.1)
                bb_mean_revert_bear = (bb_pos > 0.9) & (bb_pos.shift(1) <= 0.9)
                signals['bb_mean_revert_bull'] = bb_mean_revert_bull.astype(int)
                signals['bb_mean_revert_bear'] = bb_mean_revert_bear.astype(int)
            except Exception as e:
                print(f"Warning: Error creating Bollinger Band signals: {e}")
        
        # Volatility regime signals
        if all(k in spy_features for k in ['volatility_5', 'volatility_15', 'volatility_30']):
            try:
                vol_5 = spy_features['volatility_5']
                vol_15 = spy_features['volatility_15']
                vol_30 = spy_features['volatility_30']
                
                vol_expansion = (vol_5 > vol_15) & (vol_15 > vol_30)
                vol_contraction = (vol_5 < vol_15) & (vol_15 < vol_30)
                signals['vol_expansion'] = vol_expansion.astype(int)
                signals['vol_contraction'] = vol_contraction.astype(int)
            except Exception as e:
                print(f"Warning: Error creating volatility signals: {e}")
        
        # Time-based signals
        if 'SPY' in self.data:
            try:
                spy_data = self.data['SPY']
                time_index = spy_data.index
                time_hour = time_index.hour
                time_minute = time_index.minute
                
                # Create boolean arrays first, then convert to int
                opening_condition = ((time_hour == 9) & (time_minute >= 30)) | ((time_hour == 10) & (time_minute < 30))
                power_condition = (time_hour == 15)
                final_condition = (time_hour == 15) & (time_minute >= 30)
                
                signals['opening_hour'] = opening_condition.astype(int)
                signals['power_hour'] = power_condition.astype(int)
                signals['final_30min'] = final_condition.astype(int)
                
            except Exception as e:
                print(f"Warning: Error creating time-based signals: {e}")
        
        # Create signals DataFrame with proper error handling
        if signals:
            try:
                # Ensure all signals are pandas Series with same index
                cleaned_signals = {}
                base_index = None
                
                for key, value in signals.items():
                    if isinstance(value, pd.Series):
                        if base_index is None:
                            base_index = value.index
                        cleaned_signals[key] = value.reindex(base_index, fill_value=0)
                    elif isinstance(value, (list, np.ndarray)):
                        if base_index is None and 'SPY' in self.data:
                            base_index = self.data['SPY'].index
                        if base_index is not None:
                            cleaned_signals[key] = pd.Series(value, index=base_index[:len(value)])
                
                if cleaned_signals:
                    signal_df = pd.DataFrame(cleaned_signals)
                else:
                    # Create empty DataFrame with SPY index
                    if 'SPY' in self.data:
                        signal_df = pd.DataFrame(index=self.data['SPY'].index)
                    else:
                        signal_df = pd.DataFrame()
                        
            except Exception as e:
                print(f"Warning: Error creating signal DataFrame: {e}")
                # Fallback: create empty DataFrame
                if 'SPY' in self.data:
                    signal_df = pd.DataFrame(index=self.data['SPY'].index)
                else:
                    signal_df = pd.DataFrame()
        else:
            # Create empty DataFrame with SPY index if no signals
            if 'SPY' in self.data:
                signal_df = pd.DataFrame(index=self.data['SPY'].index)
            else:
                signal_df = pd.DataFrame()
        
        # Add composite signals if we have data
        if not signal_df.empty:
            # Total signals
            if len(signal_df.columns) > 0:
                signal_df['total_signals'] = signal_df.sum(axis=1, numeric_only=True)
            else:
                signal_df['total_signals'] = 0
            
            # Scalp signals - only create if we have required columns
            required_long_cols = ['vix_spike_bullish', 'rsi_oversold', 'bb_mean_revert_bull', 'macd_bullish']
            available_long_cols = [col for col in required_long_cols if col in signal_df.columns]
            
            if len(available_long_cols) >= 2:
                signal_df['scalp_long'] = (signal_df[available_long_cols].sum(axis=1) >= 2).astype(int)
            else:
                signal_df['scalp_long'] = 0
            
            required_short_cols = ['rsi_overbought', 'bb_mean_revert_bear', 'macd_bearish']
            available_short_cols = [col for col in required_short_cols if col in signal_df.columns]
            
            if len(available_short_cols) >= 2:
                signal_df['scalp_short'] = (signal_df[available_short_cols].sum(axis=1) >= 2).astype(int)
            else:
                signal_df['scalp_short'] = 0
        else:
            # Even for empty DataFrame, add the columns
            signal_df['total_signals'] = 0
            signal_df['scalp_long'] = 0
            signal_df['scalp_short'] = 0
        
        self.features['SPY']['signals'] = signal_df
        return signal_df
    
    def analyze_signal_performance(self):
        """Analyze the statistical performance of generated signals"""
        if 'signals' not in self.features.get('SPY', {}):
            print("No signals to analyze. Run calculate_scalping_signals() first.")
            return pd.DataFrame()
            
        try:
            signals = self.features['SPY']['signals']
            spy_returns = self.features['SPY'].get('returns', pd.Series())
            
            if signals.empty or spy_returns.empty:
                return pd.DataFrame()
            
            # Align indices between signals and returns
            common_index = signals.index.intersection(spy_returns.index)
            if len(common_index) == 0:
                return pd.DataFrame()
            
            signals_aligned = signals.loc[common_index]
            returns_aligned = spy_returns.loc[common_index]
            
            # Forward returns for signal evaluation
            forward_periods = [1, 3, 6, 12]
            results = {}
            
            for period in forward_periods:
                period_name = f"{period*5}min"
                forward_returns = returns_aligned.shift(-period)
                
                # Analyze each signal
                for signal_name in signals_aligned.columns:
                    if signal_name in ['total_signals']:
                        continue
                    
                    try:
                        signal_col = signals_aligned[signal_name]
                        
                        # Skip if signal is all zeros or empty
                        if signal_col.sum() == 0:
                            continue
                        
                        # Handle boolean and numeric signals differently
                        if signal_col.dtype == 'bool':
                            signal_active = signal_col
                        else:
                            signal_active = signal_col == 1
                        
                        # Skip if not enough signals (need at least 10 occurrences)
                        signal_count = signal_active.sum()
                        if signal_count < 10:
                            continue
                            
                        # Get forward returns for analysis - use aligned indices
                        signal_returns = forward_returns[signal_active].dropna()
                        baseline_returns = forward_returns[~signal_active].dropna()
                        
                        if len(signal_returns) > 0 and len(baseline_returns) > 0:
                            results[f"{signal_name}_{period_name}"] = {
                                'signal_mean': signal_returns.mean(),
                                'baseline_mean': baseline_returns.mean(),
                                'signal_std': signal_returns.std(),
                                'signal_sharpe': signal_returns.mean() / signal_returns.std() if signal_returns.std() > 0 else 0,
                                'win_rate': (signal_returns > 0).mean(),
                                'count': len(signal_returns),
                                'excess_return': signal_returns.mean() - baseline_returns.mean()
                            }
                    except Exception as e:
                        # Skip problematic signals silently for now
                        continue
            
            return pd.DataFrame(results).T
            
        except Exception as e:
            print(f"Error in performance analysis: {e}")
            return pd.DataFrame()
    
    def create_dashboard_data(self):
        """Prepare data for visualization dashboard"""
        if 'SPY' not in self.features:
            return None
            
        try:
            spy_data = self.data['SPY'].copy()
            spy_features = self.features['SPY']
            
            # Start with price data
            dashboard_data = spy_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            # Add technical indicators with proper alignment
            for indicator in ['rsi', 'macd', 'bb_width', 'volatility_5', 'atr']:
                if indicator in spy_features:
                    try:
                        feature_data = spy_features[indicator]
                        if isinstance(feature_data, pd.Series) and not feature_data.empty:
                            # Reindex to match dashboard_data
                            aligned_feature = feature_data.reindex(dashboard_data.index)
                            dashboard_data[indicator] = aligned_feature
                    except Exception as e:
                        print(f"Warning: Could not add {indicator} to dashboard: {e}")
            
            # Add signals with proper alignment
            if 'signals' in spy_features:
                try:
                    signals = spy_features['signals']  
                    if isinstance(signals, pd.DataFrame) and not signals.empty:
                        # Reindex signals to match dashboard_data
                        aligned_signals = signals.reindex(dashboard_data.index, fill_value=0)
                        
                        # Add signal columns with prefix
                        for col in aligned_signals.columns:
                            dashboard_data[f'signal_{col}'] = aligned_signals[col]
                except Exception as e:
                    print(f"Warning: Could not add signals to dashboard: {e}")
            
            return dashboard_data
            
        except Exception as e:
            print(f"Warning: Error creating dashboard data: {e}")
            # Return basic price data if everything else fails
            if 'SPY' in self.data:
                return self.data['SPY'][['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            return None

def run_analysis():
    """Run the complete analysis pipeline"""
    analyzer = MarketStructureAnalyzer()
    
    # Fetch data (last 30 days, 5-minute intervals)
    analyzer.fetch_data(period='30d', interval='5m')
    
    # Calculate all features
    analyzer.calculate_market_structure_features()
    
    # Generate scalping signals
    signals = analyzer.calculate_scalping_signals()
    
    # Analyze signal performance
    performance = analyzer.analyze_signal_performance()
    
    # Create dashboard data
    dashboard_data = analyzer.create_dashboard_data()
    
    return analyzer, signals, performance, dashboard_data

def generate_trading_alerts(analyzer):
    """Generate real-time trading alerts based on current signal conditions"""
    if 'SPY' not in analyzer.features or 'signals' not in analyzer.features['SPY']:
        return []
    
    signals = analyzer.features['SPY']['signals']
    if signals.empty:
        return []
        
    current_signals = signals.iloc[-1]
    alerts = []
    
    # Check Bollinger Band compression (volatility expansion setup)
    if current_signals.get('bb_compression', 0) == 1:
        alerts.append({
            'priority': 'MEDIUM',
            'type': 'VOLATILITY_SETUP',
            'message': 'BOLLINGER BAND COMPRESSION DETECTED - Volatility expansion setup',
            'action': 'Consider long straddles/strangles - expecting breakout in either direction',
            'timeframe': '30 minutes - 2 hours',
            'confidence': '70%+'
        })
    
    # Check volatility contraction (potential reversal or continuation setup)
    if current_signals.get('vol_contraction', 0) == 1:
        alerts.append({
            'priority': 'MEDIUM',
            'type': 'MOMENTUM_CHANGE',
            'message': 'VOLATILITY CONTRACTION - Momentum may be fading',
            'action': 'Watch for reversal signals or consolidation before next move',
            'timeframe': '15-60 minutes',
            'confidence': '65%+'
        })
    
    # Check MACD signals
    if current_signals.get('macd_bullish', 0) == 1:
        alerts.append({
            'priority': 'HIGH',
            'type': 'MOMENTUM_BULLISH',
            'message': 'MACD BULLISH CROSSOVER - Momentum shifting upward',
            'action': 'Consider ATM/ITM call spreads for momentum continuation',
            'timeframe': '15-45 minutes',
            'confidence': '75%+'
        })
    
    if current_signals.get('macd_bearish', 0) == 1:
        alerts.append({
            'priority': 'HIGH',
            'type': 'MOMENTUM_BEARISH',
            'message': 'MACD BEARISH CROSSOVER - Momentum shifting downward',
            'action': 'Consider ATM/ITM put spreads for momentum continuation',
            'timeframe': '15-45 minutes',
            'confidence': '75%+'
        })
    
    # Check RSI extremes
    if current_signals.get('rsi_oversold', 0) == 1:
        alerts.append({
            'priority': 'MEDIUM',
            'type': 'MEAN_REVERSION',
            'message': 'RSI OVERSOLD CONDITION - Potential bounce setup',
            'action': 'Contrarian call spreads, especially near support levels',
            'timeframe': '10-30 minutes',
            'confidence': '65%+'
        })
    
    if current_signals.get('rsi_overbought', 0) == 1:
        alerts.append({
            'priority': 'MEDIUM',
            'type': 'MEAN_REVERSION',
            'message': 'RSI OVERBOUGHT CONDITION - Potential pullback setup',
            'action': 'Contrarian put spreads, especially near resistance levels',
            'timeframe': '10-30 minutes',
            'confidence': '65%+'
        })
    
    # Check Bollinger Band mean reversion
    if current_signals.get('bb_mean_revert_bull', 0) == 1:
        alerts.append({
            'priority': 'HIGH',
            'type': 'MEAN_REVERSION',
            'message': 'BOLLINGER BAND BOUNCE - Strong mean reversion signal upward',
            'action': 'High-conviction call spreads targeting middle band',
            'timeframe': '5-20 minutes',
            'confidence': '80%+'
        })
    
    if current_signals.get('bb_mean_revert_bear', 0) == 1:
        alerts.append({
            'priority': 'HIGH',
            'type': 'MEAN_REVERSION',
            'message': 'BOLLINGER BAND REJECTION - Strong mean reversion signal downward',
            'action': 'High-conviction put spreads targeting middle band',
            'timeframe': '5-20 minutes',
            'confidence': '80%+'
        })
    
    # Check composite scalping signals
    if current_signals.get('scalp_long', 0) == 1:
        alerts.append({
            'priority': 'HIGH',
            'type': 'SCALP_LONG',
            'message': 'MULTIPLE BULLISH SIGNALS ALIGNED - High-conviction long setup',
            'action': 'Quick call spreads with tight stops - scalping opportunity',
            'timeframe': '5-15 minutes',
            'confidence': '85%+'
        })
    
    if current_signals.get('scalp_short', 0) == 1:
        alerts.append({
            'priority': 'HIGH',
            'type': 'SCALP_SHORT',
            'message': 'MULTIPLE BEARISH SIGNALS ALIGNED - High-conviction short setup',
            'action': 'Quick put spreads with tight stops - scalping opportunity',
            'timeframe': '5-15 minutes',
            'confidence': '85%+'
        })
    
    # Check time-based opportunities
    if current_signals.get('opening_hour', 0) == 1:
        alerts.append({
            'priority': 'MEDIUM',
            'type': 'TIME_OPPORTUNITY',
            'message': 'OPENING HOUR - High volatility period active',
            'action': 'Gap fade/fill strategies, volatility plays - increased opportunity',
            'timeframe': 'Next 30-60 minutes',
            'confidence': '70%+'
        })
    
    if current_signals.get('power_hour', 0) == 1:
        alerts.append({
            'priority': 'MEDIUM',
            'type': 'TIME_OPPORTUNITY',
            'message': 'POWER HOUR - Institutional activity period',
            'action': 'Trend continuation plays, momentum trades',
            'timeframe': 'Next 60 minutes',
            'confidence': '70%+'
        })
    
    if current_signals.get('final_30min', 0) == 1:
        alerts.append({
            'priority': 'HIGH',
            'type': 'GAMMA_RISK',
            'message': 'FINAL 30 MINUTES - 0DTE gamma effects in play',
            'action': 'EXTREME CAUTION - Quick scalps only, tight stops, gamma pin risk',
            'timeframe': 'Until market close',
            'confidence': 'High risk/reward'
        })
    
    # Sort alerts by priority
    priority_order = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
    alerts.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)
    
    return alerts

def create_signal_documentation():
    """Documentation of key signals for 0DTE trading"""
    return {
        "VIX_SIGNALS": {
            "vix_spike_bullish": "VIX spikes 2+ std dev - extreme fear, potential reversal",
            "vix_breakdown": "VIX-SPY correlation weakens - precedes volatility/reversals"
        },
        "MOMENTUM_SIGNALS": {
            "macd_bullish": "MACD histogram crosses positive - momentum shift up",
            "macd_bearish": "MACD histogram crosses negative - momentum shift down"
        },
        "MEAN_REVERSION": {
            "rsi_oversold": "RSI < 30 - oversold condition, potential bounce",
            "bb_mean_revert_bull": "Price bounces from lower Bollinger Band",
            "bb_mean_revert_bear": "Price rejects upper Bollinger Band"
        },
        "VOLATILITY": {
            "bb_compression": "Low volatility, coiling for breakout",
            "vol_expansion": "Short-term vol > long-term vol - momentum building"
        },
        "TIME_BASED": {
            "opening_hour": "9:30-10:30 AM - high volatility period",
            "power_hour": "3:00-4:00 PM - institutional activity",
            "final_30min": "3:30-4:00 PM - 0DTE gamma effects"
        }
    }

def enhanced_statistical_tests(analyzer):
    """Perform statistical tests on signals"""
    if 'SPY' not in analyzer.features or 'signals' not in analyzer.features['SPY']:
        return pd.DataFrame()
    
    signals = analyzer.features['SPY']['signals']
    returns = analyzer.features['SPY']['returns']
    
    if signals.empty or returns.empty:
        return pd.DataFrame()
    
    test_results = []
    
    for signal_name in ['scalp_long', 'scalp_short', 'vix_spike_bullish', 'bb_mean_revert_bull']:
        if signal_name in signals.columns:
            signal_active = signals[signal_name] == 1
            
            if signal_active.sum() > 10:
                forward_returns = returns.shift(-1)
                signal_returns = forward_returns[signal_active].dropna()
                no_signal_returns = forward_returns[~signal_active].dropna()
                
                if len(signal_returns) > 5 and len(no_signal_returns) > 5:
                    try:
                        t_stat, p_value = stats.ttest_ind(signal_returns, no_signal_returns)
                        
                        test_results.append({
                            'signal': signal_name,
                            'horizon': '5min',
                            'signal_mean': signal_returns.mean(),
                            'baseline_mean': no_signal_returns.mean(),
                            'excess_return': signal_returns.mean() - no_signal_returns.mean(),
                            't_statistic': t_stat,
                            't_p_value': p_value,
                            'significant_raw': p_value < 0.05,
                            'signal_count': len(signal_returns),
                            'win_rate': (signal_returns > 0).mean()
                        })
                    except:
                        pass
    
    if test_results:
        df = pd.DataFrame(test_results)
        # Simple multiple comparison correction
        if len(df) > 1:
            try:
                rejected, p_adjusted, _, _ = multipletests(df['t_p_value'], method='fdr_bh', alpha=0.05)
                df['p_adjusted'] = p_adjusted
                df['significant_adjusted'] = rejected
            except:
                df['p_adjusted'] = df['t_p_value']
                df['significant_adjusted'] = df['significant_raw']
        else:
            df['p_adjusted'] = df['t_p_value']
            df['significant_adjusted'] = df['significant_raw']
        return df
    
    return pd.DataFrame()

def backtest_signal_performance(analyzer):
    """Simple backtesting for key signals"""
    if 'SPY' not in analyzer.features or 'signals' not in analyzer.features['SPY']:
        return {}
    
    signals = analyzer.features['SPY']['signals']
    returns = analyzer.features['SPY']['returns']
    
    if signals.empty or returns.empty:
        return {}
    
    backtest_results = {}
    
    for signal_name in ['scalp_long', 'scalp_short']:
        if signal_name in signals.columns:
            signal_active = signals[signal_name] == 1
            total_signals = signal_active.sum()
            
            if total_signals > 5:
                direction = 1 if 'long' in signal_name else -1
                forward_returns = returns.shift(-1) * direction
                trade_returns = forward_returns[signal_active].dropna()
                
                if len(trade_returns) > 0:
                    results_df = pd.DataFrame({
                        '5min': {
                            'total_trades': len(trade_returns),
                            'win_rate': (trade_returns > 0).mean(),
                            'avg_return': trade_returns.mean(),
                            'sharpe_ratio': trade_returns.mean() / trade_returns.std() if trade_returns.std() > 0 else 0,
                            'total_return': (1 + trade_returns).prod() - 1
                        }
                    }).T
                    
                    backtest_results[signal_name] = {
                        'description': f'{signal_name} strategy',
                        'total_signals': total_signals,
                        'signal_frequency': total_signals / len(signals),
                        'results': results_df
                    }
    
    return backtest_results

def main():
    """Main execution function"""
    print("SPY 0DTE Options Scalping Analysis Framework")
    print("=" * 50)
    
    try:
        # Create analyzer and fetch data
        analyzer = MarketStructureAnalyzer()
        analyzer.fetch_data(period='30d', interval='5m')
        
        # Debug: Check data structure
        if 'SPY' in analyzer.data:
            spy_data = analyzer.data['SPY']
            print(f"SPY data shape: {spy_data.shape}")
            print(f"SPY columns: {spy_data.columns.tolist()}")
            print(f"SPY index type: {type(spy_data.index)}")
            print(f"SPY sample data:\n{spy_data.head(2)}")
        
        # Calculate features
        analyzer.calculate_market_structure_features()
        
        # Check if any features were calculated
        if 'SPY' in analyzer.features:
            spy_features = analyzer.features['SPY']
            print(f"SPY features calculated: {list(spy_features.keys())}")
            
            # Check if we have basic returns
            if 'returns' in spy_features:
                returns = spy_features['returns']
                print(f"Returns calculated: {len(returns)} data points, sample: {returns.tail(3).values}")
            else:
                print("WARNING: No returns calculated!")
        
        # Generate signals
        signals = analyzer.calculate_scalping_signals()
        print(f"Generated {len(signals.columns) if not signals.empty else 0} signal types")
        
        if not signals.empty:
            print(f"Signal columns: {signals.columns.tolist()}")
            current_signals = signals.iloc[-1]
            active_signals = current_signals[current_signals > 0]
            print(f"Currently active signals: {len(active_signals)}")
            if len(active_signals) > 0:
                print(f"Active signal names: {active_signals.index.tolist()}")
        
        # Try performance analysis
        try:
            performance = analyzer.analyze_signal_performance()
            if not performance.empty:
                print(f"Performance analysis completed: {len(performance)} results")
            else:
                print("Performance analysis returned empty DataFrame")
        except Exception as e:
            print(f"Warning: Could not analyze performance: {e}")
            performance = pd.DataFrame()
        
        # Try dashboard data
        try:
            dashboard_data = analyzer.create_dashboard_data()
        except Exception as e:
            print(f"Warning: Could not create dashboard data: {e}")
            dashboard_data = None
        
        # Generate additional analysis
        signal_docs = create_signal_documentation()
        alerts = generate_trading_alerts(analyzer)
        
        try:
            enhanced_stats = enhanced_statistical_tests(analyzer)
        except Exception as e:
            print(f"Warning: Could not run statistical tests: {e}")
            enhanced_stats = pd.DataFrame()
        
        try:
            backtest_results = backtest_signal_performance(analyzer)
        except Exception as e:
            print(f"Warning: Could not run backtests: {e}")
            backtest_results = {}
        
        # Display results
        print("\n=== SIGNAL PERFORMANCE ANALYSIS ===")
        if not performance.empty:
            print("Top performing signals:")
            print(performance.head(10).round(4))
        else:
            print("No performance data available")
        
        print("\n=== CURRENT TRADING ALERTS ===")
        if alerts:
            print(f"Found {len(alerts)} alerts:")
            for alert in alerts[:3]:
                print(f"[{alert['priority']}] {alert['type']}: {alert['message']}")
                print(f"   Action: {alert['action']}")
        else:
            print("No alerts at this time")
        
        print("\n=== STATISTICAL SIGNIFICANCE ===")
        if not enhanced_stats.empty:
            significant = enhanced_stats[enhanced_stats.get('significant_adjusted', pd.Series(dtype=bool)) == True]
            if not significant.empty:
                print("Statistically significant signals:")
                available_cols = [col for col in ['signal', 'excess_return', 'win_rate', 'p_adjusted'] if col in significant.columns]
                print(significant[available_cols].round(4))
            else:
                print("No statistically significant signals found")
        else:
            print("No statistical test results available")
        
        print("\n=== SIGNAL SUMMARY ===")
        if not signals.empty:
            current_signals = signals.iloc[-1]
            active_signals = current_signals[current_signals > 0]
            print(f"Currently active signals: {len(active_signals)}")
            for signal_name in active_signals.index:
                print(f"  - {signal_name}")
        else:
            print("No signals generated")
        
        print("\nFramework loaded successfully!")
        print(f"Data points: SPY={len(analyzer.data['SPY']) if 'SPY' in analyzer.data else 0}")
        print(f"Signals generated: {len(signals.columns) if not signals.empty else 0}")
        print(f"Features calculated: {len(analyzer.features.get('SPY', {}))}")
        
        return {
            'analyzer': analyzer,
            'signals': signals,
            'performance': performance,
            'alerts': alerts,
            'enhanced_stats': enhanced_stats,
            'backtest_results': backtest_results,
            'signal_docs': signal_docs,
            'dashboard_data': dashboard_data
        }
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
    if results:
        print("\nAnalysis completed successfully!")
        print("\nTo analyze specific signals, use:")
        print("analyzer = results['analyzer']")
        print("signals = results['signals']")
        print("alerts = results['alerts']")
    else:
        print("\nAnalysis failed - check error messages above")
