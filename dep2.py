import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import threading
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import json
import time
from collections import defaultdict
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import csv

# Selective warning suppression
warnings.filterwarnings('ignore', category=FutureWarning)

# Set font with fallback
try:
    plt.rcParams['font.family'] = 'DejaVu Sans'
except:
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

print("✅ Libraries imported successfully!")

class AdvancedAITrainer:
    """Advanced AI model with deep learning capabilities"""
    
    def __init__(self, model_dir='models', log_dir='logs'):
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.scalers = {}
        self.models = {}
        self.training_logs = {}
        self.training_lock = defaultdict(threading.Lock)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        print("🤖 AI model initialized")
    
    def log_training(self, symbol, epoch, accuracy, loss=None):
        """Log training progress with retry"""
        if symbol not in self.training_logs:
            self.training_logs[symbol] = []
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'accuracy': accuracy,
            'loss': loss
        }
        
        self.training_logs[symbol].append(log_entry)
        
        log_file = f"{self.log_dir}/{symbol}_training.json"
        for attempt in range(2):
            try:
                with open(log_file, 'w') as f:
                    json.dump(self.training_logs[symbol], f, indent=2)
                break
            except Exception as e:
                print(f"⚠️ Error saving log (attempt {attempt+1}/2): {e}")
                time.sleep(0.1)
    
    def prepare_features(self, data):
        """Prepare advanced features"""
        df = data.copy()
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                print(f"❌ Column {col} missing in data")
                return pd.DataFrame()
        
        # Base price features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volatility'] = df['returns'].rolling(20, min_periods=1).std()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)
        
        # MACD
        exp1 = df['Close'].ewm(span=12, min_periods=1).mean()
        exp2 = df['Close'].ewm(span=26, min_periods=1).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, min_periods=1).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20, min_periods=1).mean()
        bb_std = df['Close'].rolling(20, min_periods=1).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(20, min_periods=1).mean()
        df['SMA_50'] = df['Close'].rolling(50, min_periods=1).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, min_periods=1).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, min_periods=1).mean()
        
        df['SMA_Ratio'] = df['SMA_20'] / df['SMA_50']
        df['EMA_Ratio'] = df['EMA_12'] / df['EMA_26']
        
        # Volume
        df['Volume_SMA'] = df['Volume'].rolling(20, min_periods=1).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Volatility
        df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Open_Close_Range'] = (df['Close'] - df['Open']) / df['Open']
        
        # Candlestick
        df['Body_Size'] = (df['Close'] - df['Open']).abs() / df['Open']
        df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Open']
        df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Open']
        df['Is_Green'] = (df['Close'] > df['Open']).astype(int)
        
        df['Doji'] = ((df['High'] - df['Low']) / df['Open'] < 0.01).astype(int)
        df['Hammer'] = (
            (df['Lower_Shadow'] > 2 * df['Body_Size']) & 
            (df['Upper_Shadow'] < df['Body_Size'] * 0.1) &
            (df['Body_Size'] < 0.01)
        ).astype(int)
        
        # Momentum
        df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        
        # New: ATR
        df['TR'] = np.maximum(df['High'] - df['Low'], 
                            np.maximum(abs(df['High'] - df['Close'].shift()), 
                                      abs(df['Low'] - df['Close'].shift())))
        df['ATR'] = df['TR'].rolling(14, min_periods=1).mean()
        
        # Target for classification (3-day future)
        future_return = df['Close'].shift(-3) / df['Close'] - 1
        thresholds = np.percentile(future_return.dropna(), [20, 40, 60, 80])
        df['Target'] = np.where(future_return > thresholds[3], 2,
                               np.where(future_return > thresholds[2], 1,
                                       np.where(future_return < thresholds[0], -2,
                                               np.where(future_return < thresholds[1], -1, 0))))
        
        # Improved NaN handling
        df = df.fillna(0)
        
        class_counts = df['Target'].value_counts()
        print(f"📊 Class distribution: {class_counts.to_dict()}")
        
        return df

    def create_ensemble_model(self):
        """Create advanced ensemble model with hyperparams"""
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        gb = GradientBoostingClassifier(random_state=42)
        mlp = MLPClassifier(random_state=42, early_stopping=True)
        
        return VotingClassifier(estimators=[('rf', rf), ('gb', gb), ('mlp', mlp)], voting='soft')
    
    def train_model(self, symbol, data, force_retrain=False, epochs=1, progress_callback=None):
        """Train model with TimeSeriesSplit, GridSearch, SMOTE"""
        with self.training_lock[symbol]:
            model_path = f"{self.model_dir}/{symbol}_model.joblib"
            scaler_path = f"{self.model_dir}/{symbol}_scaler.joblib"
            
            if not force_retrain and os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.models[symbol] = joblib.load(model_path)
                    self.scalers[symbol] = joblib.load(scaler_path)
                    print(f"✅ Loaded pre-trained model for {symbol}")
                    return True
                except Exception as e:
                    print(f"⚠️ Error loading model: {e}")
            
            df = self.prepare_features(data)
            if len(df) < 100:
                print(f"❌ Insufficient data for {symbol}")
                return False
            
            feature_columns = [
                'returns', 'log_returns', 'volatility', 'RSI', 'MACD', 'MACD_Hist',
                'BB_Width', 'BB_Position', 'SMA_20', 'SMA_50', 'EMA_12', 'Volume_Ratio',
                'High_Low_Range', 'Open_Close_Range', 'Body_Size', 'Upper_Shadow',
                'Lower_Shadow', 'Is_Green', 'Doji', 'Hammer', 'Momentum_5', 'Momentum_10',
                'SMA_Ratio', 'EMA_Ratio', 'Volume_Change', 'ATR'
            ]
            
            available_columns = [col for col in feature_columns if col in df.columns]
            if len(available_columns) < 10:
                print(f"❌ Insufficient features for {symbol}")
                return False
                
            X = df[available_columns].values
            y = df['Target'].values
            
            unique, counts = np.unique(y, return_counts=True)
            print(f"🎯 {symbol} class distribution: {dict(zip(unique, counts))}")
            
            tscv = TimeSeriesSplit(n_splits=5)
            self.scalers[symbol] = StandardScaler()
            
            param_grid = {
                'rf__n_estimators': [50, 100],
                'gb__n_estimators': [50, 80],
                'mlp__hidden_layer_sizes': [(32,), (64,32)]
            }
            model = self.create_ensemble_model()
            grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
            
            best_score = 0
            best_model = None
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                smote = SMOTE(random_state=42)
                try:
                    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
                except:
                    X_train_res, y_train_res = X_train, y_train
                
                X_train_scaled = self.scalers[symbol].fit_transform(X_train_res)
                X_test_scaled = self.scalers[symbol].transform(X_test)
                
                grid_search.fit(X_train_scaled, y_train_res)
                
                train_score = grid_search.score(X_train_scaled, y_train_res)
                test_score = grid_search.score(X_test_scaled, y_test)
                
                if test_score > best_score:
                    best_score = test_score
                    best_model = grid_search.best_estimator_
                
                self.log_training(symbol, 1, test_score)
                
                print(f"🎯 {symbol} - CV Fold - Train Acc: {train_score:.2%}, Test Acc: {test_score:.2%}")
                
                if progress_callback:
                    progress_callback(symbol, 1, 1, train_score, test_score)
            
            if best_model is not None:
                self.models[symbol] = best_model
            
            print(f"✅ Trained {symbol} - Best test acc: {best_score:.2%}")
            
            if best_score > 0.4:
                joblib.dump(self.models[symbol], model_path)
                joblib.dump(self.scalers[symbol], scaler_path)
                return True
            else:
                print(f"❌ Low accuracy for {symbol} - Model not saved")
                return False
    
    def train_lstm(self, symbol, data, epochs=10):
        """Train LSTM model for time-series"""
        df = self.prepare_features(data)
        feature_columns = [
            'returns', 'log_returns', 'volatility', 'RSI', 'MACD', 'MACD_Hist',
            'BB_Width', 'BB_Position', 'SMA_20', 'SMA_50', 'EMA_12', 'Volume_Ratio',
            'High_Low_Range', 'Open_Close_Range', 'Body_Size', 'Upper_Shadow',
            'Lower_Shadow', 'Is_Green', 'Doji', 'Hammer', 'Momentum_5', 'Momentum_10',
            'SMA_Ratio', 'EMA_Ratio', 'Volume_Change', 'ATR'
        ]
        available_columns = [col for col in feature_columns if col in df.columns]
        X = df[available_columns].values
        y = pd.get_dummies(df['Target']).values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Reshape for LSTM [samples, timesteps, features]
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)
            
            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                return out
        
        model = LSTMModel(input_size=X_scaled.shape[2], hidden_size=50, num_layers=1, output_size=5)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        dataset = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        model.train()
        for epoch in range(epochs):
            for inputs, labels in loader:
                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
            
            print(f"LSTM Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
        
        torch.save(model.state_dict(), f"{self.model_dir}/{symbol}_lstm.pt")
        joblib.dump(scaler, f"{self.model_dir}/{symbol}_lstm_scaler.joblib")
        return True
    
    def predict(self, symbol, data):
        """Predict using ensemble model"""
        if symbol not in self.models:
            return {'error': 'Model not trained for this currency'}
        
        df = self.prepare_features(data)
        if len(df) < 20:
            return {'error': 'Insufficient data for prediction'}
        
        feature_columns = [
            'returns', 'log_returns', 'volatility', 'RSI', 'MACD', 'MACD_Hist',
            'BB_Width', 'BB_Position', 'SMA_20', 'SMA_50', 'EMA_12', 'Volume_Ratio',
            'High_Low_Range', 'Open_Close_Range', 'Body_Size', 'Upper_Shadow',
            'Lower_Shadow', 'Is_Green', 'Doji', 'Hammer', 'Momentum_5', 'Momentum_10',
            'SMA_Ratio', 'EMA_Ratio', 'Volume_Change', 'ATR'
        ]
        
        available_columns = [col for col in feature_columns if col in df.columns]
        if len(available_columns) == 0:
            return {'error': 'No features available for prediction'}
        
        X = df[available_columns].iloc[-1:].values
        
        try:
            X_scaled = self.scalers[symbol].transform(X)
        except Exception as e:
            return {'error': f'Error in scaling data: {str(e)}'}
        
        prediction = self.models[symbol].predict(X_scaled)[0]
        probabilities = self.models[symbol].predict_proba(X_scaled)[0]
        
        action_map = {
            -2: 'STRONG_SELL', -1: 'SELL', 
            0: 'HOLD', 
            1: 'BUY', 2: 'STRONG_BUY'
        }
        
        confidence = np.max(probabilities)
        
        buy_signals, sell_signals = self.detect_trading_signals(df)
        
        return {
            'action': action_map[prediction],
            'confidence': confidence,
            'probabilities': probabilities.tolist(),
            'timestamp': datetime.now(),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'technical_indicators': self.get_current_indicators(df)
        }
    
    def detect_trading_signals(self, df):
        """Detect buy/sell signals"""
        if len(df) < 30:
            return [], []
        
        buy_signals = []
        sell_signals = []
        
        current = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else current
        
        if 'RSI' in df.columns:
            if current['RSI'] < 30 and prev['RSI'] >= 30:
                buy_signals.append(('RSI Oversold', current['RSI']))
            elif current['RSI'] > 70 and prev['RSI'] <= 70:
                sell_signals.append(('RSI Overbought', current['RSI']))
        
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            if current['MACD'] > current['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
                buy_signals.append(('MACD Bullish Crossover', current['MACD']))
            elif current['MACD'] < current['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
                sell_signals.append(('MACD Bearish Crossover', current['MACD']))
        
        if all(col in df.columns for col in ['Close', 'BB_Upper', 'BB_Lower']):
            if current['Close'] < current['BB_Lower']:
                buy_signals.append(('Below Lower Bollinger Band', current['Close']))
            elif current['Close'] > current['BB_Upper']:
                sell_signals.append(('Above Upper Bollinger Band', current['Close']))
        
        if 'Hammer' in df.columns:
            if current['Hammer'] == 1:
                buy_signals.append(('Hammer Pattern', current['Close']))
        
        if 'Momentum_5' in df.columns:
            if current['Momentum_5'] > 0.02:
                buy_signals.append(('Positive Momentum', current['Momentum_5']))
            elif current['Momentum_5'] < -0.02:
                sell_signals.append(('Negative Momentum', current['Momentum_5']))
        
        return buy_signals, sell_signals
    
    def get_current_indicators(self, df):
        """Get current indicator values"""
        if len(df) == 0:
            return {}
        
        current = df.iloc[-1]
        indicators = {}
        
        indicator_columns = {
            'RSI': 'RSI', 'MACD': 'MACD', 'MACD_Signal': 'MACD_Signal',
            'BB_Upper': 'BB_Upper', 'BB_Lower': 'BB_Lower', 'BB_Middle': 'BB_Middle',
            'SMA_20': 'SMA_20', 'SMA_50': 'SMA_50', 'EMA_12': 'EMA_12',
            'Volume_Ratio': 'Volume_Ratio', 'Momentum_5': 'Momentum_5', 'ATR': 'ATR'
        }
        
        for key, column in indicator_columns.items():
            if column in df.columns:
                indicators[key] = current[column]
        
        return indicators

class TradingBot:
    """Intelligent trading bot"""
    
    def __init__(self):
        self.ai_trainer = AdvancedAITrainer()
        self.supported_pairs = {
            'BTC-USD': 'Bitcoin',
            'ETH-USD': 'Ethereum', 
            'ADA-USD': 'Cardano',
            'DOT-USD': 'Polkadot',
            'LINK-USD': 'Chainlink',
            'BNB-USD': 'Binance Coin',
            'XRP-USD': 'Ripple', 
            'DOGE-USD': 'Dogecoin',
            'SOL-USD': 'Solana',
            'MATIC-USD': 'Polygon'
        }
        self.analysis_lock = threading.Lock()
        print("🤖 Trading bot initialized!")
    
    def get_historical_data(self, symbol, period='6mo', interval='1d'):
        """Fetch historical data with retry"""
        for attempt in range(3):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                if not data.empty:
                    print(f"✅ Fetched data for {symbol} ({len(data)} records)")
                    return data
                time.sleep(2)
            except Exception as e:
                print(f"⚠️ Retry {attempt+1}/3 for {symbol}: {e}")
        print(f"❌ Failed to fetch data for {symbol}")
        return pd.DataFrame()
    
    def technical_analysis(self, data):
        """Advanced technical analysis"""
        if len(data) < 20:
            return {'error': 'Insufficient data for technical analysis'}
        
        df = data.copy()
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                return {'error': f'Column {col} missing in data'}
        
        current_price = df['Close'].iloc[-1]
        
        try:
            df['SMA_20'] = df['Close'].rolling(20, min_periods=1).mean()
            df['SMA_50'] = df['Close'].rolling(50, min_periods=1).mean()
            df['EMA_12'] = df['Close'].ewm(span=12, min_periods=1).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, min_periods=1).mean()
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            df['RSI'] = df['RSI'].fillna(50)
            
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, min_periods=1).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            df['BB_Middle'] = df['Close'].rolling(20, min_periods=1).mean()
            bb_std = df['Close'].rolling(20, min_periods=1).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            df = df.fillna(0)
            
        except Exception as e:
            return {'error': f'Error calculating indicators: {str(e)}'}
        
        current = df.iloc[-1]
        
        return {
            'current_price': current_price,
            'price_change_24h': ((current_price - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100) if len(df) > 1 else 0,
            'sma_20': current.get('SMA_20', 0),
            'sma_50': current.get('SMA_50', 0),
            'rsi': current.get('RSI', 50),
            'macd': current.get('MACD', 0),
            'macd_signal': current.get('MACD_Signal', 0),
            'macd_hist': current.get('MACD_Hist', 0),
            'bb_upper': current.get('BB_Upper', 0),
            'bb_lower': current.get('BB_Lower', 0),
            'bb_middle': current.get('BB_Middle', 0),
            'support_level': df['Low'].rolling(20, min_periods=1).min().iloc[-1],
            'resistance_level': df['High'].rolling(20, min_periods=1).max().iloc[-1],
            'volume': current['Volume'],
            'volume_change': ((current['Volume'] - df['Volume'].iloc[-2]) / df['Volume'].iloc[-2] * 100) if len(df) > 1 else 0,
            'trend': 'BULLISH' if current.get('SMA_20', 0) > current.get('SMA_50', 0) else 'BEARISH'
        }
    
    def analyze_symbol(self, symbol):
        """Comprehensive analysis of a currency"""
        with self.analysis_lock:
            print(f"🔍 Starting analysis for {symbol}...")
            
            data = self.get_historical_data(symbol, '6mo', '1d')
            if data.empty:
                return {'error': 'No data received'}
            
            ta_result = self.technical_analysis(data)
            if 'error' in ta_result:
                return {'error': ta_result['error']}
            
            ai_result = self.ai_trainer.predict(symbol, data)
            
            result = {
                'symbol': symbol,
                'name': self.supported_pairs.get(symbol, symbol),
                'timestamp': datetime.now(),
                'technical_analysis': ta_result,
                'ai_prediction': ai_result,
                'price_data': data,
                'success': True
            }
            
            print(f"✅ Analysis for {symbol} completed")
            return result
    
    def train_models(self, symbols=None, epochs=1, progress_callback=None):
        """Train AI models"""
        if symbols is None:
            symbols = list(self.supported_pairs.keys())
        
        print(f"🤖 Starting training for {len(symbols)} models...")
        
        successful = 0
        for i, symbol in enumerate(symbols):
            print(f"🔧 Training {symbol}...")
            data = self.get_historical_data(symbol, '2y', '1d')
            
            if len(data) > 200:
                if self.ai_trainer.train_model(symbol, data, force_retrain=True, epochs=epochs, progress_callback=progress_callback):
                    successful += 1
                    print(f"✅ {symbol} - Training successful")
                    self.ai_trainer.train_lstm(symbol, data, epochs=5)
                else:
                    print(f"❌ {symbol} - Training failed")
            else:
                print(f"⚠️ {symbol} - Insufficient data")
        
        print(f"🎯 Training completed: {successful}/{len(symbols)} successful")
        return successful
    
    def backtest(self, symbol, data):
        """Backtest trading strategy"""
        df = self.ai_trainer.prepare_features(data)
        predictions = []
        for i in range(20, len(df)):
            sub_df = df.iloc[:i]
            pred = self.ai_trainer.predict(symbol, sub_df)
            predictions.append(pred['action'])
        
        df['Predicted'] = [None] * 20 + predictions
        df['Position'] = np.where(df['Predicted'] == 'BUY', 1, 
                                np.where(df['Predicted'] == 'SELL', -1, 
                                        np.where(df['Predicted'] == 'STRONG_BUY', 2, 
                                                np.where(df['Predicted'] == 'STRONG_SELL', -2, 0))))
        df['Strategy_Return'] = df['Position'].shift(1) * df['returns']
        total_return = df['Strategy_Return'].sum()
        print(f"📊 Backtest return for {symbol}: {total_return:.2%}")
        return total_return
    
    def create_chart(self, symbol, data, analysis_result, ai_prediction):
        """Create professional chart with historical signals"""
        if len(data) < 30:
            return None
        
        try:
            fig = Figure(figsize=(12, 10))
            df = self.ai_trainer.prepare_features(data.copy())
            
            ax1 = fig.add_subplot(3, 1, 1)
            ax1.plot(df.index, df['Close'], label='Close Price', linewidth=1.5, color='blue')
            ax1.plot(df.index, df['SMA_20'], label='SMA 20', linewidth=1, color='orange', alpha=0.7)
            ax1.plot(df.index, df['SMA_50'], label='SMA 50', linewidth=1, color='red', alpha=0.7)
            ax1.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], alpha=0.2, label='Bollinger Bands', color='gray')
            
            # Historical signals
            buy_dates = df.index[df['Hammer'] == 1][-5:]
            ax1.scatter(buy_dates, df.loc[buy_dates]['Close'], color='green', s=50, marker='^', label='Historical Buy')
            
            if ai_prediction.get('buy_signals'):
                last_buy_date = df.index[-1]
                ax1.scatter(last_buy_date, df['Close'].iloc[-1], color='green', s=100, marker='^')
                ax1.annotate('BUY', (last_buy_date, df['Close'].iloc[-1]), 
                            textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')
            
            if ai_prediction.get('sell_signals'):
                last_sell_date = df.index[-1]
                ax1.scatter(last_sell_date, df['Close'].iloc[-1], color='red', s=100, marker='v')
                ax1.annotate('SELL', (last_sell_date, df['Close'].iloc[-1]), 
                            textcoords="offset points", xytext=(0,-15), ha='center', fontweight='bold')
            
            ax1.set_title(f'{symbol} - Price Chart with Trading Signals', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2 = fig.add_subplot(3, 1, 2)
            ax2.plot(df.index, df['RSI'], label='RSI', linewidth=1.5, color='purple')
            ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
            ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
            ax2.set_title('RSI Indicator', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 100)
            
            ax3 = fig.add_subplot(3, 1, 3)
            ax3.plot(df.index, df['MACD'], label='MACD', linewidth=1.5, color='blue')
            ax3.plot(df.index, df['MACD_Signal'], label='Signal', linewidth=1.5, color='red')
            ax3.bar(df.index, df['MACD_Hist'], label='Histogram', alpha=0.3, color='gray')
            ax3.set_title('MACD Indicator', fontsize=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            fig.tight_layout()
            return fig
        except Exception as e:
            print(f"❌ Error creating chart: {e}")
            return None

class ProfessionalTradingGUI:
    """Professional GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Intelligent Trading Bot - Pro Version")
        self.root.geometry("1400x900")
        
        self.bot = TradingBot()
        self.current_chart = None
        self.live_timer = None
        self.setup_gui()
        
        self.start_initial_training()
    
    def setup_gui(self):
        """Setup GUI"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text='🔍 Real-time Analysis')
        self.setup_analysis_tab()
        
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text='🔧 Model Training')
        self.setup_training_tab()
        
        self.chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.chart_frame, text='📊 Professional Chart')
        self.setup_chart_tab()
        
        self.status_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.status_frame, text='📈 System Status')
        self.setup_status_tab()
    
    def setup_analysis_tab(self):
        """Setup analysis tab"""
        main_frame = ttk.Frame(self.analysis_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        control_frame = ttk.LabelFrame(main_frame, text="Analysis Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(control_frame, text="Select Currency:", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5, pady=5)
        
        self.symbol_var = tk.StringVar(value='BTC-USD')
        symbol_combo = ttk.Combobox(control_frame, textvariable=self.symbol_var,
                                   values=list(self.bot.supported_pairs.keys()), 
                                   state="readonly", width=15)
        symbol_combo.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Analyze", 
                  command=self.run_analysis).grid(row=0, column=2, padx=10, pady=5)
        
        ttk.Button(control_frame, text="Analyze All", 
                  command=self.analyze_all).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Show Chart", 
                  command=self.show_chart).grid(row=0, column=4, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Start Live Monitoring", 
                  command=self.start_live).grid(row=0, column=5, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Backtest", 
                  command=self.run_backtest).grid(row=0, column=6, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Export to CSV", 
                  command=self.export_results).grid(row=0, column=7, padx=5, pady=5)
        
        self.results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding=10)
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = scrolledtext.ScrolledText(self.results_frame, 
                                                     font=('Consolas', 10),
                                                     wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)
    
    def setup_training_tab(self):
        """Setup training tab"""
        main_frame = ttk.Frame(self.training_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        control_frame = ttk.LabelFrame(main_frame, text="Training Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(control_frame, text="Training Epochs:").grid(row=0, column=0, padx=5, pady=5)
        
        self.epochs_var = tk.StringVar(value="50")
        epochs_entry = ttk.Entry(control_frame, textvariable=self.epochs_var, width=10)
        epochs_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Train Selected", 
                  command=self.train_selected).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Train All Models", 
                  command=self.train_all_models).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Long Training (3h)", 
                  command=self.long_training).grid(row=0, column=4, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Clear Models", 
                  command=self.clear_models).grid(row=0, column=5, padx=5, pady=5)
        
        self.progress_frame = ttk.Frame(control_frame)
        self.progress_frame.grid(row=1, column=0, columnspan=6, sticky='ew', pady=5)
        
        ttk.Label(self.progress_frame, text="Progress:").pack(side=tk.LEFT, padx=5)
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.progress_label = ttk.Label(self.progress_frame, text="0%")
        self.progress_label.pack(side=tk.LEFT, padx=5)
        
        self.training_status = ttk.LabelFrame(main_frame, text="Training Status", padding=10)
        self.training_status.pack(fill=tk.BOTH, expand=True)
        
        self.training_text = scrolledtext.ScrolledText(self.training_status,
                                                      font=('Consolas', 9),
                                                      wrap=tk.WORD)
        self.training_text.pack(fill=tk.BOTH, expand=True)
    
    def setup_chart_tab(self):
        """Setup chart tab"""
        main_frame = ttk.Frame(self.chart_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        control_frame = ttk.LabelFrame(main_frame, text="Chart Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(control_frame, text="Select Currency:").grid(row=0, column=0, padx=5, pady=5)
        
        self.chart_symbol_var = tk.StringVar(value='BTC-USD')
        chart_symbol_combo = ttk.Combobox(control_frame, textvariable=self.chart_symbol_var,
                                         values=list(self.bot.supported_pairs.keys()),
                                         state="readonly", width=15)
        chart_symbol_combo.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Load Chart", 
                  command=self.load_chart).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Save Chart", 
                  command=self.save_chart).grid(row=0, column=3, padx=5, pady=5)
        
        self.chart_container = ttk.Frame(main_frame)
        self.chart_container.pack(fill=tk.BOTH, expand=True)
        
        self.chart_placeholder = ttk.Label(self.chart_container, 
                                          text="Analyze a currency to view chart",
                                          font=('Arial', 12))
        self.chart_placeholder.pack(expand=True)
    
    def setup_status_tab(self):
        """Setup status tab"""
        main_frame = ttk.Frame(self.status_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        info_frame = ttk.LabelFrame(main_frame, text="System Info", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        info_text = f"""
🤖 Intelligent Trading Bot - Pro Version

📊 New Features:
• Deep learning with LSTM
• Backtesting capabilities
• Live monitoring every 5 minutes
• Advanced indicators (ATR, RSI, MACD, Bollinger Bands)
• Class imbalance handling with SMOTE
• Hyperparameter tuning with GridSearchCV

🎯 Supported Currencies:
{chr(10).join([f'• {sym} - {name}' for sym, name in self.bot.supported_pairs.items()])}

🛠️ Technologies:
• Scikit-Learn for ensemble AI
• PyTorch for LSTM
• Yahoo Finance for real-time data
• Matplotlib for charts
• Joblib for model persistence

📈 Supported Indicators:
• RSI, MACD, Bollinger Bands, ATR
• Moving Averages (SMA, EMA)
• Volume, Candlestick Patterns (Hammer, Doji)

✅ Status: System ready with advanced features
"""
        
        status_text = scrolledtext.ScrolledText(info_frame, font=('Arial', 10), wrap=tk.WORD)
        status_text.pack(fill=tk.BOTH, expand=True)
        status_text.insert(tk.END, info_text)
        status_text.config(state=tk.DISABLED)
    
    def start_initial_training(self):
        """Start initial training in background"""
        self.log_training("🔧 Starting initial model training in background...")
        
        def train_background():
            successful = self.bot.train_models(['BTC-USD'], epochs=5)
            self.log_training(f"✅ Initial training completed: {successful}/1 models")
        
        thread = threading.Thread(target=train_background, daemon=True)
        thread.start()
    
    def run_analysis(self):
        """Run analysis for selected currency"""
        symbol = self.symbol_var.get()
        self.log_result(f"🔍 Starting analysis for {symbol}...")
        
        def analyze():
            try:
                result = self.bot.analyze_symbol(symbol)
                self.display_analysis_result(result)
                
                self.current_chart = self.bot.create_chart(
                    symbol, 
                    result['price_data'], 
                    result['technical_analysis'], 
                    result['ai_prediction']
                )
                
            except Exception as e:
                self.log_result(f"❌ Analysis error: {str(e)}")
        
        thread = threading.Thread(target=analyze, daemon=True)
        thread.start()
    
    def analyze_all(self):
        """Analyze all currencies"""
        self.log_result("🔍 Starting analysis for all currencies...")
        
        def analyze_all_thread():
            results = []
            for symbol in self.bot.supported_pairs.keys():
                try:
                    result = self.bot.analyze_symbol(symbol)
                    results.append(result)
                    self.log_result(f"✅ {symbol} - Analysis completed")
                except Exception as e:
                    self.log_result(f"❌ {symbol} - Error: {str(e)}")
            
            self.display_summary(results)
        
        thread = threading.Thread(target=analyze_all_thread, daemon=True)
        thread.start()
    
    def train_selected(self):
        """Train selected model"""
        symbol = self.symbol_var.get()
        try:
            epochs = int(self.epochs_var.get())
        except:
            epochs = 50
        
        self.log_training(f"🔧 Training model for {symbol} for {epochs} epochs...")
        
        def train_thread():
            try:
                data = self.bot.get_historical_data(symbol, '2y', '1d')
                success = self.bot.ai_trainer.train_model(
                    symbol, data, True, epochs=epochs,
                    progress_callback=self.update_progress
                )
                
                if success:
                    self.log_training(f"✅ Model for {symbol} trained successfully")
                else:
                    self.log_training(f"❌ Error training model for {symbol}")
            except Exception as e:
                self.log_training(f"❌ Training error: {str(e)}")
            
            self.root.after(0, lambda: self.progress_bar.config(value=0))
            self.root.after(0, lambda: self.progress_label.config(text="0%"))
        
        thread = threading.Thread(target=train_thread, daemon=True)
        thread.start()
    
    def train_all_models(self):
        """Train all models"""
        try:
            epochs = int(self.epochs_var.get())
        except:
            epochs = 10
        
        self.log_training(f"🔧 Starting training for all models for {epochs} epochs...")
        
        def train_all_thread():
            successful = self.bot.train_models(epochs=epochs, progress_callback=self.update_progress)
            self.log_training(f"🎯 Training completed: {successful}/{len(self.bot.supported_pairs)} successful")
            
            self.root.after(0, lambda: self.progress_bar.config(value=0))
            self.root.after(0, lambda: self.progress_label.config(text="0%"))
        
        thread = threading.Thread(target=train_all_thread, daemon=True)
        thread.start()
    
    def long_training(self):
        """Long training (3 hours)"""
        self.log_training("🔧 Starting long training (3 hours)...")
        
        def long_train_thread():
            start_time = time.time()
            end_time = start_time + 10800
            symbols = list(self.bot.supported_pairs.keys())
            current_symbol_index = 0
            
            while time.time() < end_time and current_symbol_index < len(symbols):
                symbol = symbols[current_symbol_index]
                self.log_training(f"🔧 Training {symbol}...")
                data = self.bot.get_historical_data(symbol, '2y', '1d')
                
                if len(data) > 200:
                    self.bot.ai_trainer.train_model(
                        symbol, data, True, epochs=10,
                        progress_callback=self.update_progress
                    )
                    self.bot.ai_trainer.train_lstm(symbol, data, epochs=5)
                    self.log_training(f"✅ {symbol} - Training completed")
                
                current_symbol_index += 1
                if current_symbol_index >= len(symbols):
                    current_symbol_index = 0
                    self.log_training("🔄 Starting new training cycle...")
            
            self.log_training("✅ Long training completed")
            self.root.after(0, lambda: self.progress_bar.config(value=0))
            self.root.after(0, lambda: self.progress_label.config(text="0%"))
        
        thread = threading.Thread(target=long_train_thread, daemon=True)
        thread.start()
    
    def update_progress(self, symbol, epoch, total_epochs, train_score, test_score):
        """Update progress bar"""
        progress = (epoch / total_epochs) * 100
        
        def update():
            self.progress_bar.config(value=progress)
            self.progress_label.config(text=f"{progress:.1f}%")
            self.log_training(f"📊 {symbol} - Epoch {epoch}/{total_epochs} - Acc: {test_score:.2%}")
        
        self.root.after(0, update)
    
    def clear_models(self):
        """Clear saved models"""
        try:
            import shutil
            if os.path.exists('models'):
                shutil.rmtree('models')
                os.makedirs('models', exist_ok=True)
            if os.path.exists('logs'):
                shutil.rmtree('logs')
                os.makedirs('logs', exist_ok=True)
            self.bot.ai_trainer.models.clear()
            self.bot.ai_trainer.scalers.clear()
            self.bot.ai_trainer.training_logs.clear()
            self.log_training("✅ All models and logs cleared")
        except Exception as e:
            self.log_training(f"❌ Error clearing models: {str(e)}")
    
    def start_live(self):
        """Start live monitoring every 5 minutes"""
        symbol = self.symbol_var.get()
        self.log_result(f"🔄 Starting live monitoring for {symbol} every 5 min...")
        
        def live_thread():
            self.run_analysis()
            self.live_timer = threading.Timer(300, self.start_live)
            self.live_timer.start()
        
        if self.live_timer:
            self.live_timer.cancel()
        thread = threading.Thread(target=live_thread, daemon=True)
        thread.start()
    
    def run_backtest(self):
        """Run backtest for selected currency"""
        symbol = self.symbol_var.get()
        self.log_result(f"📊 Starting backtest for {symbol}...")
        
        def backtest_thread():
            try:
                data = self.bot.get_historical_data(symbol, '2y', '1d')
                if not data.empty:
                    total_return = self.bot.backtest(symbol, data)
                    self.log_result(f"📊 Backtest result for {symbol}: {total_return:.2%} return")
                else:
                    self.log_result(f"❌ No data for backtest")
            except Exception as e:
                self.log_result(f"❌ Backtest error: {str(e)}")
        
        thread = threading.Thread(target=backtest_thread, daemon=True)
        thread.start()
    
    def export_results(self):
        """Export analysis results to CSV"""
        symbol = self.symbol_var.get()
        result = self.bot.analyze_symbol(symbol)
        if not result.get('success', False):
            messagebox.showerror("Error", f"Cannot export: {result.get('error', 'Unknown error')}")
            return
        
        ta = result['technical_analysis']
        ai = result['ai_prediction']
        filename = f"analysis_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Field', 'Value'])
            writer.writerow(['Symbol', symbol])
            writer.writerow(['Name', result['name']])
            writer.writerow(['Timestamp', result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')])
            writer.writerow(['Current Price', ta['current_price']])
            writer.writerow(['Price Change 24h', ta['price_change_24h']])
            writer.writerow(['RSI', ta['rsi']])
            writer.writerow(['MACD', ta['macd']])
            writer.writerow(['Prediction', ai.get('action', 'N/A')])
            writer.writerow(['Confidence', ai.get('confidence', 0)])
            writer.writerow(['Probabilities', str(ai.get('probabilities', []))])
        
        messagebox.showinfo("Success", f"Results exported to {filename}")
    
    def show_chart(self):
        """Show chart tab"""
        self.notebook.select(2)
        self.load_chart()
    
    def load_chart(self):
        """Load chart in chart tab"""
        if self.current_chart is None:
            messagebox.showwarning("Warning", "Analyze a currency first")
            return
        
        for widget in self.chart_container.winfo_children():
            widget.destroy()
        
        canvas = FigureCanvasTkAgg(self.current_chart, self.chart_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def save_chart(self):
        """Save chart"""
        if self.current_chart is None:
            messagebox.showwarning("Warning", "No chart to save")
            return
        
        try:
            filename = f"chart_{self.symbol_var.get()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.current_chart.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Chart saved as {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving chart: {str(e)}")
    
    def display_analysis_result(self, result):
        """Display analysis result"""
        if not result.get('success', False):
            self.log_result(f"❌ Error: {result.get('error', 'Unknown error')}")
            return
        
        symbol = result['symbol']
        name = result['name']
        ta = result['technical_analysis']
        ai = result['ai_prediction']
        
        output = f"""
🎯 Professional Analysis for {name} ({symbol})
{'='*70}

📊 Price Info:
• Current Price: ${ta['current_price']:,.2f}
• 24h Change: {ta['price_change_24h']:+.2f}%
• Volume: {ta['volume']:,.0f}
• Trend: {ta['trend']}

📈 Technical Analysis:
• RSI: {ta['rsi']:.1f} ({'Overbought' if ta['rsi'] > 70 else 'Oversold' if ta['rsi'] < 30 else 'Normal'})
• MACD: {ta['macd']:.4f} | Signal: {ta['macd_signal']:.4f}
• SMA 20: ${ta['sma_20']:.2f}
• SMA 50: ${ta['sma_50']:.2f}
• Support: ${ta['support_level']:.2f}
• Resistance: ${ta['resistance_level']:.2f}

🤖 AI Prediction:
• Signal: {ai.get('action', 'N/A')}
• Confidence: {ai.get('confidence', 0):.2%}
• Probabilities: {ai.get('probabilities', [])}
• Time: {ai.get('timestamp', 'N/A')}

🎯 Identified Signals:
"""
        
        buy_signals = ai.get('buy_signals', [])
        if buy_signals:
            output += "\n🟢 Buy Signals:\n"
            for signal, value in buy_signals:
                output += f"   • {signal}: {value:.4f}\n"
        else:
            output += "\n🟢 No strong buy signals\n"
        
        sell_signals = ai.get('sell_signals', [])
        if sell_signals:
            output += "\n🔴 Sell Signals:\n"
            for signal, value in sell_signals:
                output += f"   • {signal}: {value:.4f}\n"
        else:
            output += "\n🔴 No strong sell signals\n"

        indicators = ai.get('technical_indicators', {})
        if indicators:
            output += f"""
📊 Current Indicators:
• RSI: {indicators.get('RSI', 0):.1f}
• MACD: {indicators.get('MACD', 0):.4f}
• Bollinger Upper: ${indicators.get('BB_Upper', 0):.2f}
• Bollinger Lower: ${indicators.get('BB_Lower', 0):.2f}
• Volume Ratio: {indicators.get('Volume_Ratio', 0):.2f}
• ATR: {indicators.get('ATR', 0):.2f}
"""
        
        output += f"""
💡 Final Recommendation:
"""
        
        action = ai.get('action', 'HOLD')
        confidence = ai.get('confidence', 0)
        
        if 'STRONG_BUY' in action and confidence > 0.7:
            output += "🟢 Strong Buy - Ideal entry conditions"
        elif 'BUY' in action and confidence > 0.6:
            output += "🟢 Buy - Good entry conditions"
        elif 'STRONG_SELL' in action and confidence > 0.7:
            output += "🔴 Strong Sell - Consider exiting"
        elif 'SELL' in action and confidence > 0.6:
            output += "🔴 Sell - Consider reducing exposure"
        else:
            output += "⚪ Neutral - Wait for clearer signals"
        
        output += f"\n\n🕒 Analysis Time: {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
        output += f"\n{'='*70}"
        
        self.log_result(output)
    
    def display_summary(self, results):
        """Display summary of all analyses"""
        successful = [r for r in results if r.get('success', False)]
        
        output = f"""
📊 Summary of All Analyses
{'='*50}
Currencies Analyzed: {len(successful)}/{len(results)}

🔍 Strong Signals:
"""
        
        strong_signals = []
        for result in successful:
            ai = result['ai_prediction']
            if 'STRONG' in ai.get('action', '') and ai.get('confidence', 0) > 0.7:
                strong_signals.append(
                    f"• {result['name']}: {ai['action']} (Confidence: {ai['confidence']:.2%})"
                )
        
        if strong_signals:
            output += '\n'.join(strong_signals)
        else:
            output += "• No strong signals identified"
        
        output += f"\n\n💡 Top Opportunities:\n"
        
        sorted_results = sorted(successful, 
                               key=lambda x: x['ai_prediction'].get('confidence', 0), 
                               reverse=True)
        
        for i, result in enumerate(sorted_results[:3]):
            ai = result['ai_prediction']
            output += f"{i+1}. {result['name']}: {ai['action']} ({ai['confidence']:.2%})\n"
        
        output += f"\n{'='*50}"
        self.log_result(output)
    
    def log_result(self, message):
        """Log message to results"""
        def update():
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, message)
            self.results_text.see(tk.END)
        
        self.root.after(0, update)
    
    def log_training(self, message):
        """Log message to training"""
        def update():
            self.training_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')} - {message}\n")
            self.training_text.see(tk.END)
        
        self.root.after(0, update)

def main():
    """Main function"""
    try:
        print("🚀 Starting trading bot...")
        root = tk.Tk()
        style = ttk.Style()
        style.theme_use('clam')
        app = ProfessionalTradingGUI(root)
        print("✅ Bot ready!")
        print("📱 Displaying GUI...")
        root.mainloop()
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to close...")

if __name__ == "__main__":
    main()
