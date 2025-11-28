"""AI-based prediction strategy using machine learning."""

import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from kucoin_bot.strategies.base import BaseStrategy, Signal, SignalType
from kucoin_bot.utils.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class AIPredictor(BaseStrategy):
    """AI-based prediction strategy using ensemble learning."""

    def __init__(
        self,
        lookback_period: int = 100,
        prediction_threshold: float = 0.6,
        retrain_interval: int = 1000,
    ):
        """Initialize AI predictor strategy."""
        super().__init__("AIPredictor")
        self.lookback_period = lookback_period
        self.prediction_threshold = prediction_threshold
        self.retrain_interval = retrain_interval

        self._model: RandomForestClassifier | None = None
        self._scaler = StandardScaler()
        self._training_data: list[tuple[list[float], int]] = []
        self._predictions_count = 0
        self._is_trained = False

    def get_required_history_length(self) -> int:
        """Return required history length."""
        return self.lookback_period

    def _extract_features(self, prices: list[float], volumes: list[float]) -> list[float]:
        """Extract features from price and volume data."""
        features = []

        # Price returns at different timeframes
        returns_5 = (prices[-1] - prices[-6]) / prices[-6] if len(prices) > 5 else 0
        returns_10 = (prices[-1] - prices[-11]) / prices[-11] if len(prices) > 10 else 0
        returns_20 = (prices[-1] - prices[-21]) / prices[-21] if len(prices) > 20 else 0

        features.extend([returns_5, returns_10, returns_20])

        # Technical indicators
        if len(prices) >= 14:
            rsi = TechnicalIndicators.rsi(prices, 14)
            features.append(rsi[-1] / 100 if rsi else 0.5)
        else:
            features.append(0.5)

        # MACD features
        if len(prices) >= 26:
            macd_line, signal_line, histogram = TechnicalIndicators.macd(prices)
            if macd_line and signal_line and histogram:
                features.append(macd_line[-1] / prices[-1] if prices[-1] != 0 else 0)
                features.append(histogram[-1] / prices[-1] if prices[-1] != 0 else 0)
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0])

        # Bollinger Band position
        if len(prices) >= 20:
            upper, middle, lower = TechnicalIndicators.bollinger_bands(prices, 20)
            if upper and lower and middle:
                bb_width = (upper[-1] - lower[-1]) / middle[-1] if middle[-1] != 0 else 0
                bb_position = (prices[-1] - lower[-1]) / (upper[-1] - lower[-1]) if upper[-1] != lower[-1] else 0.5
                features.extend([bb_width, bb_position])
            else:
                features.extend([0, 0.5])
        else:
            features.extend([0, 0.5])

        # Volume features
        if len(volumes) >= 20:
            avg_volume = sum(volumes[-20:]) / 20
            volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1
            features.append(min(volume_ratio, 5))  # Cap at 5x
        else:
            features.append(1)

        # Volatility
        if len(prices) >= 20:
            price_std = np.std(prices[-20:])
            volatility = price_std / np.mean(prices[-20:])
            features.append(volatility)
        else:
            features.append(0)

        # Price momentum (rate of change)
        if len(prices) >= 10:
            momentum = (prices[-1] - prices[-10]) / prices[-10]
            features.append(momentum)
        else:
            features.append(0)

        return features

    def _generate_label(self, prices: list[float], horizon: int = 5) -> int:
        """Generate label for training (1: up, -1: down, 0: neutral)."""
        if len(prices) < horizon + 1:
            return 0

        future_return = (prices[-1] - prices[-horizon - 1]) / prices[-horizon - 1]

        if future_return > 0.005:  # 0.5% threshold
            return 1
        elif future_return < -0.005:
            return -1
        return 0

    def _train_model(self) -> None:
        """Train the prediction model."""
        if len(self._training_data) < 100:
            return

        try:
            x_data = np.array([d[0] for d in self._training_data])
            y = np.array([d[1] for d in self._training_data])

            # Filter out neutral labels for clearer signal
            mask = y != 0
            if np.sum(mask) < 50:
                return

            x_data = x_data[mask]
            y = y[mask]

            self._scaler.fit(x_data)
            x_scaled = self._scaler.transform(x_data)

            self._model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,
            )
            self._model.fit(x_scaled, y)
            self._is_trained = True
            logger.info("AI model trained successfully")

        except Exception as e:
            logger.error(f"Model training failed: {e}")

    async def analyze(
        self, symbol: str, prices: list[float], volumes: list[float]
    ) -> Signal | None:
        """Analyze using AI prediction."""
        if len(prices) < self.get_required_history_length():
            return None

        # Extract features
        features = self._extract_features(prices, volumes)

        # Add to training data (with delayed label from past)
        if len(prices) >= self.lookback_period + 5:
            past_features = self._extract_features(
                prices[:-5], volumes[:-5] if len(volumes) > 5 else volumes
            )
            label = self._generate_label(prices)
            self._training_data.append((past_features, label))

            # Keep training data size manageable
            if len(self._training_data) > 10000:
                self._training_data = self._training_data[-5000:]

        # Train or retrain model
        self._predictions_count += 1
        if not self._is_trained or self._predictions_count % self.retrain_interval == 0:
            self._train_model()

        if not self._is_trained or self._model is None:
            return None

        try:
            # Make prediction
            x_data = np.array([features])
            x_scaled = self._scaler.transform(x_data)

            prediction = self._model.predict(x_scaled)[0]
            probabilities = self._model.predict_proba(x_scaled)[0]

            # Get confidence
            max_prob = max(probabilities)

            if max_prob < self.prediction_threshold:
                return None

            current_price = prices[-1]
            signal_type = SignalType.HOLD

            if prediction == 1:
                signal_type = SignalType.LONG
            elif prediction == -1:
                signal_type = SignalType.SHORT
            else:
                return None

            # Calculate stops based on recent volatility
            volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
            stop_distance = current_price * volatility * 2

            if signal_type == SignalType.LONG:
                stop_loss = current_price - stop_distance
                take_profit = current_price + stop_distance * 1.5
            else:
                stop_loss = current_price + stop_distance
                take_profit = current_price - stop_distance * 1.5

            return Signal(
                signal_type=signal_type,
                symbol=symbol,
                confidence=max_prob,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=5,
                reason=f"AI prediction: {signal_type.value}, confidence: {max_prob:.2%}",
            )

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
