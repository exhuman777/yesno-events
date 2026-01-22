#!/usr/bin/env python3
"""
Polymarket Real-Time Data Streaming (RTDS) Client
==================================================
Python implementation of the Polymarket WebSocket client.
Based on: https://github.com/Polymarket/real-time-data-client

Topics:
- activity: trades, orders_matched
- comments: comment_created, comment_removed, reaction_created, reaction_removed
- crypto_prices: update (BTCUSDT, ETHUSDT, SOLUSDT, etc.)
- equity_prices: update (AAPL, TSLA, NVDA, etc.)
- rfq: request_created, quote_created, etc.
"""

import json
import threading
import time
from typing import Callable, Optional, Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False
    print("Warning: websocket-client not installed. Run: pip install websocket-client")


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Subscription:
    """Subscription request structure"""
    topic: str
    type: str = "*"  # "*" for all types
    filters: str = ""  # JSON string for filters
    clob_auth: Optional[Dict[str, str]] = None


@dataclass
class Message:
    """Incoming message structure"""
    topic: str
    type: str
    payload: Dict[str, Any]
    connection_id: Optional[str] = None


@dataclass
class Trade:
    """Trade message from activity topic"""
    asset: str  # ERC1155 token ID
    conditionId: str  # Market ID
    eventSlug: str
    outcome: str
    outcomeIndex: int
    price: float
    side: str  # BUY/SELL
    size: int
    slug: str  # Market slug
    timestamp: int
    title: str
    transactionHash: str
    # User info
    name: Optional[str] = None
    pseudonym: Optional[str] = None
    proxyWallet: Optional[str] = None
    profileImage: Optional[str] = None
    bio: Optional[str] = None
    icon: Optional[str] = None


@dataclass
class Comment:
    """Comment message structure"""
    id: str
    body: str
    parentEntityType: str  # "Event" or "Series"
    parentEntityID: int
    parentCommentID: Optional[str]
    userAddress: str
    replyAddress: Optional[str]
    createdAt: str
    updatedAt: str


@dataclass
class Reaction:
    """Reaction message structure"""
    id: str
    commentID: int
    reactionType: str
    icon: str
    userAddress: str
    createdAt: str


@dataclass
class CryptoPrice:
    """Crypto price update"""
    symbol: str  # BTCUSDT, ETHUSDT, etc.
    timestamp: int  # milliseconds
    value: float


@dataclass
class EquityPrice:
    """Equity price update"""
    symbol: str  # AAPL, TSLA, etc.
    timestamp: int  # milliseconds
    value: float


@dataclass
class RFQRequest:
    """RFQ Request structure"""
    requestId: str
    proxyAddress: str
    market: str
    token: str
    complement: str
    state: str
    side: str
    sizeIn: float
    sizeOut: float
    price: float
    expiry: int


@dataclass
class RFQQuote:
    """RFQ Quote structure"""
    quoteId: str
    requestId: str
    proxyAddress: str
    token: str
    state: str
    side: str
    sizeIn: float
    sizeOut: float
    condition: str
    complement: str
    expiry: int


# ============================================================================
# RTDS CLIENT
# ============================================================================

class RealTimeDataClient:
    """
    Polymarket Real-Time Data Streaming WebSocket Client

    Usage:
        def on_message(msg: Message):
            print(f"{msg.topic}/{msg.type}: {msg.payload}")

        def on_connect(client: RealTimeDataClient):
            client.subscribe([
                Subscription(topic="crypto_prices", type="update", filters='{"symbol":"BTCUSDT"}'),
                Subscription(topic="activity", type="trades"),
            ])

        client = RealTimeDataClient(on_message=on_message, on_connect=on_connect)
        client.connect()
    """

    # WebSocket endpoint
    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    # Available topics and types
    TOPICS = {
        "activity": ["trades", "orders_matched"],
        "comments": ["comment_created", "comment_removed", "reaction_created", "reaction_removed"],
        "crypto_prices": ["update"],
        "crypto_prices_chainlink": ["update"],
        "equity_prices": ["update"],
        "rfq": ["request_created", "request_edited", "request_canceled", "request_expired",
                "quote_created", "quote_edited", "quote_canceled", "quote_expired"],
        "clob_user": ["*"],  # Requires auth
    }

    # Available crypto symbols
    CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT"]

    # Available equity symbols
    EQUITY_SYMBOLS = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "NFLX", "PLTR", "OPEN", "RKLB", "ABNB"]

    def __init__(
        self,
        on_message: Optional[Callable[[Message], None]] = None,
        on_connect: Optional[Callable[['RealTimeDataClient'], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_close: Optional[Callable[[], None]] = None,
        auto_reconnect: bool = True,
        reconnect_delay: float = 5.0,
    ):
        """
        Initialize RTDS client.

        Args:
            on_message: Callback for incoming messages
            on_connect: Callback when connection established
            on_error: Callback for errors
            on_close: Callback when connection closes
            auto_reconnect: Whether to reconnect on disconnect
            reconnect_delay: Seconds to wait before reconnecting
        """
        self.on_message_callback = on_message
        self.on_connect_callback = on_connect
        self.on_error_callback = on_error
        self.on_close_callback = on_close
        self.auto_reconnect = auto_reconnect
        self.reconnect_delay = reconnect_delay

        self.ws: Optional[websocket.WebSocketApp] = None
        self.connected = False
        self.subscriptions: List[Subscription] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = False
        self.connection_id: Optional[str] = None

        # Message buffers for recent data
        self.recent_trades: List[Trade] = []
        self.recent_comments: List[Comment] = []
        self.crypto_prices: Dict[str, CryptoPrice] = {}
        self.equity_prices: Dict[str, EquityPrice] = {}

        self._max_buffer_size = 100

    def connect(self, blocking: bool = False) -> None:
        """
        Connect to the WebSocket server.

        Args:
            blocking: If True, blocks until disconnected. If False, runs in background thread.
        """
        if not HAS_WEBSOCKET:
            raise ImportError("websocket-client not installed. Run: pip install websocket-client")

        self._stop_flag = False

        self.ws = websocket.WebSocketApp(
            self.WS_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        if blocking:
            self.ws.run_forever()
        else:
            self._thread = threading.Thread(target=self._run_forever, daemon=True)
            self._thread.start()

    def _run_forever(self) -> None:
        """Run WebSocket with auto-reconnect."""
        while not self._stop_flag:
            try:
                self.ws.run_forever()
            except Exception as e:
                if self.on_error_callback:
                    self.on_error_callback(e)

            if self._stop_flag:
                break

            if self.auto_reconnect:
                print(f"[RTDS] Reconnecting in {self.reconnect_delay}s...")
                time.sleep(self.reconnect_delay)
                self.ws = websocket.WebSocketApp(
                    self.WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
            else:
                break

    def disconnect(self) -> None:
        """Disconnect from the WebSocket server."""
        self._stop_flag = True
        self.connected = False
        if self.ws:
            self.ws.close()

    def subscribe(self, subscriptions: List[Subscription]) -> None:
        """
        Subscribe to topics.

        Args:
            subscriptions: List of Subscription objects

        Example:
            client.subscribe([
                Subscription(topic="activity", type="trades"),
                Subscription(topic="crypto_prices", type="update", filters='{"symbol":"BTCUSDT"}'),
            ])
        """
        if not self.connected:
            print("[RTDS] Not connected. Queueing subscriptions...")
            self.subscriptions.extend(subscriptions)
            return

        payload = {
            "type": "subscribe",
            "subscriptions": []
        }

        for sub in subscriptions:
            sub_dict = {
                "topic": sub.topic,
                "type": sub.type,
            }
            if sub.filters:
                sub_dict["filters"] = sub.filters
            if sub.clob_auth:
                sub_dict["clob_auth"] = sub.clob_auth
            payload["subscriptions"].append(sub_dict)

        self.subscriptions.extend(subscriptions)
        self._send(payload)

    def unsubscribe(self, subscriptions: List[Subscription]) -> None:
        """
        Unsubscribe from topics.

        Args:
            subscriptions: List of Subscription objects to unsubscribe
        """
        if not self.connected:
            return

        payload = {
            "type": "unsubscribe",
            "subscriptions": []
        }

        for sub in subscriptions:
            sub_dict = {
                "topic": sub.topic,
                "type": sub.type,
            }
            if sub.filters:
                sub_dict["filters"] = sub.filters
            payload["subscriptions"].append(sub_dict)

        # Remove from tracked subscriptions
        for sub in subscriptions:
            self.subscriptions = [s for s in self.subscriptions
                                  if not (s.topic == sub.topic and s.type == sub.type)]

        self._send(payload)

    def _send(self, payload: dict) -> None:
        """Send a message to the WebSocket server."""
        if self.ws and self.connected:
            self.ws.send(json.dumps(payload))

    def _on_open(self, ws) -> None:
        """Handle WebSocket connection opened."""
        self.connected = True
        print("[RTDS] Connected to Polymarket WebSocket")

        # Re-subscribe to queued subscriptions
        if self.subscriptions:
            queued = self.subscriptions.copy()
            self.subscriptions = []
            self.subscribe(queued)

        if self.on_connect_callback:
            self.on_connect_callback(self)

    def _on_message(self, ws, message: str) -> None:
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)

            # Extract connection ID if present
            if "connection_id" in data:
                self.connection_id = data["connection_id"]

            # Parse message
            topic = data.get("topic", "")
            msg_type = data.get("type", "")
            payload = data.get("payload", data)

            msg = Message(
                topic=topic,
                type=msg_type,
                payload=payload,
                connection_id=self.connection_id,
            )

            # Buffer recent data
            self._buffer_message(msg)

            # Call user callback
            if self.on_message_callback:
                self.on_message_callback(msg)

        except json.JSONDecodeError as e:
            print(f"[RTDS] JSON decode error: {e}")
        except Exception as e:
            print(f"[RTDS] Message handling error: {e}")

    def _buffer_message(self, msg: Message) -> None:
        """Buffer recent messages by type."""
        try:
            if msg.topic == "activity" and msg.type == "trades":
                trade = Trade(**{k: v for k, v in msg.payload.items() if k in Trade.__dataclass_fields__})
                self.recent_trades.append(trade)
                if len(self.recent_trades) > self._max_buffer_size:
                    self.recent_trades.pop(0)

            elif msg.topic == "comments" and msg.type == "comment_created":
                comment = Comment(**{k: v for k, v in msg.payload.items() if k in Comment.__dataclass_fields__})
                self.recent_comments.append(comment)
                if len(self.recent_comments) > self._max_buffer_size:
                    self.recent_comments.pop(0)

            elif msg.topic == "crypto_prices" and msg.type == "update":
                price = CryptoPrice(
                    symbol=msg.payload.get("symbol", ""),
                    timestamp=msg.payload.get("timestamp", 0),
                    value=msg.payload.get("value", 0),
                )
                self.crypto_prices[price.symbol] = price

            elif msg.topic == "equity_prices" and msg.type == "update":
                price = EquityPrice(
                    symbol=msg.payload.get("symbol", ""),
                    timestamp=msg.payload.get("timestamp", 0),
                    value=msg.payload.get("value", 0),
                )
                self.equity_prices[price.symbol] = price

        except Exception as e:
            pass  # Silently ignore buffer errors

    def _on_error(self, ws, error) -> None:
        """Handle WebSocket error."""
        print(f"[RTDS] Error: {error}")
        if self.on_error_callback:
            self.on_error_callback(error)

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        """Handle WebSocket connection closed."""
        self.connected = False
        print(f"[RTDS] Connection closed: {close_status_code} {close_msg}")
        if self.on_close_callback:
            self.on_close_callback()

    # ========================================================================
    # CONVENIENCE METHODS
    # ========================================================================

    def subscribe_trades(self, event_slug: Optional[str] = None, market_slug: Optional[str] = None) -> None:
        """Subscribe to trade activity."""
        filters = ""
        if event_slug:
            filters = json.dumps({"event_slug": event_slug})
        elif market_slug:
            filters = json.dumps({"market_slug": market_slug})

        self.subscribe([Subscription(topic="activity", type="trades", filters=filters)])

    def subscribe_orders_matched(self, event_slug: Optional[str] = None, market_slug: Optional[str] = None) -> None:
        """Subscribe to matched orders."""
        filters = ""
        if event_slug:
            filters = json.dumps({"event_slug": event_slug})
        elif market_slug:
            filters = json.dumps({"market_slug": market_slug})

        self.subscribe([Subscription(topic="activity", type="orders_matched", filters=filters)])

    def subscribe_comments(self, parent_entity_id: int, parent_entity_type: str = "Event") -> None:
        """Subscribe to comments for an event or series."""
        filters = json.dumps({
            "parentEntityID": parent_entity_id,
            "parentEntityType": parent_entity_type,
        })
        self.subscribe([Subscription(topic="comments", type="*", filters=filters)])

    def subscribe_crypto_price(self, symbol: str) -> None:
        """Subscribe to crypto price updates (e.g., BTCUSDT, ETHUSDT)."""
        if symbol not in self.CRYPTO_SYMBOLS:
            print(f"[RTDS] Warning: {symbol} not in known symbols: {self.CRYPTO_SYMBOLS}")
        filters = json.dumps({"symbol": symbol})
        self.subscribe([Subscription(topic="crypto_prices", type="update", filters=filters)])

    def subscribe_all_crypto(self) -> None:
        """Subscribe to all crypto price updates."""
        for symbol in self.CRYPTO_SYMBOLS:
            self.subscribe_crypto_price(symbol)

    def subscribe_equity_price(self, symbol: str) -> None:
        """Subscribe to equity price updates (e.g., AAPL, TSLA)."""
        if symbol not in self.EQUITY_SYMBOLS:
            print(f"[RTDS] Warning: {symbol} not in known symbols: {self.EQUITY_SYMBOLS}")
        filters = json.dumps({"symbol": symbol})
        self.subscribe([Subscription(topic="equity_prices", type="update", filters=filters)])

    def subscribe_all_equities(self) -> None:
        """Subscribe to all equity price updates."""
        for symbol in self.EQUITY_SYMBOLS:
            self.subscribe_equity_price(symbol)

    def subscribe_rfq(self) -> None:
        """Subscribe to RFQ (Request for Quote) messages."""
        self.subscribe([Subscription(topic="rfq", type="*")])

    def get_crypto_price(self, symbol: str) -> Optional[CryptoPrice]:
        """Get latest crypto price from buffer."""
        return self.crypto_prices.get(symbol)

    def get_equity_price(self, symbol: str) -> Optional[EquityPrice]:
        """Get latest equity price from buffer."""
        return self.equity_prices.get(symbol)

    def get_recent_trades(self, limit: int = 10) -> List[Trade]:
        """Get recent trades from buffer."""
        return self.recent_trades[-limit:]

    def get_recent_comments(self, limit: int = 10) -> List[Comment]:
        """Get recent comments from buffer."""
        return self.recent_comments[-limit:]


# ============================================================================
# STANDALONE USAGE
# ============================================================================

def main():
    """Example usage of RTDS client."""

    def on_message(msg: Message):
        timestamp = datetime.now().strftime("%H:%M:%S")

        if msg.topic == "crypto_prices":
            symbol = msg.payload.get("symbol", "?")
            value = msg.payload.get("value", 0)
            print(f"[{timestamp}] CRYPTO {symbol}: ${value:,.2f}")

        elif msg.topic == "equity_prices":
            symbol = msg.payload.get("symbol", "?")
            value = msg.payload.get("value", 0)
            print(f"[{timestamp}] EQUITY {symbol}: ${value:,.2f}")

        elif msg.topic == "activity" and msg.type == "trades":
            side = msg.payload.get("side", "?")
            size = msg.payload.get("size", 0)
            price = msg.payload.get("price", 0)
            title = msg.payload.get("title", "?")[:40]
            print(f"[{timestamp}] TRADE {side} {size}@{price:.2f} - {title}")

        elif msg.topic == "comments":
            body = msg.payload.get("body", "")[:50]
            user = msg.payload.get("userAddress", "?")[:10]
            print(f"[{timestamp}] COMMENT {user}...: {body}")

        else:
            print(f"[{timestamp}] {msg.topic}/{msg.type}: {str(msg.payload)[:80]}")

    def on_connect(client: RealTimeDataClient):
        print("[RTDS] Setting up subscriptions...")

        # Subscribe to crypto prices
        client.subscribe_crypto_price("BTCUSDT")
        client.subscribe_crypto_price("ETHUSDT")

        # Subscribe to equity prices
        client.subscribe_equity_price("TSLA")
        client.subscribe_equity_price("NVDA")

        # Subscribe to all trades
        client.subscribe_trades()

        print("[RTDS] Subscriptions active. Waiting for data...")

    print("=" * 60)
    print("Polymarket RTDS Client - Real-Time Data Streaming")
    print("=" * 60)
    print("Connecting to WebSocket...")
    print("Press Ctrl+C to exit")
    print("")

    client = RealTimeDataClient(
        on_message=on_message,
        on_connect=on_connect,
        auto_reconnect=True,
    )

    try:
        client.connect(blocking=True)
    except KeyboardInterrupt:
        print("\n[RTDS] Disconnecting...")
        client.disconnect()
        print("[RTDS] Goodbye!")


if __name__ == "__main__":
    main()
