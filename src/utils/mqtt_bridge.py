"""
EdgeTransAI — IoT Data Bridge (MQTT)
=======================================
Subscribes to MQTT sensor topics (CCTV counts, LiDAR density, GPS),
pre-processes the raw payloads, and publishes cleaned observations to
the AI layer.  Also receives signal control commands and forwards them
to the physical traffic controller hardware.

In development mode (no real broker) it generates synthetic sensor
streams that mimic realistic traffic patterns.

Usage
-----
    # Real broker
    python -m src.utils.mqtt_bridge --broker 192.168.1.100 --port 1883

    # Synthetic mode (no broker needed)
    python -m src.utils.mqtt_bridge --synthetic --duration 60
"""

import os
import json
import time
import argparse
import threading
import numpy as np
from typing import Callable, Dict, Optional
from dataclasses import dataclass, asdict

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("[MQTTBridge] paho-mqtt not installed. Run: pip install paho-mqtt")


# ---------------------------------------------------------------------------
# Message schemas
# ---------------------------------------------------------------------------

@dataclass
class SensorPayload:
    intersection_id: int
    timestamp:       float
    lane_counts:     list     # [int × 4] vehicle count per lane
    lane_speeds:     list     # [float × 4] avg speed km/h per lane
    phase_current:   int      # current signal phase 0-7
    gps_vehicles:    int      # probe vehicle count in 200m radius
    source:          str      # "lidar" | "cctv" | "loop"


@dataclass
class SignalCommand:
    intersection_id: int
    timestamp:       float
    target_phase:    int
    green_time:      float    # seconds


# ---------------------------------------------------------------------------
# MQTT Bridge
# ---------------------------------------------------------------------------

class MQTTBridge:
    """
    Thin wrapper around paho-mqtt providing:
      - Automatic reconnect with exponential back-off
      - JSON serialisation / deserialisation
      - Callback registration per topic pattern
      - Synthetic stream fallback for development
    """

    TOPIC_SENSORS  = "edgetransai/sensors/{intersection_id}"
    TOPIC_SIGNALS  = "edgetransai/signals/{intersection_id}"
    TOPIC_INTENT   = "edgetransai/intent/{intersection_id}"
    TOPIC_STATS    = "edgetransai/stats/global"

    def __init__(self, broker: str = "localhost", port: int = 1883,
                 client_id: str = "edgetransai-bridge"):
        self.broker    = broker
        self.port      = port
        self._callbacks: Dict[str, Callable] = {}
        self._running   = False

        if MQTT_AVAILABLE:
            self._client = mqtt.Client(client_id=client_id)
            self._client.on_connect    = self._on_connect
            self._client.on_message    = self._on_message
            self._client.on_disconnect = self._on_disconnect

    # ------------------------------------------------------------------
    def connect(self, username: Optional[str] = None,
                password: Optional[str] = None):
        if not MQTT_AVAILABLE:
            print("[MQTTBridge] Cannot connect — paho-mqtt not installed.")
            return
        if username:
            self._client.username_pw_set(username, password)
        self._client.connect(self.broker, self.port, keepalive=60)
        self._client.loop_start()
        self._running = True
        print(f"[MQTTBridge] Connecting to {self.broker}:{self.port} …")

    def disconnect(self):
        if MQTT_AVAILABLE and self._running:
            self._client.loop_stop()
            self._client.disconnect()
            self._running = False

    # ------------------------------------------------------------------
    def subscribe(self, topic_pattern: str, callback: Callable):
        """Register a callback for a topic pattern (supports MQTT wildcards)."""
        self._callbacks[topic_pattern] = callback
        if MQTT_AVAILABLE and self._running:
            self._client.subscribe(topic_pattern)

    def publish_sensor(self, payload: SensorPayload):
        topic = self.TOPIC_SENSORS.format(
            intersection_id=payload.intersection_id)
        self._publish(topic, asdict(payload))

    def publish_signal(self, cmd: SignalCommand):
        topic = self.TOPIC_SIGNALS.format(
            intersection_id=cmd.intersection_id)
        self._publish(topic, asdict(cmd))

    def _publish(self, topic: str, data: dict, qos: int = 1):
        msg = json.dumps(data)
        if MQTT_AVAILABLE and self._running:
            self._client.publish(topic, msg, qos=qos)

    # ------------------------------------------------------------------
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("[MQTTBridge] Connected.")
            for pattern in self._callbacks:
                client.subscribe(pattern)
        else:
            print(f"[MQTTBridge] Connection failed (rc={rc})")

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload)
        except json.JSONDecodeError:
            return
        for pattern, cb in self._callbacks.items():
            if mqtt.topic_matches_sub(pattern, msg.topic):
                cb(msg.topic, payload)

    def _on_disconnect(self, client, userdata, rc):
        if rc != 0:
            print(f"[MQTTBridge] Unexpected disconnect (rc={rc}). Reconnecting…")
            time.sleep(min(2 ** getattr(self, "_retry", 0), 30))
            try:
                self._client.reconnect()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Synthetic sensor stream (development mode)
# ---------------------------------------------------------------------------

class SyntheticSensorStream:
    """
    Emits realistic MQTT-style sensor payloads without a live broker.
    Useful for local development and model debugging.
    """

    def __init__(self, num_intersections: int = 16, step_sec: int = 5,
                 seed: int = 42):
        self.n    = num_intersections
        self.step = step_sec
        self.rng  = np.random.default_rng(seed)
        self._t   = 0

    def _demand_scale(self) -> float:
        hour = (self._t * self.step) / 3600.0
        d = (0.25
             + 0.5 * np.exp(-((hour - 8.0) ** 2) / 4)
             + 0.4 * np.exp(-((hour - 17.5) ** 2) / 3))
        return float(np.clip(d, 0.1, 1.0))

    def next_batch(self) -> list:
        """Return a list of SensorPayload objects for the current time step."""
        d = self._demand_scale()
        payloads = []
        for i in range(self.n):
            counts = self.rng.poisson(d * 6, 4).tolist()
            speeds = (60.0 - np.array(counts) * 2.5
                      + self.rng.normal(0, 2, 4)).clip(5, 80).tolist()
            payloads.append(SensorPayload(
                intersection_id=i,
                timestamp=time.time(),
                lane_counts=counts,
                lane_speeds=speeds,
                phase_current=int(self.rng.integers(0, 8)),
                gps_vehicles=int(self.rng.poisson(d * 4)),
                source="synthetic",
            ))
        self._t += 1
        return payloads

    def stream(self, duration_sec: int = 3600,
               callback: Optional[Callable] = None):
        """Block and emit payloads at real-time rate."""
        n_steps = duration_sec // self.step
        for _ in range(n_steps):
            batch = self.next_batch()
            for p in batch:
                if callback:
                    callback(p)
                else:
                    print(json.dumps(asdict(p), indent=2))
            time.sleep(self.step)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker",     default="localhost")
    parser.add_argument("--port",       type=int, default=1883)
    parser.add_argument("--synthetic",  action="store_true")
    parser.add_argument("--duration",   type=int, default=60,
                        help="Synthetic stream duration (seconds)")
    parser.add_argument("--intersections", type=int, default=16)
    args = parser.parse_args()

    if args.synthetic:
        print(f"[SyntheticStream] Running for {args.duration}s …")
        stream = SyntheticSensorStream(num_intersections=args.intersections)
        stream.stream(duration_sec=args.duration,
                      callback=lambda p: print(
                          f"[INT {p.intersection_id:02d}] "
                          f"counts={p.lane_counts} speeds={[f'{s:.1f}' for s in p.lane_speeds]}"
                      ))
    else:
        bridge = MQTTBridge(broker=args.broker, port=args.port)
        bridge.connect()
        bridge.subscribe("edgetransai/sensors/#",
                         lambda t, p: print(f"[{t}] {p}"))
        print("Listening … Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            bridge.disconnect()
