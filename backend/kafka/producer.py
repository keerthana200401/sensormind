import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import time
import random
import pandas as pd
from kafka import KafkaProducer
from dotenv import load_dotenv

load_dotenv()

# ── Setup producer ────────────────────────────────────────────
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

TOPIC = 'sensor-stream'

# ── Load real sensor data ─────────────────────────────────────
df           = pd.read_parquet('../../data/features/train_features.parquet')
feature_cols = ['sensor_13', 'sensor_15', 'sensor_11',
                'sensor_2',  'sensor_4',  'cycle']

# Pick a few engines to simulate
ENGINES = ['Engine_001', 'Engine_002', 'Engine_003']

def simulate_stream(interval_seconds: float = 2.0):
    """
    Simulates live sensor stream by sending real data
    row by row to Kafka topic every interval_seconds.
    """
    print(f"Starting sensor stream → topic: '{TOPIC}'")
    print(f"Sending 1 reading every {interval_seconds}s")
    print("Press Ctrl+C to stop\n")

    count = 0
    while True:
        # Pick random engine and sample
        engine_id = random.choice(ENGINES)
        sample    = df.sample(1).iloc[0]

        message = {
            'engine_id' : engine_id,
            'cycle'     : int(sample.get('cycle', 0)),
            'sensor_13' : round(float(sample.get('sensor_13', 0)), 4),
            'sensor_15' : round(float(sample.get('sensor_15', 0)), 4),
            'sensor_11' : round(float(sample.get('sensor_11', 0)), 4),
            'sensor_2'  : round(float(sample.get('sensor_2',  0)), 4),
            'sensor_4'  : round(float(sample.get('sensor_4',  0)), 4),
            'timestamp' : time.time()
        }

        producer.send(TOPIC, value=message)
        count += 1

        print(f"[{count}] Sent → {engine_id} | "
              f"cycle: {message['cycle']} | "
              f"sensor_13: {message['sensor_13']} | "
              f"RUL zone: {'⚠️ CRITICAL' if message['cycle'] > 200 else '✅ OK'}")

        time.sleep(interval_seconds)


if __name__ == '__main__':
    simulate_stream(interval_seconds=2.0)