import chromadb
from chromadb.utils import embedding_functions
import os
import json

# ── 1. Setup ChromaDB ─────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), 'chroma_db')

client = chromadb.PersistentClient(path=DB_PATH)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="failure_reports",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

# ── 2. Historical failure reports ─────────────────────────────
FAILURE_REPORTS = [
    {
        "id": "FR001",
        "text": """
        Unit: Engine 7 | Date: 2024-01-15 | Severity: CRITICAL
        Sensor 13 (exhaust gas temperature) showed gradual rise of 4.2%
        over 20 cycles before complete failure. Pattern: slow monotonic
        increase with occasional spikes. Root cause: turbine blade fouling
        due to combustion deposits. Action taken: emergency borescope
        inspection confirmed blade degradation. Engine removed from service.
        Recommendation: trigger alert when sensor_13 rises >3% over 15 cycles.
        """,
        "metadata": {
            "sensor": "sensor_13",
            "failure_type": "turbine_blade_fouling",
            "severity": "CRITICAL",
            "rul_at_detection": 18
        }
    },
    {
        "id": "FR002",
        "text": """
        Unit: Engine 12 | Date: 2024-02-03 | Severity: HIGH
        Sensor 15 (bypass ratio pressure) dropped 2.8% over 25 cycles.
        Pattern: stepwise decrease with high variance. Root cause: compressor
        blade erosion causing reduced airflow efficiency. Action taken:
        scheduled maintenance at cycle 340, compressor cleaned and inspected.
        Outcome: sensor readings normalized after maintenance.
        Recommendation: monitor sensor_15 rolling std — high variance = early warning.
        """,
        "metadata": {
            "sensor": "sensor_15",
            "failure_type": "compressor_blade_erosion",
            "severity": "HIGH",
            "rul_at_detection": 25
        }
    },
    {
        "id": "FR003",
        "text": """
        Unit: Engine 23 | Date: 2024-02-20 | Severity: HIGH
        Sensor 11 (high pressure turbine coolant bleed) anomaly detected.
        Readings showed 5% deviation from baseline over 30 cycles.
        Pattern: oscillating readings with upward trend. Root cause:
        cooling duct partial blockage reducing turbine cooling efficiency.
        Action: cooling system flushed at cycle 289. Prevented full failure.
        Recommendation: sensor_11 delta > 0.15 per cycle warrants inspection.
        """,
        "metadata": {
            "sensor": "sensor_11",
            "failure_type": "cooling_duct_blockage",
            "severity": "HIGH",
            "rul_at_detection": 30
        }
    },
    {
        "id": "FR004",
        "text": """
        Unit: Engine 31 | Date: 2024-03-10 | Severity: MEDIUM
        Multiple sensors (sensor_13, sensor_15) showed simultaneous drift.
        Pattern: correlated degradation across exhaust and pressure systems.
        Root cause: fuel injector fouling causing incomplete combustion,
        affecting both temperature and pressure downstream.
        Action: fuel system cleaned, injectors replaced at cycle 412.
        Recommendation: when sensor_13 and sensor_15 both drift together,
        suspect fuel system first.
        """,
        "metadata": {
            "sensor": "sensor_13,sensor_15",
            "failure_type": "fuel_injector_fouling",
            "severity": "MEDIUM",
            "rul_at_detection": 45
        }
    },
    {
        "id": "FR005",
        "text": """
        Unit: Engine 45 | Date: 2024-03-28 | Severity: CRITICAL
        Rapid RUL drop detected — from 45 to 8 cycles within 10 operational
        cycles. Sensor 13 spiked 8% above baseline. Pattern: sudden onset,
        no gradual warning. Root cause: foreign object ingestion (bird strike)
        causing immediate turbine damage. Action: emergency shutdown initiated.
        Recommendation: sudden spike in sensor_13 > 5% in single cycle =
        immediate shutdown protocol.
        """,
        "metadata": {
            "sensor": "sensor_13",
            "failure_type": "foreign_object_ingestion",
            "severity": "CRITICAL",
            "rul_at_detection": 8
        }
    },
    {
        "id": "FR006",
        "text": """
        Unit: Engine 52 | Date: 2024-04-05 | Severity: LOW
        Sensor 2 (fan inlet temperature) showed seasonal variation pattern.
        Readings elevated during summer operations. Pattern: gradual rise
        correlated with ambient temperature. Root cause: operating in
        high-temperature environment without inlet cooling adjustment.
        Action: operational parameter adjustment, no mechanical fault found.
        Recommendation: normalize sensor_2 against op_setting_1 before
        flagging as anomaly in high-temp operating conditions.
        """,
        "metadata": {
            "sensor": "sensor_2",
            "failure_type": "thermal_operating_condition",
            "severity": "LOW",
            "rul_at_detection": 90
        }
    },
    {
        "id": "FR007",
        "text": """
        Unit: Engine 61 | Date: 2024-04-19 | Severity: HIGH
        Bearing wear detected through vibration pattern in sensor_11 readings.
        Pattern: increasing oscillation amplitude over 40 cycles, standard
        deviation tripling from baseline. Root cause: main shaft bearing
        lubrication failure causing metal-on-metal wear.
        Action: oil analysis confirmed metal particles, bearing replaced.
        Recommendation: sensor_11 rolling_std > 3x baseline = bearing inspection.
        """,
        "metadata": {
            "sensor": "sensor_11",
            "failure_type": "bearing_wear",
            "severity": "HIGH",
            "rul_at_detection": 22
        }
    },
    {
        "id": "FR008",
        "text": """
        Unit: Engine 74 | Date: 2024-05-02 | Severity: MEDIUM
        Sensor 4 (low pressure turbine outlet temperature) gradual increase.
        Pattern: slow linear degradation over 60 cycles. Root cause:
        turbine seal degradation allowing hot gas leakage into cooler sections.
        Action: seal replacement during scheduled maintenance.
        Recommendation: linear trend in sensor_4 over 50+ cycles warrants
        seal inspection even without anomaly flag trigger.
        """,
        "metadata": {
            "sensor": "sensor_4",
            "failure_type": "turbine_seal_degradation",
            "severity": "MEDIUM",
            "rul_at_detection": 35
        }
    },
    {
        "id": "FR009",
        "text": """
        Unit: Engine 88 | Date: 2024-05-20 | Severity: HIGH
        Isolation Forest flagged anomaly 35 cycles before manual detection.
        Sensor readings: sensor_13 +2.1%, sensor_15 -1.8%, sensor_11 +3.2%.
        Pattern: multi-sensor simultaneous degradation. Root cause: compressor
        stall event causing cascade damage across multiple systems.
        Action: immediate inspection, stall damage repaired.
        Recommendation: multi-sensor anomaly flag = highest priority alert,
        do not wait for single sensor threshold.
        """,
        "metadata": {
            "sensor": "sensor_13,sensor_15,sensor_11",
            "failure_type": "compressor_stall",
            "severity": "HIGH",
            "rul_at_detection": 35
        }
    },
    {
        "id": "FR010",
        "text": """
        Unit: Engine 95 | Date: 2024-06-08 | Severity: CRITICAL
        Complete engine failure occurred with only 5 cycles warning.
        All key sensors (13, 15, 11) in critical zone simultaneously.
        Pattern: accelerating degradation in final 20 cycles.
        Root cause: deferred maintenance — previous HIGH alerts ignored
        for 3 maintenance cycles due to operational pressure.
        Action: engine total loss, replacement required.
        Lesson: HIGH risk alerts must trigger maintenance within 2 cycles.
        Cost of ignored alert: $2.3M engine replacement + 18 days downtime.
        """,
        "metadata": {
            "sensor": "sensor_13,sensor_15,sensor_11",
            "failure_type": "complete_engine_failure",
            "severity": "CRITICAL",
            "rul_at_detection": 5
        }
    }
]

# ── 3. Ingest into ChromaDB ───────────────────────────────────
def ingest_failure_reports():
    existing = collection.count()
    if existing >= len(FAILURE_REPORTS):
        print(f"ChromaDB already has {existing} reports — skipping ingest")
        return

    print(f"Ingesting {len(FAILURE_REPORTS)} failure reports into ChromaDB...")

    collection.add(
        ids       = [r['id'] for r in FAILURE_REPORTS],
        documents = [r['text'] for r in FAILURE_REPORTS],
        metadatas = [r['metadata'] for r in FAILURE_REPORTS]
    )
    print(f"Ingested {collection.count()} reports ✅")

# ── 4. Query function ─────────────────────────────────────────
def query_similar_failures(query_text: str, n_results: int = 3) -> list:
    """
    Search ChromaDB for historically similar failure reports.
    Returns top n most similar reports with their metadata.
    """
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )

    reports = []
    for i in range(len(results['documents'][0])):
        reports.append({
            'id'      : results['ids'][0][i],
            'text'    : results['documents'][0][i].strip(),
            'metadata': results['metadatas'][0][i],
            'distance': round(results['distances'][0][i], 4)
        })
    return reports


# ── 5. Test ───────────────────────────────────────────────────
if __name__ == '__main__':
    ingest_failure_reports()

    print("\nTesting query...")
    test_query = "sensor_13 exhaust temperature rising, engine showing degradation"
    results = query_similar_failures(test_query, n_results=3)

    print(f"\nTop {len(results)} similar failure reports for query:")
    print(f"'{test_query}'\n")
    for r in results:
        print(f"  [{r['id']}] {r['metadata']['failure_type']} "
              f"| severity: {r['metadata']['severity']} "
              f"| distance: {r['distance']}")

    print("\nChromaDB vectorstore ready ✅")