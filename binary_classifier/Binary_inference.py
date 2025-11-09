import pyshark
import pandas as pd
import numpy as np
from scipy.stats import entropy
from tensorflow.keras.models import load_model
import joblib
import datetime as dt

# --------------------- Helper Functions ---------------------
def _safe_datetime(val):
    try:
        return pd.to_datetime(val, errors='coerce')
    except Exception:
        return pd.NaT

def _safe_int(val, default=0):
    try:
        if val is None:
            return default
        if isinstance(val, (int, np.integer)) and not np.isnan(val):
            return int(val)
        s = str(val).strip()
        if s == '':
            return default
        if s.lower() in ('true', 't', 'yes', 'y'):
            return 1
        if s.lower() in ('false', 'f', 'no', 'n'):
            return 0
        return int(float(s))
    except Exception:
        return default

def _safe_str(val):
    try:
        if val is None:
            return None
        s = str(val)
        return s if s not in ('nan', '') else None
    except Exception:
        return None

def shannon_entropy(series):
    counts = series.value_counts()
    return entropy(counts, base=2) if len(counts) > 0 else 0

def get_attr_safe(layer, name):
    """Safely get an attribute from a Pyshark layer object."""
    try:
        return getattr(layer, name)
    except Exception:
        return None

# --------------------- Step 1: Read PCAP ---------------------
PCAP_PATH = "model/mitm.pcap"
print(f"📂 Reading PCAP file: {PCAP_PATH} ...")
cap = pyshark.FileCapture(PCAP_PATH, use_json=True, include_raw=True, keep_packets=False)

packets = []
for pkt in cap:
    try:
        # Timestamp extraction (unchanged)
        ts = None
        for attr in ('sniff_time', 'sniff_timestamp', 'frame_info.time_epoch', 'frame.time'):
            try:
                parts = attr.split('.')
                cur = pkt
                for p in parts:
                    cur = getattr(cur, p)
                ts = cur
                break
            except Exception:
                ts = None
        ts_parsed = _safe_datetime(ts)

        # Layer checks
        ip_layer = pkt.ip if hasattr(pkt, 'ip') else None
        tcp_layer = pkt.tcp if hasattr(pkt, 'tcp') else None
        udp_layer = pkt.udp if hasattr(pkt, 'udp') else None
        http_layer = pkt.http if hasattr(pkt, 'http') else None
        icmp_layer = pkt.icmp if hasattr(pkt, 'icmp') else None
        dns_layer = pkt.dns if hasattr(pkt, 'dns') else None

        # Parse all safely
        packets.append({
            'frame.time': ts_parsed,

            # IP
            'ip.src_host': _safe_str(get_attr_safe(ip_layer, 'src')) if ip_layer else None,
            'ip.dst_host': _safe_str(get_attr_safe(ip_layer, 'dst')) if ip_layer else None,

            # TCP
            'tcp.srcport': _safe_int(get_attr_safe(tcp_layer, 'srcport')) if tcp_layer else 0,
            'tcp.dstport': _safe_int(get_attr_safe(tcp_layer, 'dstport')) if tcp_layer else 0,
            'tcp.connection.syn': _safe_int(
                get_attr_safe(tcp_layer, 'flags_syn') or get_attr_safe(tcp_layer, 'syn')
            ) if tcp_layer else 0,
            'tcp.connection.synack': (
                1 if tcp_layer and
                str(get_attr_safe(tcp_layer, 'flags_syn') or '').lower() in ('1', 'true') and
                str(get_attr_safe(tcp_layer, 'flags_ack') or '').lower() in ('1', 'true')
                else 0
            ),
            'tcp.connection.rst': _safe_int(
                get_attr_safe(tcp_layer, 'flags_reset') or get_attr_safe(tcp_layer, 'flags_rst')
            ) if tcp_layer else 0,
            'tcp.connection.fin': _safe_int(get_attr_safe(tcp_layer, 'flags_fin')) if tcp_layer else 0,
            'tcp.flags.ack': _safe_int(get_attr_safe(tcp_layer, 'flags_ack')) if tcp_layer else 0,
            'tcp.len': _safe_int(
                get_attr_safe(tcp_layer, 'len') or get_attr_safe(pkt, 'length')
            ) if tcp_layer else 0,
            'tcp.payload': _safe_str(get_attr_safe(tcp_layer, 'payload')) if tcp_layer else '',

            # UDP (optional if you want to track UDP ports)
            'udp.srcport': _safe_int(get_attr_safe(udp_layer, 'srcport')) if udp_layer else 0,
            'udp.dstport': _safe_int(get_attr_safe(udp_layer, 'dstport')) if udp_layer else 0,

            # HTTP
            'http.request.method': _safe_str(get_attr_safe(http_layer, 'request_method')) if http_layer else None,
            'http.request.full_uri': _safe_str(get_attr_safe(http_layer, 'request_full_uri')) if http_layer else None,
            'http.content_length': _safe_int(get_attr_safe(http_layer, 'content_length')) if http_layer else None,

            # ICMP
            'icmp.checksum': _safe_str(get_attr_safe(icmp_layer, 'checksum')) if icmp_layer else None,

            # DNS
            'dns.qry.name': _safe_str(get_attr_safe(dns_layer, 'qry_name')) if dns_layer else None,
        })

    except Exception as e:
        print(f"⚠️ Packet skipped due to error: {e}")

cap.close()
df = pd.DataFrame(packets)
print(f"✅ Captured {len(df):,} packets")

# --------------------- Step 2: Validate & Sort ---------------------
df['frame.time'] = pd.to_datetime(df['frame.time'], errors='coerce')
df.dropna(subset=['frame.time'], inplace=True)
if df.empty:
    print("❌ No valid packets with timestamps found in PCAP.")
    raise SystemExit(0)

df = df.sort_values('frame.time').reset_index(drop=True)
df.set_index('frame.time', inplace=True)

# --------------------- Step 3: Feature Engineering ---------------------
print("⚙️ Extracting 5-second window features...")
window_size = '5S'
windows = df.groupby(pd.Grouper(freq=window_size))

features = []
for time, group in windows:
    if group.empty:
        continue

    tcp_len = pd.to_numeric(group.get('tcp.len', pd.Series(dtype=float)), errors='coerce')
    http_cl = pd.to_numeric(group.get('http.content_length', pd.Series(dtype=float)), errors='coerce')

    row = {
        'window_start': time,
        'packet_count': len(group),
        'syn_count': pd.to_numeric(group.get('tcp.connection.syn', 0), errors='coerce').sum(),
        'synack_count': pd.to_numeric(group.get('tcp.connection.synack', 0), errors='coerce').sum(),
        'rst_count': pd.to_numeric(group.get('tcp.connection.rst', 0), errors='coerce').sum(),
        'fin_count': pd.to_numeric(group.get('tcp.connection.fin', 0), errors='coerce').sum(),
        'tcp_len_total': tcp_len.sum(skipna=True),
        'tcp_len_avg': tcp_len.mean(skipna=True) if not tcp_len.empty else 0,
        'unique_src_ips': group['ip.src_host'].nunique(),
        'unique_dst_ips': group['ip.dst_host'].nunique(),
        'src_entropy': shannon_entropy(group['ip.src_host'].dropna().astype(str)),
        'dst_entropy': shannon_entropy(group['ip.dst_host'].dropna().astype(str)),
        'src_port_entropy': shannon_entropy(group.get('tcp.srcport', pd.Series()).dropna().astype(str)),
        'dst_port_entropy': shannon_entropy(group.get('tcp.dstport', pd.Series()).dropna().astype(str)),
        'syn_ratio': pd.to_numeric(group.get('tcp.connection.syn', 0), errors='coerce').sum() / (len(group) + 1e-6),
        'rst_ratio': pd.to_numeric(group.get('tcp.connection.rst', 0), errors='coerce').sum() / (len(group) + 1e-6),
        'http_get_count': (group.get('http.request.method') == 'GET').sum(),
        'http_post_count': (group.get('http.request.method') == 'POST').sum(),
        'http_uri_entropy': shannon_entropy(group.get('http.request.full_uri', pd.Series()).dropna().astype(str)),
        'http_avg_content_length': http_cl.mean(skipna=True) if not http_cl.empty else 0,
        'icmp_count': group['icmp.checksum'].notnull().sum() if 'icmp.checksum' in group else 0,
        'dns_query_count': group['dns.qry.name'].notnull().sum() if 'dns.qry.name' in group else 0,
        'dns_query_entropy': shannon_entropy(group.get('dns.qry.name', pd.Series()).dropna().astype(str)),
        'avg_payload_len': group.get('tcp.payload', pd.Series()).dropna().astype(str).apply(len).mean() if 'tcp.payload' in group else 0,
    }
    features.append(row)

feature_df = pd.DataFrame(features)
if feature_df.empty:
    print("❌ No feature windows extracted.")
    raise SystemExit(0)

print(f"✅ Extracted {len(feature_df):,} windows")

# --------------------- Step 4: Run Inference ---------------------
print("🧠 Loading scaler and model...")
try:
    scaler = joblib.load("model/scaler.save")
    model = load_model("model/binary_with_mitm.h5")
    print("✅ Model and scaler loaded successfully.")
except Exception as e:
    print("❌ Failed to load model or scaler:", e)
    raise SystemExit(1)

X_features = feature_df.drop(columns=['window_start']).apply(pd.to_numeric, errors='coerce').fillna(0)
X_scaled = scaler.transform(X_features)

print("🔍 Running predictions...")
predictions = model.predict(X_scaled, verbose=0).flatten()
labels = (predictions > 0.5).astype(int)

# Compute confidence as the model's certainty in its predicted class
confidence = np.where(labels == 1, predictions, 1 - predictions)

results = pd.DataFrame({
    'window_start': feature_df['window_start'],
    'prediction': labels,
    'confidence': confidence
})

results.to_csv("inference_output.csv", index=False)

print("\n====================== 🧾 INFERENCE RESULTS ======================")
print(f"Total windows analyzed: {len(results)}")
print(f"Predicted ATTACK windows: {(labels==1).sum()}")
print(f"Predicted NORMAL windows: {(labels==0).sum()}")
print("\nTop 10 predictions:")
print(results.head(10).to_string(index=False))
print("\n===============================================================")
print("✅ Inference complete! Full results saved to inference_output.csv")
