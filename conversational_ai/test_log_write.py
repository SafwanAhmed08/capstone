from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.notify_rasa import append_alert_to_log, LOG_FILE

print('LOG_FILE:', LOG_FILE)
append_alert_to_log({'test':'ok','Attack':'ddos_test'}, parsed_attack='ddos_test', confidence=0.88, pcap='tcp.pcap')
print('Wrote test entry')
with open(LOG_FILE,'r',encoding='utf-8') as fh:
    lines = fh.read().splitlines()
    print('Last line:', lines[-1])
