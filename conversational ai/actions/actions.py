# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

import pandas as pd
import os
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import re
import google.generativeai as genai
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv('api.env')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Verify API key
if not GOOGLE_API_KEY:
    logger.error("❌ No API key found in api.env file")
    raise ValueError("Missing GOOGLE_API_KEY in environment variables")

# Load CSV with absolui te path to ensure it's found
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
csv_path = os.path.join(project_root, "mitre_mitigations.csv")

try:
    df = pd.read_csv(csv_path)
    print(f"✅ Action server loaded {len(df)} mitigations from {csv_path}")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    print(f"Tried path: {csv_path}")
    df = pd.DataFrame()

class ActionMitigationLookup(Action):
    def name(self) -> Text:
        return "action_lookup_mitigation"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        if df.empty:
            dispatcher.utter_message(text="❌ Error: MITRE ATT&CK data not loaded. Please check the action server logs.")
            return []
        
        query = tracker.latest_message.get("text", "").lower()
        print(f"🔍 Searching for query: '{query}'")
        print(f"🔍 DataFrame shape: {df.shape}")

        # Hardcoded overrides for specific topics/keywords
        normalized_query = query.strip().lower()
        overrides = {
            # DDoS TCP variants
            "ddos_tcp": (
                "✅ Mitigations for DDoS over TCP (curated):\n\n"
                "• Deploy upstream scrubbing/anycast DDoS protection; advertise through multiple POPs.\n"
                "• Enable SYN cookies, SYN proxying, and backlog tuning on edge services.\n"
                "• Enforce connection limits per IP/ASN; drop suspicious long-lived half-open connections.\n"
                "• Use rate limiting and surge queues at L7; prefer stateless frontends.\n"
                "• Geo/IP reputation filtering during attacks; block reflection/amplification sources.\n"
                "• Autoscale horizontally; implement circuit breakers and graceful degradation."
            ),
            "tcp syn flood": (
                "✅ Mitigations for DDoS over TCP (curated):\n\n"
                "• Deploy upstream scrubbing/anycast DDoS protection; advertise through multiple POPs.\n"
                "• Enable SYN cookies, SYN proxying, and backlog tuning on edge services.\n"
                "• Enforce connection limits per IP/ASN; drop suspicious long-lived half-open connections.\n"
                "• Use rate limiting and surge queues at L7; prefer stateless frontends.\n"
                "• Geo/IP reputation filtering during attacks; block reflection/amplification sources.\n"
                "• Autoscale horizontally; implement circuit breakers and graceful degradation."
            ),
            "syn flood": (
                "✅ Mitigations for DDoS over TCP (curated):\n\n"
                "• Deploy upstream scrubbing/anycast DDoS protection; advertise through multiple POPs.\n"
                "• Enable SYN cookies, SYN proxying, and backlog tuning on edge services.\n"
                "• Enforce connection limits per IP/ASN; drop suspicious long-lived half-open connections.\n"
                "• Use rate limiting and surge queues at L7; prefer stateless frontends.\n"
                "• Geo/IP reputation filtering during attacks; block reflection/amplification sources.\n"
                "• Autoscale horizontally; implement circuit breakers and graceful degradation."
            ),
            # DDoS HTTP variants
            "ddos_http": (
                "✅ Mitigations for HTTP flood (curated):\n\n"
                "• Put a CDN/WAF in front; enable bot mitigation, JS challenges, and CAPTCHA.\n"
                "• Cache aggressively with stale-while-revalidate; offload static and semi-static content.\n"
                "• Rate-limit by IP/session/API key; enforce request budgets and backoff.\n"
                "• Optimize heavy endpoints; avoid expensive fan-out and unbounded queries.\n"
                "• Use request prioritization and shed low-priority traffic during incidents.\n"
                "• Monitor anomalous user-agents and referrers; block abusive patterns."
            ),
            "http flood": (
                "✅ Mitigations for HTTP flood (curated):\n\n"
                "• Put a CDN/WAF in front; enable bot mitigation, JS challenges, and CAPTCHA.\n"
                "• Cache aggressively with stale-while-revalidate; offload static and semi-static content.\n"
                "• Rate-limit by IP/session/API key; enforce request budgets and backoff.\n"
                "• Optimize heavy endpoints; avoid expensive fan-out and unbounded queries.\n"
                "• Use request prioritization and shed low-priority traffic during incidents.\n"
                "• Monitor anomalous user-agents and referrers; block abusive patterns."
            ),
            # DDoS ICMP variants
            "ddos_icmp": (
                "✅ Mitigations for ICMP flood (curated):\n\n"
                "• Rate-limit ICMP at network edge; drop malformed/fragmented echo requests.\n"
                "• Disable unnecessary ICMP types on internet-facing interfaces.\n"
                "• Use ACLs to block known amplifiers and reflection paths.\n"
                "• Prefer provider-level DDoS filtering and blackhole communities when overwhelmed.\n"
                "• Monitor interface PPS/bitrate; auto-trigger mitigation runbooks."
            ),
            "icmp flood": (
                "✅ Mitigations for ICMP flood (curated):\n\n"
                "• Rate-limit ICMP at network edge; drop malformed/fragmented echo requests.\n"
                "• Disable unnecessary ICMP types on internet-facing interfaces.\n"
                "• Use ACLs to block known amplifiers and reflection paths.\n"
                "• Prefer provider-level DDoS filtering and blackhole communities when overwhelmed.\n"
                "• Monitor interface PPS/bitrate; auto-trigger mitigation runbooks."
            ),
            # Vulnerability scanning variants
            "vulnerability scanning": (
                "✅ Mitigations for vulnerability scanning (curated):\n\n"
                "• Enforce authenticated scans in approved maintenance windows; log and alert on out-of-window scans.\n"
                "• Restrict scan sources with network ACLs; allowlist only designated scanners.\n"
                "• Rate-limit scan traffic; enable IDS/IPS signatures for common scanners (e.g., nmap).\n"
                "• Segment management networks; isolate scanners from production and sensitive segments.\n"
                "• Use least-privilege scan credentials; rotate and vault secrets.\n"
                "• Monitor for excessive service probes and banner grabs; auto-quarantine offending hosts."
            ),
            "vulnerability scan": (
                "✅ Mitigations for vulnerability scanning (curated):\n\n"
                "• Enforce authenticated scans in approved maintenance windows; log and alert on out-of-window scans.\n"
                "• Restrict scan sources with network ACLs; allowlist only designated scanners.\n"
                "• Rate-limit scan traffic; enable IDS/IPS signatures for common scanners (e.g., nmap).\n"
                "• Segment management networks; isolate scanners from production and sensitive segments.\n"
                "• Use least-privilege scan credentials; rotate and vault secrets.\n"
                "• Monitor for excessive service probes and banner grabs; auto-quarantine offending hosts."
            ),
            "vuln scanning": (
                "✅ Mitigations for vulnerability scanning (curated):\n\n"
                "• Enforce authenticated scans in approved maintenance windows; log and alert on out-of-window scans.\n"
                "• Restrict scan sources with network ACLs; allowlist only designated scanners.\n"
                "• Rate-limit scan traffic; enable IDS/IPS signatures for common scanners (e.g., nmap).\n"
                "• Segment management networks; isolate scanners from production and sensitive segments.\n"
                "• Use least-privilege scan credentials; rotate and vault secrets.\n"
                "• Monitor for excessive service probes and banner grabs; auto-quarantine offending hosts."
            ),
            "vulnerability assessment": (
                "✅ Mitigations for vulnerability scanning (curated):\n\n"
                "• Enforce authenticated scans in approved maintenance windows; log and alert on out-of-window scans.\n"
                "• Restrict scan sources with network ACLs; allowlist only designated scanners.\n"
                "• Rate-limit scan traffic; enable IDS/IPS signatures for common scanners (e.g., nmap).\n"
                "• Segment management networks; isolate scanners from production and sensitive segments.\n"
                "• Use least-privilege scan credentials; rotate and vault secrets.\n"
                "• Monitor for excessive service probes and banner grabs; auto-quarantine offending hosts."
            ),
            # MiTM variants
            "mitm": (
                "✅ Mitigations for Man-in-the-Middle (curated):\n\n"
                "• Enforce TLS 1.2+ with HSTS and certificate pinning where applicable.\n"
                "• Use DNSSEC and DoT/DoH; monitor and alert on DNS poisoning attempts.\n"
                "• Implement mutual TLS for service-to-service traffic and admin interfaces.\n"
                "• Enable 802.1X/NAC on LAN; detect ARP spoofing and rogue APs.\n"
                "• Prefer SSH with strong KEX/MACs; rotate host keys; enforce MFA for admins.\n"
                "• Use signed updates and package repositories; verify software provenance."
            ),
            "man in the middle": (
                "✅ Mitigations for Man-in-the-Middle (curated):\n\n"
                "• Enforce TLS 1.2+ with HSTS and certificate pinning where applicable.\n"
                "• Use DNSSEC and DoT/DoH; monitor and alert on DNS poisoning attempts.\n"
                "• Implement mutual TLS for service-to-service traffic and admin interfaces.\n"
                "• Enable 802.1X/NAC on LAN; detect ARP spoofing and rogue APs.\n"
                "• Prefer SSH with strong KEX/MACs; rotate host keys; enforce MFA for admins.\n"
                "• Use signed updates and package repositories; verify software provenance."
            ),
            # Port scanning variants
            "portscanning": (
                "✅ Mitigations for port scanning (curated):\n\n"
                "• Block unsolicited inbound with default-deny; expose only necessary ports.\n"
                "• Implement network segmentation; place sensitive services behind jump hosts/VPN.\n"
                "• Rate-limit and tarp it: slow responses to suspected scanners to waste their time.\n"
                "• Enable IDS/IPS signatures for common scan patterns (SYN, FIN, XMAS, NULL).\n"
                "• Use port-knocking or single-packet auth for rarely used admin services.\n"
                "• Monitor for repeated probing and auto-block offending IPs temporarily."
            ),
            "port scanning": (
                "✅ Mitigations for port scanning (curated):\n\n"
                "• Block unsolicited inbound with default-deny; expose only necessary ports.\n"
                "• Implement network segmentation; place sensitive services behind jump hosts/VPN.\n"
                "• Rate-limit and tarp it: slow responses to suspected scanners to waste their time.\n"
                "• Enable IDS/IPS signatures for common scan patterns (SYN, FIN, XMAS, NULL).\n"
                "• Use port-knocking or single-packet auth for rarely used admin services.\n"
                "• Monitor for repeated probing and auto-block offending IPs temporarily."
            ),
            "port_scan": (
                "✅ Mitigations for port scanning (curated):\n\n"
                "• Block unsolicited inbound with default-deny; expose only necessary ports.\n"
                "• Implement network segmentation; place sensitive services behind jump hosts/VPN.\n"
                "• Rate-limit and tarp it: slow responses to suspected scanners to waste their time.\n"
                "• Enable IDS/IPS signatures for common scan patterns (SYN, FIN, XMAS, NULL).\n"
                "• Use port-knocking or single-packet auth for rarely used admin services.\n"
                "• Monitor for repeated probing and auto-block offending IPs temporarily."
            ),
        }

        for key, curated_response in overrides.items():
            if key in normalized_query:
                dispatcher.utter_message(text=curated_response)
                return []
        
        # Search by ID (e.g., T1174)
        if query.strip().upper().startswith('T') and any(char.isdigit() for char in query):
            print(f"🔍 Searching by ID pattern: {query.upper()}")
            result = df[df['ID'].str.contains(query.upper(), na=False)]
            print(f"🔍 ID search results: {len(result)} matches")
        else:
            # Search by name or description
            print(f"🔍 Searching by name/description: {query}")
            name_match = df[df['Name'].str.lower().str.contains(query, na=False)]
            desc_match = df[df['Description'].str.lower().str.contains(query, na=False)]
            result = pd.concat([name_match, desc_match]).drop_duplicates()
            print(f"🔍 Name matches: {len(name_match)}, Description matches: {len(desc_match)}")
            print(f"🔍 Total combined results: {len(result)}")
        
        if not result.empty:
            if len(result) == 1:
                row = result.iloc[0]
                response = f"✅ **Mitigation {row['ID']} - {row['Name']}:**\n\n{row['Description']}"
            else:
                response = f"🔍 Found {len(result)} relevant mitigations:\n\n"
                for idx, row in result.head(3).iterrows():
                    response += f"• **{row['ID']}**: {row['Name']}\n"
                if len(result) > 3:
                    response += f"\n... and {len(result) - 3} more. Please be more specific."
        else:
            # Show some sample data to help debug
            sample_ids = df['ID'].head(5).tolist()
            sample_names = df['Name'].head(5).tolist()
            response = f"❌ I couldn't find a relevant mitigation in MITRE ATT&CK for '{query}'.\n\n"
            response += "💡 **Try these examples:**\n"
            response += f"• Search by ID: {', '.join(sample_ids)}\n"
            response += f"• Search by name: {', '.join(sample_names)}\n"
            response += "• Or try: 'privilege escalation', 'vulnerability scanning', 'network segmentation'"
        
        dispatcher.utter_message(text=response)
        return []

class ActionMitigationList(Action):
    def name(self) -> Text:
        return "action_list_mitigations"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        if df.empty:
            dispatcher.utter_message(text="❌ Error: MITRE ATT&CK data not loaded. Please check the action server logs.")
            return []
        
        # Get a sample of available mitigations
        sample_mitigations = df.head(10)
        
        response = "📋 **Available MITRE ATT&CK Mitigations (showing first 10):**\n\n"
        for idx, row in sample_mitigations.iterrows():
            response += f"• **{row['ID']}**: {row['Name']}\n"
        
        response += f"\n💡 **Total available**: {len(df)} mitigations\n"
        response += "Ask me about specific techniques or use IDs like 'T1174' for detailed information."
        
        dispatcher.utter_message(text=response)
        return []

class ActionRunAnomalyDetection(Action):
    def name(self) -> Text:
        return "action_run_anomaly_detection"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        text = tracker.latest_message.get("text", "").lower()
        
        # Only trigger if text specifically mentions 'device' or contains a file path
        if not ("device" in text or "\\" in text or "/" in text):
            # Return empty list to allow fallback to LLM
            return []
            
        device_match = re.search(r"device\s+([A-Za-z0-9_\-\.]+)", text, re.IGNORECASE)
        path_match = re.search(r"([A-Za-z]:\\\\[^\s]+|/[^\s]+)", text)

        if device_match:
            target = device_match.group(1)
            dispatcher.utter_message(text=f"Starting anomaly detection for device '{target}'. I'll analyze recent traffic and report anomalies.")
        elif path_match:
            target = path_match.group(1)
            dispatcher.utter_message(text=f"Starting anomaly detection on logs at '{target}'. Results will include anomaly scores and flagged events.")
        else:
            dispatcher.utter_message(text="Please specify a device (e.g., 'device sensor-01') or a log file path to analyze.")

        return []


class ActionAnalyzePcap(Action):
    """Analyze PCAP file and detect attacks using ML model"""
    def name(self) -> Text:
        return "action_analyze_pcap"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        text = tracker.latest_message.get("text", "").lower()
        
        # Extract PCAP file path from user message
        pcap_match = re.search(r"([A-Za-z]:[^\s]+\.pcap|/[^\s]+\.pcap)", text, re.IGNORECASE)
        
        if not pcap_match:
            dispatcher.utter_message(
                text="Please provide a PCAP file path. Example:\n"
                     "• 'Analyze C:/Users/shrij/OneDrive/Desktop/Capstone Phase 3/model/tcp.pcap'\n"
                     "• 'Check /path/to/traffic.pcap for attacks'"
            )
            return []
        
        pcap_file = pcap_match.group(1)
        
        if not os.path.exists(pcap_file):
            dispatcher.utter_message(text=f"❌ PCAP file not found: {pcap_file}")
            return []
        
        # Inform user that analysis is starting
        dispatcher.utter_message(text=f"🔍 Analyzing network traffic from:\n`{os.path.basename(pcap_file)}`\n\n⏳ Processing PCAP file...")
        
        try:
            # Import the inference function
            import sys
            model_path = os.path.join(os.path.dirname(current_dir), '..', 'model')
            sys.path.insert(0, model_path)
            
            from inferencegpu import analyze_pcap_for_rasa
            
            # Run analysis
            result = analyze_pcap_for_rasa(pcap_file)
            
            if not result['success']:
                dispatcher.utter_message(text=f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                return []
            
            # Build response message
            if result['attack_detected']:
                attack_type = result['primary_attack']
                confidence = result['confidence']
                severity = result['severity']
                
                response = f"🚨 **{severity} Severity Attack Detected!**\n\n"
                response += f"**Attack Type:** {attack_type}\n"
                response += f"**Confidence:** {confidence:.1%}\n"
                response += f"**Windows Analyzed:** {result['windows_analyzed']}\n"
                response += f"**Attack Coverage:** {result['attack_percentage']:.1f}%\n\n"
                
                # Attack-specific initial advice
                if 'TCP' in attack_type:
                    response += "⚠️ **DDoS TCP SYN Flood detected!**\n\n"
                    dispatcher.utter_message(text=response)
                    dispatcher.utter_message(text="Would you like me to provide mitigation strategies for DDoS_TCP attacks?")
                
                elif 'HTTP' in attack_type:
                    response += "⚠️ **DDoS HTTP Flood detected!**\n\n"
                    dispatcher.utter_message(text=response)
                    dispatcher.utter_message(text="Would you like me to provide mitigation strategies for DDoS_HTTP attacks?")
                
                elif 'ICMP' in attack_type:
                    response += "⚠️ **DDoS ICMP Flood detected!**\n\n"
                    dispatcher.utter_message(text=response)
                    dispatcher.utter_message(text="Would you like me to provide mitigation strategies for DDoS_ICMP attacks?")
                
                elif 'MITM' in attack_type:
                    response += "⚠️ **Man-in-the-Middle attack detected!**\n\n"
                    dispatcher.utter_message(text=response)
                    dispatcher.utter_message(text="Would you like me to provide mitigation strategies for MITM attacks?")
                
                elif 'Port' in attack_type:
                    response += "⚠️ **Port Scanning activity detected!**\n\n"
                    dispatcher.utter_message(text=response)
                    dispatcher.utter_message(text="Would you like me to provide mitigation strategies for Port Scanning?")
                
                else:
                    dispatcher.utter_message(text=response)
                    dispatcher.utter_message(text=f"Type 'mitigation for {attack_type}' to get defense strategies.")
            
            else:
                response = "✅ **No Attacks Detected**\n\n"
                response += f"**Traffic Status:** Normal\n"
                response += f"**Confidence:** {result['confidence']:.1%}\n"
                response += f"**Windows Analyzed:** {result['windows_analyzed']}\n\n"
                response += "Your network traffic appears to be operating normally."
                dispatcher.utter_message(text=response)
        
        except Exception as e:
            logger.error(f"Error in PCAP analysis: {str(e)}")
            dispatcher.utter_message(text=f"❌ An error occurred during analysis: {str(e)}")
        
        return []
class ActionLLMFallback(Action):
    def name(self) -> Text:
        return "action_llm_fallback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        user_message = tracker.latest_message.get('text')
        logger.debug(f"Processing message: {user_message}")

        try:
            # Configure Gemini
            genai.configure(api_key=GOOGLE_API_KEY)
            
            # Debug available models
            logger.debug("Attempting to list available models...")
            try:
                models = genai.list_models()
                logger.debug("Available models:")
                for m in models:
                    logger.debug(f"- {m.name}")
            except Exception as model_error:
                logger.error(f"Error listing models: {str(model_error)}")

            # Use just gemini-pro without models/ prefix
            model = genai.GenerativeModel('gemini-2.5-flash-lite')

            prompt = f"""As a smart factory cybersecurity assistant, please answer: {user_message}
            Focus on industrial cybersecurity, IoT security.
            Keep the response concise, informative, and practical."""

            logger.debug("Sending request to Gemini API")
            response = model.generate_content(prompt)
            llm_response = response.text

            logger.debug(f"LLM Response received: {llm_response[:100]}...")
            dispatcher.utter_message(text=llm_response)

        except Exception as e:
            logger.error(f"Error in LLM fallback: {str(e)}")
            dispatcher.utter_message(text="I apologize, but I encountered an error processing your request. Please try rephrasing your question.")
            logger.debug(f"Full error details: {repr(e)}")
        
        return []