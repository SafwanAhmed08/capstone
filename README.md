# Smart Factory Cybersecurity Assistant

A comprehensive AI-driven cybersecurity framework designed for IoT and Smart Factory environments. This project integrates **Deep Learning (NIDS)**, **Reinforcement Learning (Response)**, **Network Simulations**, and **Conversational AI** to detect, analyze, and mitigate cyber threats in real-time.

## 🚀 Key Features

*   **Intrusion Detection System (NIDS)**: A "Mixture of Experts" Deep Learning model capable of detecting various attacks including:
    *   DDoS (TCP SYN, HTTP, ICMP Floods)
    *   Port Scanning
    *   Man-in-the-Middle (MITM) attacks
*   **Conversational Interface**: A **Rasa-powered** chatbot that allows security analysts to query logs, ask for mitigation strategies (mapped to MITRE ATT&CK), and monitor device status via natural language.
*   **Traffic Simulation**: Uses **ns-3** to generate realistic attack scenarios and datasets for training.
*   **Hardware-in-the-Loop**: Includes Python scripts for monitoring physical IoT devices (Arduino + DHT sensors) and responding to network anomalies.
*   **Automated Mitigation**: Reinforcement Learning (RL) agents for dynamic thresholding and automated response.

## 📂 Project Structure

| Directory | Description |
| :--- | :--- |
| `conversational_ai/` | **Rasa** project for the chatbot (actions, domain, data, UI). |
| `inference/` & `inference_gpu.py` | Scripts for running the detection models on PCAP files. |
| `experts/` | Trained Keras models for specific attack types (the "Experts"). |
| `simulations/` | **C++ (ns-3)** code for generating attack traffic (DDoS, MITM, etc.). |
| `rl_model/` | Reinforcement Learning environment and agents for defense strategies. |
| `binary_classifier/` | Models for the initial benign vs. malicious traffic classification. |
| `deprecated_*/` | Older model versions and datasets. |
| `backend/` | (Implied) Python scripts connecting the hardware and server monitoring. |

---

## 🛠️ Prerequisites

*   **Python 3.8+**
*   **Wireshark/TShark**: Required for `pyshark` packet analysis.
*   **ns-3** (Optional): If you plan to run new network simulations.
*   **Arduino IDE** (Optional): For flashing the `dht11_reader.ino` if using hardware.

### Key Python Libraries
*   `tensorflow` (Deep Learning)
*   `rasa` (Chatbot)
*   `pyshark` (Packet Parsing)
*   `pandas`, `numpy`, `scikit-learn`
*   `pyserial` (Arduino communication)

---

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/capstone.git
    cd capstone
    ```

2.  **Install Python Dependencies**
    It is recommended to use a virtual environment.
    ```bash
    # Install dependencies for the conversational AI
    pip install -r conversational_ai/requirements.txt
    
    # Install additional dependencies for inference
    pip install tensorflow pyshark pyserial
    ```

---

## 🖥️ Usage

### 1. Run the Intrusion Detection System
To analyze network traffic (PCAP files) using the GPU-accelerated inference engine:

```bash
python inference_gpu.py
```
*   Follow the on-screen prompts to select the type of PCAP file (e.g., HTTP Flood, Port Scan) you wish to analyze.
*   The system will load the appropriate "Expert" model and output detection results.

### 2. Start the Conversational AI (Chatbot)
The chatbot requires two terminals to run: the Rasa server and the Action server.

**Terminal 1 (Action Server):**
```bash
cd conversational_ai
rasa run actions
```

**Terminal 2 (Rasa REST API / Shell):**
```bash
cd conversational_ai
# To chat in the terminal:
rasa shell

# OR to run the server for the web UI:
rasa run --enable-api --cors "*"
```
*   **Web UI**: Open `conversational_ai/ui/index.html` in your browser to interact with the bot visually.

### 3. Run Network Simulations
Simulations are written in C++ for the **ns-3** simulator.
1.  Copy the files from `simulations` to your ns-3 `scratch/` directory.
2.  Run with waf:
    ```bash
    ./ns3 run ddos_http1
    ```

### 4. Hardware Monitor (Optional)
If you have the Arduino set up:
1.  Connect the Arduino (check `ARDUINO_PORT` in `server_monitor.py`).
2.  Run the monitor:
    ```bash
    python server_monitor.py
    ```

---

## 🧠 Model Architecture

The core detection engine uses a hierarchical approach:
1.  **Binary Classifier**: First filters traffic as "Benign" or "Attack".
2.  **Mixture of Experts**: If flagged as an attack, specific models (Experts) classify the exact threat type (e.g., `expert_ddos_http.h5`, `expert_port_scanning.h5`).
3.  **Reinforcement Learning**: An RL agent `rl_threshold_env.py` learns the optimal threshold for triggering alerts based on network conditions.

## 🤝 Contributing

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
