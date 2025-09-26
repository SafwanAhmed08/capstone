import serial
import time

# --- Configuration ---
# Set your Arduino's serial port.
# Windows: 'COM3', 'COM4', etc.
# Mac/Linux: '/dev/tty.usbmodemXXXX' or '/dev/tty.acmXXXX'
ARDUINO_PORT = 'COM3'
BAUD_RATE = 9600

# --- Script Variables ---
attack_counter = 0
remediation_sent = False

print("--- Starting Smart Factory Monitor (Final Hardware Reset Version) ---")
try:
    # --- Connection Logic ---
    # Create the serial object first.
    ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=4)
    # Set dtr=False AFTER connecting. This prevents an immediate reset
    # and is compatible with all versions of pyserial.
    ser.dtr = False
    time.sleep(2) # Wait for the connection to stabilize.

    # --- Main Loop ---
    while True:
        # Read a line from the Arduino.
        line = ser.readline().decode('utf-8').strip()

        # If the line is empty, the Arduino might be busy or rebooting.
        if not line:
            if remediation_sent:
                print("...Waiting for sensor to reboot...")
            continue

        # Print any diagnostic messages from the Arduino (like "NORMAL MODE").
        if not line.startswith('T:'):
            print(f"[Arduino]: {line}")
            continue

        # --- Data Parsing and Verification ---
        parts = line.split(',')
        temp = float(parts[0].split(':')[1])
        humidity = float(parts[1].split(':')[1])
        checksum_rcv = int(parts[2].split(':')[1])
        is_data_valid = (int(temp) + int(humidity)) == checksum_rcv

        if is_data_valid:
            if remediation_sent:
                # PHASE 3: VERIFICATION AND EXIT
                # This block runs after the reset is successful.
                print("\n==================================================")
                print("✅ HARDWARE RESET SUCCESSFUL!")
                print(f"THE ACTUAL TEMPERATURE IS NOW: {temp}°C")
                print("==================================================")
                break  # Exit the program.
            else:
                # PHASE 1: NORMAL OPERATION
                print(f"Temp: {temp}°C ... Status: Data Integrity OK.")
        else:
            # PHASE 2: ATTACK DETECTION AND REMEDIATION
            attack_counter += 1
            print(f"Temp: {temp}°C ... 🚨 ALERT #{attack_counter}: DATA TAMPERING DETECTED!")

            # After 5 consecutive attacks, trigger the hardware reset.
            if attack_counter >= 5 and not remediation_sent:
                print("\n🤖 AI: Persistent attack detected. Forcing hardware reset of sensor node...")

                # Toggle the DTR line to force the Arduino to reset.
                ser.dtr = True
                time.sleep(0.1)
                ser.dtr = False

                remediation_sent = True
                print("...Reset command sent. Waiting for node to come back online...")

except serial.SerialException as e:
    print(f"\nERROR: Could not connect to the Arduino on port {ARDUINO_PORT}.")
    print(f"Please check the port and ensure no other programs are using it.")
except Exception as e:
    print(f"\nAn error occurred: {e}")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
    print("--- Monitor closed. ---")