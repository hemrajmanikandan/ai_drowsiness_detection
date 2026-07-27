"""
Handles the physical alert (Arduino buzzer) so main.py doesn't have to
know about serial connections directly.
"""
import time

from src import config


class BuzzerAlert:
    def __init__(self, port=config.ARDUINO_PORT, baudrate=config.ARDUINO_BAUDRATE):
        self.arduino = None
        self.buzzer_on = False

        if not port:
            print("[INFO] No ARDUINO_PORT configured — running without buzzer hardware.")
            return

        try:
            import serial  # imported lazily so the app still runs without pyserial installed
            self.arduino = serial.Serial(port, baudrate)
            time.sleep(2)
            print(f"[INFO] Arduino connected on {port}")
        except Exception as e:
            print(f"[WARNING] Arduino not connected: {e}")

    def set(self, drowsy: bool):
        """Turn the buzzer on/off only when the state actually changes."""
        if drowsy and not self.buzzer_on:
            if self.arduino:
                self.arduino.write(b'1')
            self.buzzer_on = True
        elif not drowsy and self.buzzer_on:
            if self.arduino:
                self.arduino.write(b'0')
            self.buzzer_on = False
