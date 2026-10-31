#!/usr/bin/env python3
"""Mostra as mensagens de um tópico MQTT (credenciais em MQTT_USER/MQTT_PASS)."""
import argparse
import os

import paho.mqtt.client as mqtt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broker", default=os.getenv("MQTT_BROKER", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    parser.add_argument("--topic", default="lab/+/uplink")
    args = parser.parse_args()

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            client.subscribe(args.topic)

    def on_message(client, userdata, message):
        text = message.payload.decode("utf-8", errors="replace")
        # Caracteres de controle (sequências ANSI, por exemplo) viram escapes visíveis.
        text = "".join(c if c.isprintable() else repr(c)[1:-1] for c in text)
        print(f"[{message.topic}] {text}")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    if os.getenv("MQTT_USER"):
        client.username_pw_set(os.environ["MQTT_USER"], os.getenv("MQTT_PASS", ""))
    try:
        client.connect(args.broker, args.port, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        pass
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
