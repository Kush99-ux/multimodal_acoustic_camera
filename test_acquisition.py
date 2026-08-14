import time

from acoustic.acquisition import AudioAcquisition

with AudioAcquisition() as audio:
    print("Recording for 5 seconds...\n")

    start = time.time()
    last_print = 0.0

    while time.time() - start < 5:
        frame = audio.read()

        # Print diagnostics every 1 second
        if time.time() - last_print >= 1.0:
            print(f"Overall Level: {audio.rms_db():.2f} dB")
            audio.print_channel_status()

            inactive = audio.check_microphone_health()

            if inactive:
                print(f"Inactive channels: {[ch + 1 for ch in inactive]}")
            else:
                print("All microphones active")

            print()
            last_print = time.time()

        time.sleep(0.05)

print("Done.")