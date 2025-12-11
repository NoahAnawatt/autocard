import pyautogui
import pyperclip
import time

# File to save copied text
output_file = "copied_text.txt"

print("Text copier running. Select text and press Ctrl+C to copy it...")
print("Press Ctrl+C again to copy new text, script will append to file. Ctrl+C to stop.")
print("Click off this window NOW")
time.sleep(3)

previous_text = ""

try:
    while True:
        pyautogui.hotkey('command', 'c') # for windows, change command to ctrl
        time.sleep(0.1)  

        text = pyperclip.paste()

        if text and text != previous_text:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(text + "\n")
            print(f"Saved: {text}")
            previous_text = text

        time.sleep(0.5)  

except KeyboardInterrupt:
    print("\nExiting script.")

