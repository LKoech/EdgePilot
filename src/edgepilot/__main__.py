"""Allow running as `python -m edgepilot`."""

import sys

if __name__ == "__main__":
    if "--voice" in sys.argv:
        from edgepilot.main import voice_mode

        voice_mode()
    else:
        from edgepilot.main import interactive_loop

        interactive_loop()
