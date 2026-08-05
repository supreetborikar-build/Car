class CommandProcessor:

    def __init__(self):

        self.manual = False

    def process(self, text):

        steering = None
        throttle = None

        if "manual mode" in text:

            self.manual = True
            print("Manual Mode Enabled")
            return None

        if "autonomous mode" in text:

            self.manual = False
            print("AI Driving Enabled")
            return None

        if not self.manual:
            return None

        ##################################

        if "forward" in text or "move forward" in text:

            steering = 0.0
            throttle = 0.35

        elif "left" in text:

            steering = -0.7
            throttle = 0.25

        elif "right" in text:

            steering = 0.7
            throttle = 0.25

        elif "stop" in text:

            steering = 0
            throttle = 0

        elif "reverse" in text:

            steering = 0
            throttle = -0.3

        elif "faster" in text:

            steering = 0
            throttle = 0.6

        elif "slower" in text:

            steering = 0
            throttle = 0.15

        else:

            return None

        return steering, throttle