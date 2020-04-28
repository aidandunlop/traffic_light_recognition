CLASS_LABEL_MAP = {
    "go": 1,
    "goForward": 2,
    "goLeft": 3,
    "warning": 4,
    "warningLeft": 5,
    "stop": 6,
    "stopLeft": 7,
}

REVERSED_CLASS_LABEL_MAP = dict((reversed(item) for item in CLASS_LABEL_MAP.items()))

bad_images = ["nightClip3--00172.jpg"]
