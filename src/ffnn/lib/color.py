# ANSI color codes as constants

RESET = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"

# Foreground colors
BLACK = "\033[30m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

def progress_bar(progress, total, length=20):
    percent = progress / total
    bar = "█" * int(length * percent) + " " * (length - int(length * percent))

    return bar