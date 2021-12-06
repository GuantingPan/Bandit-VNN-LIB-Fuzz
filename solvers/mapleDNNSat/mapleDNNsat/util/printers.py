import sys
from .. import config


def help_msg(): config.print_help()


def warning(*args, **kwargs):
    print("[mapleDNNsat] warning:", end=' ')
    for arg in args:
        try:
            print(arg, end=' ')
        except:
            warning("Failed to print in warning.")
    for karg in kwargs:
        try:
            print("{}={}".format(karg, kwargs[karg]), end=' ')
        except:
            warning("Failed to print in warning.")
    print()


def die(*args, help=False, **kwargs):
    if help:
        help_msg()
    print("[mapleDNNsat] error:", end=' ')
    for arg in args:
        try:
            print(arg, end=' ')
        except:
            warning("Failed to print in warning.")
    for karg in kwargs:
        try:
            print("{}={}".format(karg, kwargs[karg]), end=' ')
        except:
            warning("Failed to print in warning.")
    print()
    sys.exit(1)
