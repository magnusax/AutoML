import sys

def show_progress(it, total_fits):
    """
    Show dynamical progress bar.
    """
    p = (it+1)*(100/float(total_fits))
    sys.stdout.write('\r')
    sys.stdout.write("[%-100s] %d%%" % ('=' * int(p), p))
    sys.stdout.flush()
    return
