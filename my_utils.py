import time
from collections import deque
t = time.time()
prev_ts = deque([])

# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', printEnd="\r", t_update_interval=50):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """

    global t, prev_ts

    if iteration == 0:
        eta = ""
        prev_ts = deque([time.time()])
    else:
        prev_ts.append(time.time())
        part = len(prev_ts)
        delta = prev_ts[-1] - prev_ts[0]
        eta = ("eta: {0:." + str(decimals) +
               "f} seconds").format((delta / part) * (total - iteration))
        if part >= t_update_interval:
            prev_ts.popleft()

    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s %s' %
          (prefix, bar, percent, eta, suffix), end=printEnd)
    # if iteration != 0:
    #     print(part, total, other_part, delta)
    # Print New Line on Complete
    if iteration == total - 1:
        print()


def flatten(l):
    return [item for sl in l for item in sl]
