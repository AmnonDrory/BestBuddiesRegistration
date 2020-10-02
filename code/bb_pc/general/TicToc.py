from time import time

tic_time = None
def tic():
    global tic_time
    tic_time = time()
def toc(string=None, outfile=None):
    global tic_time
    if string is None:
        string = ''
    string +=  " elapsed:" + str(time()-tic_time)
    print(string)
    if outfile is not None:
        print(string, file=outfile)
