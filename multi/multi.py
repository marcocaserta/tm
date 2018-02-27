from multiprocessing import Pool
from math import sqrt
import time
from timeit import default_timer as timer

#  def f(x):
#      return x+1
#
#  def main():
#
#      x = range(10)
#      y_serial = []
#      for i in x:
#          y_serial += [f(i)]
#
#      print("y_serial is ", y_serial)
#
#      p = Pool(5)
#      y_parallel = p.map(f,x)
#      print("y_parallel is ", y_parallel)
#      p.close()
#      p.join()
#

work = (["A", 5], ["B", 2], ["C", 1], ["D", 3])

def myfun(x):
    sqrt(x**2)
    #  return sqrt(x**2)
def myfun2(x):
    print("0 and 1", x[0], " ", x[1])
    sqrt(x[0]*x[1])

def sequential(x):
    """
    Execute operation sequentially. Note that if we want the result, we need
    to store it into a list.
    """
    #  result = list(map(myfun, x))
    list(map(myfun, x))
    #  return result

def parallel(x):
    p = Pool(4)
    #  result = p.map(myfun, x)
    #  p.map(myfun, x)
    p.map(myfun2, x)
    p.close()
    p.join()
    #  return result
    


def work_log(work_data):
    print(" Process %s waiting %s seconds" % (work_data[0], work_data[1]))
    time.sleep(int(work_data[1]))
    print(" Process %s Finished." % work_data[0])


def pool_handler():
    #  numbers = range(100000000)
    numbers = range(100)
    start = timer()
    resultSeq = sequential(numbers)
    end = timer()
    print("Sequential Time = ", end-start)

    x = []
    for i in range(10):
        x.append([i,i])

    start = timer()
    #  resultPar = parallel(numbers)
    resultPar = parallel(x)
    end = timer()
    print("Parallel Time = ", end-start)

    #  p = Pool(4)
    # note that map accepts only one list 
    #  p.map(work_log, work)


if __name__ == '__main__':
    pool_handler()

#  if __name__ == '__main__':
#      main()
