#-*- using:utf-8 -*-
import time

if __name__ == '__main__':
    start = time.time()
    
    #この間に何かしらの処理
    
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"
