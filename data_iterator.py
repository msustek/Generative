import numpy as np

class MyBatchDataIter:
    def __init__(self, data, batch_size = 1, shuffle = False, start_ind = 0):
        self.orig_data = np.array(data)
        self.data = self.orig_data.copy()        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.start_ind = start_ind
        self.size = len(self.data)
        self.indices = np.arange(self.size)
        self.max_iter = self.size / self.batch_size

    def __iter__(self):
        if self.shuffle:
            self.indices = np.random.permutation(len(self.orig_data))
            self.data = self.orig_data[self.indices]
        self.n = 0        
        return self

    def __next__(self):                    
        if self.n < self.max_iter:
            self.n += 1                
            return self.data[(self.n-1)*self.batch_size:self.n*self.batch_size], self.start_ind + self.indices[(self.n-1)*self.batch_size:self.n*self.batch_size]
        else:
            raise StopIteration