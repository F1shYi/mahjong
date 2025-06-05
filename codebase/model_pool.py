from multiprocessing.shared_memory import SharedMemory, ShareableList
import _pickle as cPickle
import time

class ModelPoolServer:
    
    def __init__(self, capacity, name):
        self.capacity = capacity
        self.n = 0
        self.model_list = [None] * capacity
        # shared_model_list: N metadata {id, _addr} + n
        metadata_size = 1024
        self.shared_model_list = ShareableList([' ' * metadata_size] * capacity + [self.n], name = name)
    def cleanup(self):
        for m in self.model_list:
            if m and 'memory' in m:
                try:
                    m['memory'].close()
                    m['memory'].unlink()
                except Exception:
                    pass
    def push(self, state_dict, metadata = {}):
        n = self.n % self.capacity
        if self.model_list[n]:
            old_mem = self.model_list[n].get('memory')
            if old_mem:
                try:
                    old_mem.close()
                    old_mem.unlink()
                except FileNotFoundError:
                    pass  # 可能已被別的進程釋放
                except Exception as e:
                    print(f"[Warning] Failed to cleanup memory: {e}")
        
        # data = cPickle.dumps(state_dict) # model parameters serialized to bytes
        # name = 'model-%d' % self.n
        # memory = SharedMemory(create = True, size = len(data), name = name)
        # # print(memory.buf.dtype)
        # # print(data.dtype)
        # # memory.buf[:] = data[:]
        # memory.buf[:len(data)] = data
        # # print('Created model', self.n, 'in shared memory', memory.name)
        
        # metadata = metadata.copy()
        # metadata['_addr'] = memory.name
        # metadata['id'] = self.n
        # self.model_list[n] = metadata
        # self.shared_model_list[n] = cPickle.dumps(metadata)
        # self.n += 1
        # self.shared_model_list[-1] = self.n
        # metadata['memory'] = memory
        data = cPickle.dumps(state_dict)
        name = f'model-{self.n}'
        try:
            memory = SharedMemory(create=True, size=len(data), name=name)
            memory.buf[:len(data)] = data

            metadata = metadata.copy()
            metadata['_addr'] = memory.name
            metadata['id'] = self.n
            metadata['memory'] = memory

            self.model_list[n] = metadata
            self.shared_model_list[n] = cPickle.dumps(metadata)
            self.n += 1
            self.shared_model_list[-1] = self.n

        except Exception as e:
            print(f"[Error] Failed to create shared memory: {e}")

class ModelPoolClient:
    
    def __init__(self, name):
        while True:
            try:
                self.shared_model_list = ShareableList(name = name)
                n = self.shared_model_list[-1]
                break
            except:
                time.sleep(0.1)
        self.capacity = len(self.shared_model_list) - 1
        self.model_list = [None] * self.capacity
        self.n = 0
        self._update_model_list()
    
    def _update_model_list(self):
        n = self.shared_model_list[-1]
        if n > self.n:
            # new models available, update local list
            for i in range(max(self.n, n - self.capacity), n):
                self.model_list[i % self.capacity] = cPickle.loads(self.shared_model_list[i % self.capacity])
            self.n = n
    
    def get_model_list(self):
        self._update_model_list()
        model_list = []
        if self.n >= self.capacity:
            model_list.extend(self.model_list[self.n % self.capacity :])
        model_list.extend(self.model_list[: self.n % self.capacity])
        return model_list
    
    def get_latest_model(self):
        self._update_model_list()
        while self.n == 0:
            time.sleep(0.1)
            self._update_model_list()
        return self.model_list[(self.n + self.capacity - 1) % self.capacity]
        
    def load_model(self, metadata):
        self._update_model_list()
        n = metadata['id']
        if n < self.n - self.capacity: return None
        memory = SharedMemory(name = metadata['_addr'])
        state_dict = cPickle.loads(memory.buf)
        return state_dict