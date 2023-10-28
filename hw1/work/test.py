class dataloader():
    arr = [0, 10, 20, 30, 40]
    _idx = -1
    def __init__(self):
        pass
    def __iter__(self):
        self._idx = -1
        return self
    def __next__(self):
        self._idx += 1
        if self._idx < len(self.arr):
            return self.arr[self._idx]
        else:
            raise StopIteration
    def __len__(self):
        return len(self.arr)
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.arr):
            raise IndexError
        return self.arr[idx]
if __name__ == "__main__" :
    data = dataloader()
    for i in data:
        print(i)

    for i in data:
        print(i)
