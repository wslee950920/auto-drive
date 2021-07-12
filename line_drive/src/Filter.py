from collections import deque

class Filter:
    def __init__(self, n):
        self.samples=n
        self.data=deque([])
        self.weights=list(range(1, n+1))
        self.alpha=0.3

    def add_sample(self, new_sample):
        if len(self.data)<self.samples:
            self.data.append(new_sample)
        else:
            self.data.popleft()
            self.data.append(new_sample)

    def get_mm(self):
        diff=float(sum(self.data))/len(self.data)

        return diff

    def get_wmm(self):
        s=0
        for i, x in enumerate(self.data):
            s+=x*self.weights[i]

        diff=float(s)/sum(self.weights[:len(self.data)])
        
        return diff

    def low_pass(self, x_esti, x_mess):
        x_esti=self.alpha*x_esti+(1-self.alpha)*x_mess
        
        return x_esti
