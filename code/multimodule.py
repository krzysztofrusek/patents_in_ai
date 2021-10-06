import glob
import os
import tensorflow as tf

class MultiModule(tf.Module):
    def __init__(self, components:list, name=None):
        super(MultiModule, self).__init__(name=name)
        self.components = components
    def __call__(self, x):
        return [c(x) for c in self.components]

    @staticmethod
    def load(cls, pattern='plg/checkpoint/*'):
        bootstrap=[]

        for p in glob.glob(pattern):
            model = cls(name='b'+p.split('/')[-1])
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.read(os.path.join(p,'Estimator'))
            bootstrap.append(model)

        mm = MultiModule(bootstrap)
        return mm
