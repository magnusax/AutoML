# Temporarily mute a GazerMetaLearner
class Mute():
    def __init__(self, learner):
        self.learner = learner
        self.verbose = None
    def __enter__(self):
        self.verbose = self.learner.verbose
        if self.verbose > 0:
            self.learner.verbose = 0        
    def __exit__(self, type, value, traceback):
        self.learner.verbose = self.verbose