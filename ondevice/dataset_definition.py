class EmagerDataset():
    def __init__(self, differential=False):
        self.name = "emager"
        self.sampling_rate = 1000
        self.sensors_dim = (4,16)
        self.nb_experiment = 10
        self.nb_class = 6
        self.utility_filtered = False
        self.differential = differential

class CapgmyoDataset():
    def __init__(self):
        self.name = "capgmyo"
        self.sampling_rate = 1000
        self.sensors_dim = (8,16)
        self.nb_experiment = 10
        self.nb_class = 8
        self.utility_filtered = True
    
class HyserDataset():
    def __init__(self):
        self.name = "hyser"
        self.sampling_rate = 2000 #(should be 2048 but 2000 is used for nb of samples purpose)
        self.sensors_dim = (16,16)
        self.nb_experiment = 2
        self.nb_class = 34
        self.utility_filtered = True
