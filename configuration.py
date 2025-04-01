class CausalTGANConfig(object):
    def __init__(self, causal_graph,
                 z_dim, pac_num, D_iter):
        self.causal_graph = causal_graph
        self.z_dim = z_dim
        self.pac_num = pac_num
        self.D_iter = D_iter

class TrainingOptions:
    """
    Configuration options for the training
    """

    def __init__(self, lamb, edges, label,
                 batch_size, number_of_epochs,
                 runs_folder,
                 experiment_name,
                 sensitive_attribute):
        self.lamb = lamb
        self.edges = edges
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.runs_folder = runs_folder
        self.experiment_name = experiment_name
        self.label = label
        self.sensitive_attribute = sensitive_attribute
