import numpy as np

"""
Defines classes for variable nodes, factor nodes and edges. 
"""

class ndimGaussian:
    def __init__(self, dimensionality, eta=None, lam=None):
        self.dim = dimensionality

        if eta is not None and len(eta)==self.dim:
            self.eta = eta
        else:
            self.eta = np.zeros(self.dim)

        if lam is not None and lam.shape==(self.dim, self.dim):
            self.lam = lam
        else:
            self.lam = np.matrix(np.zeros([self.dim, self.dim]))


class Graph:
    def __init__(self):

        self.var_nodes = []
        self.edges = []

        self.n_edges = 0
        self.n_nodes = 0

        self.eta_damping = 0.


class VariableNode:
    def __init__(self, variableID, dofs):
        self.variableID = variableID
        self.edges = []

        # Node variables are position of landmark in world frame. Initialize variable nodes at origin
        self.mu = np.zeros(dofs)
        self.Sigma = np.zeros([dofs,dofs])

        self.belief = ndimGaussian(dofs)

        self.prior = ndimGaussian(dofs)
        self.prior_lambda_end = -1  # -1 flag if the sigma of self.prior is prior_sigma_end
        self.prior_lambda_logdiff = -1

        self.dofs = dofs

    def update_belief(self):
        """
            Update local belief estimate by taking product of all incoming messages along all edges.
            Then put new belief estimate on all edges.
        """
        # Update local belief
        eta = self.prior.eta.copy()
        lam = self.prior.lam.copy()
        for edge in self.edges:
            message_ix = edge.adj_vIDs.index(self.variableID)
            eta_inward, lam_inward = edge.f2v_messages[message_ix].eta, edge.f2v_messages[message_ix].lam
            eta += eta_inward
            lam += lam_inward

        self.belief.eta = eta 
        self.belief.lam = lam
        self.Sigma = np.linalg.inv(self.belief.lam)
        self.mu = self.Sigma @ self.belief.eta
        
        # Put belief on adjacent edges
        for edge in self.edges:
            belief_ix = edge.adj_vIDs.index(self.variableID)
            edge.adj_beliefs[belief_ix].eta, edge.adj_beliefs[belief_ix].lam = self.belief.eta, self.belief.lam


class Edge:
    def __init__(self, edgeID, adj_var_nodes, measurement, gauss_noise_std, meas_fn, jac_fn, *args):
        self.edgeID = edgeID

        self.dofs_conditional_vars = 0
        self.adj_var_nodes = adj_var_nodes
        self.adj_vIDs = []
        self.adj_beliefs = []
        self.messages = []

        for adj_var_node in self.adj_var_nodes:
            self.dofs_conditional_vars += adj_var_node.dofs
            self.adj_vIDs.append(adj_var_node.variableID)
            self.adj_beliefs.append(ndimGaussian(adj_var_node.dofs))
            self.messages.append(ndimGaussian(adj_var_node.dofs))

        self.factor = ndimGaussian(self.dofs_conditional_vars)
        self.linpoint = np.zeros(self.dofs_conditional_vars)  # linearisation point

        self.measurement = measurement

        # Measurement model
        self.gauss_noise_std = gauss_noise_std
        self.meas_fn = meas_fn
        self.jac_fn = jac_fn
        self.args = args

    def compute_factor(self, linpoint=None):
        """
            Compute the factor given the linearisation point.
        """
        if linpoint is None:
            self.linpoint = []
            for belief in self.adj_beliefs:
                self.linpoint += np.linalg.inv(belief.lam) @ belief.eta
        else:
            self.linpoint = linpoint

        J = self.jac_fn(self.linpoint, *self.args)
        pred_measurement = self.meas_fn(self.linpoint, *self.args)
        # self.Sigma_meas = self.robustify_loss(pred_coords)

        meas_model_lambda = np.eye(2) / self.gauss_noise_std**2
        lambda_factor = J.t @ meas_model_lambda @ J
        eta_factor = (J.T @ meas_model_lambda) @ (J @ self.linpoint + self.measurement - pred_measurement)

        return eta_factor, lambda_factor

    def compute_messages(self, eta_damping):
        """
            Compute all outgoing messages.
        """
        eta_factor, lam_factor = self.factor.eta.copy(), self.factor.lam.copy()

        start_dim = 0
        for v in range(len(self.adj_vIDs)):

            # Take product of factor with incoming messages
            mess_start_dim = 0
            for var in range(len(self.adj_vIDs)):
                if var != v:
                    var_dofs = self.adj_var_nodes[var].dofs
                    eta_factor[mess_start_dim:mess_start_dim + var_dofs] += self.adj_beliefs[var].eta - self.messages[var].eta
                    lam_factor[mess_start_dim:mess_start_dim + var_dofs, mess_start_dim:mess_start_dim + var_dofs] += self.adj_beliefs[var].lam - self.messages[var].lam
                mess_start_dim += self.adj_var_nodes[var].dofs

            # Divide up parameters of distribution
            mess_dofs = self.adj_var_nodes[v].dofs
            eo = eta_factor[start_dim:start_dim + mess_dofs]
            eno = np.concatenate((eta_factor[:start_dim], eta_factor[start_dim + mess_dofs:]))

            loo = lam_factor[start_dim:start_dim + mess_dofs, start_dim:start_dim + mess_dofs]
            lono = np.hstack((lam_factor[start_dim:start_dim + mess_dofs, :start_dim],
                              lam_factor[start_dim:start_dim + mess_dofs, start_dim + mess_dofs:]))
            lnoo = np.vstack((lam_factor[:start_dim, start_dim:start_dim + mess_dofs],
                              lam_factor[start_dim + mess_dofs:, start_dim:start_dim + mess_dofs]))
            lnono = np.block([[lam_factor[:start_dim, :start_dim], lam_factor[:start_dim, start_dim + mess_dofs:]],
                              [lam_factor[start_dim + mess_dofs:, :start_dim], lam_factor[start_dim + mess_dofs:, start_dim + mess_dofs:]]])

            # Compute outgoing messages
            self.messages[v].lam = loo - lono @ np.linalg.inv(lnono) @ lnoo
            new_message_eta = eo - lono @ np.linalg.inv(lnono) @ eno
            self.messages[v].eta = (1 - eta_damping) * new_message_eta + eta_damping * self.messages[v].eta

            start_dim += self.adj_var_nodes[v].dofs
