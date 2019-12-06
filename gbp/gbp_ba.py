import numpy as np
from gbp import gbp
from gbp.factors import reprojection
from data_handling import read_balfile

"""
    Defines child classes of GBP parent classes for Bundle Adjustment.
    Also defines the function to create the factor graph. 
"""


class BAFactorGraph(gbp.FactorGraph):
    def __init__(self, **kwargs):
        gbp.FactorGraph.__init__(self, nonlinear_factors=True, **kwargs)

        self.cam_nodes = []
        self.lmk_nodes = []
        self.var_nodes = self.cam_nodes + self.lmk_nodes

    def generate_priors_var(self, weaker_factor=100):
        """
            Sets automatically the std of the priors such that standard deviations of prior factors are a factor of weaker_factor
            weaker than the standard deviations of the adjacent factors.
            NB. Jacobian of measurement function effectively sets the scale of the factors.
        """
        for var_node in self.var_nodes:
            max_factor_lam = 0.
            for factor in var_node.adj_factors:
                max_factor_lam = max(max_factor_lam, np.max(factor.factor.lam))

            lam_prior = np.eye(var_node.dofs) * max_factor_lam / (weaker_factor ** 2)
            var_node.prior.lam = lam_prior
            var_node.prior.eta = lam_prior @ var_node.mu

    def weaken_priors(self, weakening_factor):
        """
            Increases the variance of the priors by the specified factor.
        """
        for var_node in self.var_nodes:
            var_node.prior.eta *= weakening_factor
            var_node.prior.lam *= weakening_factor

    def set_priors_var(self, priors):
        """
            Means of prior have already been set when graph was initialised. Here we set the variance of the prior factors.
            priors: list of length number of variable nodes where each element is the covariance matrix of the prior
                    distribution for that variable node.
        """
        for v, var_node in enumerate(self.var_nodes):
            var_node.prior.lam = np.linalg.inv(priors[v])
            var_node.prior.eta = var_node.prior.lam @ var_node.mu

    def compute_residuals(self):
        residuals = []
        for factor in self.factors:
            if isinstance(factor, ReprojectionFactor):
                residuals += list(factor.compute_residual())
        return residuals

    def are(self):
        """
            Computes the Average Reprojection Error across the whole graph.
        """
        are = 0
        for factor in self.factors:
            if isinstance(factor, ReprojectionFactor):
                are += np.linalg.norm(factor.compute_residual())
        return are / len(self.factors)


class LandmarkVariableNode(gbp.VariableNode):
    def __init__(self, variable_id, dofs, l_id=None):
        gbp.VariableNode.__init__(self, variable_id, dofs)
        self.l_id = l_id


class FrameVariableNode(gbp.VariableNode):
    def __init__(self, variable_id, dofs, c_id=None):
        gbp.VariableNode.__init__(self, variable_id, dofs)
        self.c_id = c_id


class ReprojectionFactor(gbp.Factor):
    def __init__(self, factor_id, adj_var_nodes, measurement, gauss_noise_std, loss, Nstds, K):

        gbp.Factor.__init__(self, factor_id, adj_var_nodes, measurement, gauss_noise_std,
                            reprojection.meas_fn, reprojection.jac_fn, loss, Nstds, K)

    def reprojection_err(self):
        """
            Returns the reprojection error at the factor in pixels.
        """
        return np.linalg.norm(self.compute_residual())


def create_ba_graph(bal_file, configs):
    """
        Create graph object from bal style file.
    """
    n_keyframes, n_points, n_edges, cam_means, lmk_means, measurements, measurements_camIDs, \
            measurements_lIDs, K = read_balfile.read_balfile(bal_file)

    graph = BAFactorGraph(eta_damping=configs['eta_damping'],
                         beta=configs['beta'],
                         num_undamped_iters=configs['num_undamped_iters'],
                         min_linear_iters=configs['min_linear_iters'])

    variable_id = 0
    factor_id = 0
    n_edges = 0

    # Initialize variable nodes for frames with prior
    for m, init_params in enumerate(cam_means):
        new_cam_node = FrameVariableNode(variable_id, 6, m)
        new_cam_node.mu = init_params
        graph.cam_nodes.append(new_cam_node)
        variable_id += 1

    # Initialize variable nodes for landmarks with prior
    for l, init_loc in enumerate(lmk_means):
        new_lmk_node = LandmarkVariableNode(variable_id, 3, l)
        new_lmk_node.mu = init_loc
        graph.lmk_nodes.append(new_lmk_node)
        variable_id += 1

    # Initialize measurement factor nodes and the required edges.
    for camID in range(n_keyframes):
        for f, measurement in enumerate(measurements):
            if measurements_camIDs[f] == camID:
                cam_node = graph.cam_nodes[camID]
                lmk_node = graph.lmk_nodes[measurements_lIDs[f]]

                new_factor = ReprojectionFactor(factor_id, [cam_node, lmk_node], measurement,
                                                       configs['gauss_noise_std'], configs['loss'], configs['Nstds'], K)
                linpoint = np.concatenate((cam_node.mu, lmk_node.mu))
                new_factor.compute_factor(linpoint)
                cam_node.adj_factors.append(new_factor)
                lmk_node.adj_factors.append(new_factor)

                graph.factors.append(new_factor)
                factor_id += 1
                n_edges += 2

    graph.n_factor_nodes = factor_id
    graph.n_var_nodes = variable_id
    graph.var_nodes = graph.cam_nodes + graph.lmk_nodes
    graph.n_edges = n_edges

    return graph


