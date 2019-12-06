import numpy as np
from gbp import gbp
from gbp.factors import linear_displacement

np.random.seed(0)

var_dofs = 2
max_meas_dist = 4.0
gauss_noise_std = 1

n_varnodes = 100
edges_per_node = 200

# Create priors
priors_mu = np.random.rand(n_varnodes, var_dofs) * 10  # grid goes from 0 to 10 along x and y axis
prior_sigma = 3 * np.eye(var_dofs)

prior_lambda = np.linalg.inv(prior_sigma)
priors_lambda = [prior_lambda] * n_varnodes
priors_eta = []
for mu in priors_mu:
    priors_eta.append(np.dot(prior_lambda, mu))

# Generate connections between variables
gt_measurements, noisy_measurements = [], []
measurements_nodeIDs = []
num_edges_pernode = np.zeros(n_varnodes)
n_edges = 0

for i, mu in enumerate(priors_mu):
    dists = []
    for j, mu1 in enumerate(priors_mu):
        dists.append(np.linalg.norm(mu - mu1))
    for j in np.array(dists).argsort()[1:edges_per_node + 1]:  # As closest node is itself
        mu1 = priors_mu[j]
        # if np.linalg.norm(mu - mu1) < max_meas_dist and i<j:  # Second condition to avoid double counting
        n_edges += 1
        gt_measurements.append(mu - mu1)
        noisy_measurements.append(mu - mu1 + np.random.normal(0., gauss_noise_std, var_dofs))
        measurements_nodeIDs.append([i, j])

        num_edges_pernode[i] += 1
        num_edges_pernode[j] += 1



graph = gbp.FactorGraph(nonlinear_factors=False)

# Initialize variable nodes for frames with prior
for i in range(n_varnodes):
    new_var_node = gbp.VariableNode(i, 2)
    new_var_node.prior.eta = priors_eta[i]
    new_var_node.prior.lam = priors_lambda[i]
    graph.var_nodes.append(new_var_node)

for f, measurement in enumerate(noisy_measurements):
    new_factor = gbp.Factor(f,
                            [graph.var_nodes[measurements_nodeIDs[f][0]], graph.var_nodes[measurements_nodeIDs[f][1]]],
                            measurement,
                            gauss_noise_std,
                            linear_displacement.meas_fn,
                            linear_displacement.jac_fn,
                            loss='huber',
                            mahalanobis_threshold=2)

    graph.var_nodes[measurements_nodeIDs[f][0]].adj_factors.append(new_factor)
    graph.var_nodes[measurements_nodeIDs[f][1]].adj_factors.append(new_factor)
    graph.factors.append(new_factor)

graph.update_all_beliefs()
graph.compute_all_factors()

graph.n_var_nodes = n_varnodes
graph.n_factor_nodes = len(noisy_measurements)
graph.n_edges = 2 * len(noisy_measurements)


for i in range(10):
    graph.synchronous_iteration(robustify=True)
    print(f'Iteration {i} // Energy {graph.energy():.4f}')
