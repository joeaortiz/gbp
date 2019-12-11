"""
N-dim Pose Graph Estimation.

Linear problem where we are estimating the position in N-dim space of a number of nodes.
Linear factors connect each node to the K closest nodes in the space.
The linear factors measure the distance between the nodes in each of the N dimensions.
"""

import numpy as np
from gbp import gbp
from gbp.factors import linear_displacement

np.random.seed(0)

# Parameters
N = 6  # dofs for each variable / dimensionality of space
max_meas_dist = 4.00
gauss_noise_std = 1.

n_varnodes = 50
K = 10

# Create priors
priors_mu = np.random.rand(n_varnodes, N) * 10  # grid goes from 0 to 10 along x and y axis
prior_sigma = 3 * np.eye(N)

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
    for j in np.array(dists).argsort()[1:K + 1]:  # As closest node is itself
        mu1 = priors_mu[j]
        # if np.linalg.norm(mu - mu1) < max_meas_dist and i<j:  # Second condition to avoid double counting
        n_edges += 1
        gt_measurements.append(mu - mu1)
        noisy_measurements.append(mu - mu1 + np.random.normal(0., gauss_noise_std, N))
        measurements_nodeIDs.append([i, j])

        num_edges_pernode[i] += 1
        num_edges_pernode[j] += 1



graph = gbp.FactorGraph(nonlinear_factors=False)

# Initialize variable nodes for frames with prior
for i in range(n_varnodes):
    new_var_node = gbp.VariableNode(i, N)
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
                            loss=None,
                            mahalanobis_threshold=2)

    graph.var_nodes[measurements_nodeIDs[f][0]].adj_factors.append(new_factor)
    graph.var_nodes[measurements_nodeIDs[f][1]].adj_factors.append(new_factor)
    graph.factors.append(new_factor)

graph.update_all_beliefs()
graph.compute_all_factors()

graph.n_var_nodes = n_varnodes
graph.n_factor_nodes = len(noisy_measurements)
graph.n_edges = 2 * len(noisy_measurements)

print(f'Number of variable nodes {graph.n_var_nodes}')
print(f'Number of edges per variable node {K}')
print(f'Number of dofs at each variable node {N}\n')

mu, sigma = graph.joint_distribution_cov()  # Get batch solution


for i in range(50):
    graph.synchronous_iteration()

    print(f'Iteration {i}   //   Energy {graph.energy():.4f}   //   '
          f'Av distance of means from MAP {np.linalg.norm(graph.get_means() - mu):4f}')
