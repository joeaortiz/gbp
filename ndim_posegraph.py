"""
N-dim Pose Graph Estimation.

Linear problem where we are estimating the position in N-dim space of a number of nodes.
Linear factors connect each node to the M closest nodes in the space.
The linear factors measure the distance between the nodes in each of the N dimensions.
"""

import numpy as np
import argparse

from gbp import gbp
from gbp.factors import linear_displacement

np.random.seed(0)


parser = argparse.ArgumentParser()
parser.add_argument("--n_varnodes", type=int, default=50,
                    help="Number of variable nodes.")
parser.add_argument("--dim", type=int, default=6,
                    help="Dimensionality of space nodes exist in (dofs of variables)")
parser.add_argument("--M", type=int, default=10,
                    help="Each node is connected to its k closest neighbours by a measurement.")
parser.add_argument("--gauss_noise_std", type=float, default=1.,
                    help="Standard deviation of Gaussian noise added to measurement model (pixels)")

parser.add_argument("--n_iters", type=int, default=50,
                    help="Number of iterations of GBP")

args = parser.parse_args()
print('Configs: \n', args)


# Create priors
priors_mu = np.random.rand(args.n_varnodes, args.dim) * 10  # grid goes from 0 to 10 along x and y axis
prior_sigma = 3 * np.eye(args.dim)

prior_lambda = np.linalg.inv(prior_sigma)
priors_lambda = [prior_lambda] * args.n_varnodes
priors_eta = []
for mu in priors_mu:
    priors_eta.append(np.dot(prior_lambda, mu))

# Generate connections between variables
gt_measurements, noisy_measurements = [], []
measurements_nodeIDs = []
num_edges_per_node = np.zeros(args.n_varnodes)
n_edges = 0

for i, mu in enumerate(priors_mu):
    dists = []
    for j, mu1 in enumerate(priors_mu):
        dists.append(np.linalg.norm(mu - mu1))
    for j in np.array(dists).argsort()[1:args.M + 1]:  # As closest node is itself
        mu1 = priors_mu[j]
        if [j, i] not in measurements_nodeIDs:  # To avoid double counting
            n_edges += 1
            gt_measurements.append(mu - mu1)
            noisy_measurements.append(mu - mu1 + np.random.normal(0., args.gauss_noise_std, args.dim))
            measurements_nodeIDs.append([i, j])

            num_edges_per_node[i] += 1
            num_edges_per_node[j] += 1


graph = gbp.FactorGraph(nonlinear_factors=False)

# Initialize variable nodes for frames with prior
for i in range(args.n_varnodes):
    new_var_node = gbp.VariableNode(i, args.dim)
    new_var_node.prior.eta = priors_eta[i]
    new_var_node.prior.lam = priors_lambda[i]
    graph.var_nodes.append(new_var_node)

for f, measurement in enumerate(noisy_measurements):
    new_factor = gbp.Factor(f,
                            [graph.var_nodes[measurements_nodeIDs[f][0]], graph.var_nodes[measurements_nodeIDs[f][1]]],
                            measurement,
                            args.gauss_noise_std,
                            linear_displacement.meas_fn,
                            linear_displacement.jac_fn,
                            loss=None,
                            mahalanobis_threshold=2)

    graph.var_nodes[measurements_nodeIDs[f][0]].adj_factors.append(new_factor)
    graph.var_nodes[measurements_nodeIDs[f][1]].adj_factors.append(new_factor)
    graph.factors.append(new_factor)

graph.update_all_beliefs()
graph.compute_all_factors()

graph.n_var_nodes = args.n_varnodes
graph.n_factor_nodes = len(noisy_measurements)
graph.n_edges = 2 * len(noisy_measurements)

print(f'Number of variable nodes {graph.n_var_nodes}')
print(f'Number of edges per variable node {args.M}')
print(f'Number of dofs at each variable node {args.dim}\n')

mu, sigma = graph.joint_distribution_cov()  # Get batch solution


for i in range(args.n_iters):
    graph.synchronous_iteration()

    print(f'Iteration {i}   //   Energy {graph.energy():.4f}   //   '
          f'Av distance of means from MAP {np.linalg.norm(graph.get_means() - mu):4f}')
