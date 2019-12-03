import numpy as np
import matplotlib.pylab as plt
import time

np.random.seed(1)

# Parameters
var_dofs = 6
max_meas_dist = 4.001
meas_variance = 0.5

n_varnodes = 1000
edges_per_node = 200

# Create priors
priors_mu = np.random.rand(n_varnodes, var_dofs) * 10  # grid goes from 0 to 10 along x and y axis
prior_sigma = 3 * np.eye(var_dofs)

prior_lambda = np.linalg.inv(prior_sigma)
priors_lambda = list(prior_lambda.flatten()) * n_varnodes
priors_eta = []
for mu in priors_mu:
    priors_eta += list(np.dot(prior_lambda, mu))


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
        gt_measurements += list(mu - mu1)
        noisy_measurements += list(mu - mu1 + np.random.normal(0., np.sqrt(meas_variance), var_dofs))
        measurements_nodeIDs += [i, j]

        num_edges_pernode[i] += 1
        num_edges_pernode[j] += 1

# Generate factor potentials
J = np.hstack((-np.eye(var_dofs), np.eye(var_dofs)))
factor_potential_lambda = np.dot(J.T, J) / meas_variance
factor_potentials_eta = []
for i in range(int(len(noisy_measurements) / var_dofs)):
    factor_potentials_eta += list(np.dot(J.T, np.array(noisy_measurements[i*var_dofs:i*var_dofs + var_dofs])) / meas_variance)

# factor_potentials_lambda = list(factor_potential_lambda.flatten()) * n_edges

# Store factor potentials 4x4 matrix as 4 2x2 block matrices one after the other
factor_potentials_lambda = []
factor_potentials_lambda += list(factor_potential_lambda[:var_dofs, :var_dofs].flatten())
factor_potentials_lambda += list(factor_potential_lambda[:var_dofs, var_dofs:].flatten())
factor_potentials_lambda += list(factor_potential_lambda[var_dofs:, :var_dofs].flatten())
factor_potentials_lambda += list(factor_potential_lambda[var_dofs:, var_dofs:].flatten())
factor_potentials_lambda = factor_potentials_lambda * n_edges

print('nvarnodes', n_varnodes)
print('nedges', n_edges)
print('num edges per node', num_edges_pernode)
# print(min(num_edges_pernode), max(num_edges_pernode), np.mean(num_edges_pernode))
print('n edges + nvarnodes', n_edges + n_varnodes)

# # Save priors
# with open("../data/priors_eta.txt", 'w') as f:
#     for entry in priors_eta:
#         f.write(str(entry) + '\n')
# with open("../data/priors_lambda.txt", 'w') as f:
#     for entry in priors_lambda:
#         f.write(str(entry) + '\n')
#
# # Save measurement information
# with open("../data/gt_measurements.txt", 'w') as f:
#     for entry in gt_measurements:
#         f.write(str(entry) + '\n')
# with open("../data/noisy_measurements.txt", 'w') as f:
#     for entry in noisy_measurements:
#         f.write(str(entry) + '\n')
# meas_variances = [meas_variance] * n_edges
# with open("../data/meas_variances.txt", 'w') as f:
#     for entry in meas_variances:
#         f.write(str(entry) + '\n')
# with open("../data/measurements_nodeIDs.txt", 'w') as f:
#     for entry in measurements_nodeIDs:
#         f.write(str(entry) + '\n')
#
# # Save factor potentials
# with open("../data/factor_potentials_lambda.txt", 'w') as f:
#     for entry in factor_potentials_lambda:
#         f.write(str(entry) + '\n')
# with open("../data/factor_potentials_eta.txt", 'w') as f:
#     for entry in factor_potentials_eta:
#         f.write(str(entry) + '\n')
#
# with open("../data/num_edges_pernode.txt", 'w') as f:
#     for entry in num_edges_pernode:
#         f.write(str(entry) + '\n')
#
# with open("../data/n_varnodes.txt", 'w') as f:
#     f.write(str(n_varnodes))
# with open("../data/n_edges.txt", 'w') as f:
#     f.write(str(n_edges))
# with open("../data/var_dofs.txt", 'w') as f:
#     f.write(str(var_dofs))
# var_nodes_dofs = [var_dofs] * n_varnodes
# with open("../data/var_nodes_dofs.txt", 'w') as f:
#     for entry in var_nodes_dofs:
#         f.write(str(entry) + '\n')
#


"""
Defines classes for variable nodes, factor nodes and edges. 
"""

class ndimGaussian:
    def __init__(self, dimensionality, eta=None, Lambda=None):
        self.dim = dimensionality

        if eta is not None and eta.shape == (1, self.dim):
            self.eta = eta
        else:
            self.eta = np.zeros(self.dim)

        if Lambda is not None and Lambda.shape == (self.dim, self.dim):
            self.Lambda = Lambda
        else:
            self.Lambda = np.zeros([self.dim, self.dim])


class Graph:
    def __init__(self):
        self.var_nodes = []
        self.edges = []

        self.n_edges = 0
        self.n_nodes = 0

        self.eta_damping = 0.
        self.Lambda_damping = 0.


class VariableNode:
    def __init__(self, variableID, dofs):
        self.variableID = variableID
        self.edges = []

        # Node variables are position of landmark in world frame. Initialize variable nodes at origin
        self.mu = np.zeros(dofs)
        self.Sigma = np.zeros([dofs, dofs])

        self.belief = ndimGaussian(dofs)

        self.prior = ndimGaussian(dofs)

        self.dofs = dofs

    def updateBelief(self, prob_damping=0., dl=False):
        """ Update local belief estimate by taking product of all incoming messages along all edges.
            Then put new belief estimate on all edges. """

        # Update belief
        eta = self.prior.eta.copy()
        Lambda = self.prior.Lambda.copy()
        for edge in self.edges:
            if edge.var0ID == self.variableID:
                eta_inward, Lambda_inward = edge.Message10.eta, edge.Message10.Lambda
            elif edge.var1ID == self.variableID:
                eta_inward, Lambda_inward = edge.Message01.eta, edge.Message01.Lambda
            eta += eta_inward
            Lambda += Lambda_inward

        self.belief.eta = eta
        self.belief.Lambda = Lambda
        self.Sigma = np.linalg.inv(self.belief.Lambda)
        self.mu = (self.Sigma @ self.belief.eta.T).T

        # Put belief on edges
        for edge in self.edges:
            if edge.var0ID == self.variableID:
                edge.Belief0.eta, edge.Belief0.Lambda = self.belief.eta, self.belief.Lambda
            elif edge.var1ID == self.variableID:
                edge.Belief1.eta, edge.Belief1.Lambda = self.belief.eta, self.belief.Lambda

# ----------------------- Edge Class ---------------------------

class Edge:
    def __init__(self, edgeID, var_node0, var_node1, factor_eta, factor_lambda):
        # Node 0 is camera, node 1 is landmark

        self.edgeID = edgeID

        self.var_node0 = var_node0
        self.var_node1 = var_node1
        self.var0ID = var_node0.variableID
        self.var1ID = var_node1.variableID
        self.n_vars0 = var_node0.dofs
        self.n_vars1 = var_node1.dofs

        self.potential = ndimGaussian(self.n_vars0 + self.n_vars1)
        self.potential.eta = factor_eta
        self.potential.Lambda = factor_lambda

        self.Belief0 = ndimGaussian(self.n_vars0)
        self.Belief1 = ndimGaussian(self.n_vars1)

        self.Message01 = ndimGaussian(self.n_vars1)
        self.Message10 = ndimGaussian(self.n_vars0)

    def computeMessages(self, eta_damping=0., Lambda_damping=0.):
        var0_dofs = self.n_vars0

        eta_factor, Lambda_factor = self.potential.eta, self.potential.Lambda

        # Compute messages
        newMessage01_eta = eta_factor[var0_dofs:] - np.dot(
            np.dot(Lambda_factor[var0_dofs:, 0:var0_dofs], np.linalg.inv(Lambda_factor[0:var0_dofs, 0:var0_dofs] + \
                                                                       self.Belief0.Lambda - self.Message10.Lambda)),
            (eta_factor[0:var0_dofs] + self.Belief0.eta - self.Message10.eta).T).T
        newMessage01_Lambda = Lambda_factor[var0_dofs:, var0_dofs:] - np.dot(
            np.dot(Lambda_factor[var0_dofs:, 0:var0_dofs], np.linalg.inv(Lambda_factor[0:var0_dofs, 0:var0_dofs] + \
                                                                       self.Belief0.Lambda - self.Message10.Lambda)),
            Lambda_factor[0:var0_dofs, var0_dofs:])

        newMessage10_eta = eta_factor[0:var0_dofs] - np.dot(
            np.dot(Lambda_factor[0:var0_dofs, var0_dofs:], np.linalg.inv(Lambda_factor[var0_dofs:, var0_dofs:] + \
                                                                       self.Belief1.Lambda - self.Message01.Lambda)),
            (eta_factor[var0_dofs:] + self.Belief1.eta - self.Message01.eta).T).T
        newMessage10_Lambda = Lambda_factor[0:var0_dofs, 0:var0_dofs] - np.dot(
            np.dot(Lambda_factor[0:var0_dofs, var0_dofs:], np.linalg.inv(Lambda_factor[var0_dofs:, var0_dofs:] + \
                                                                       self.Belief1.Lambda - self.Message01.Lambda)),
            Lambda_factor[var0_dofs:, 0:var0_dofs])

        # Update messages using damping
        self.Message01.eta = (1 - eta_damping) * newMessage01_eta + eta_damping * self.Message01.eta
        self.Message01.Lambda = (1 - Lambda_damping) * newMessage01_Lambda + Lambda_damping * self.Message01.Lambda
        self.Message10.eta = (1 - eta_damping) * newMessage10_eta + eta_damping * self.Message10.eta
        self.Message10.Lambda = (1 - Lambda_damping) * newMessage10_Lambda + Lambda_damping * self.Message10.Lambda


""" Create functions to update beliefs and send messages"""
def all_beliefs(graph):
    for varn in graph.var_nodes:
        varn.updateBelief()
    return graph


def all_mess(graph):
    for edg in graph.edges:
        edg.computeMessages(g.eta_damping, g.Lambda_damping)
    return  graph


def sync_upt(graph):
    graph = all_mess(graph)
    graph = all_beliefs(graph)
    return graph


""" Printing functions"""
def pmess(g):
    m_eta, m_lambda = [], []
    for vn in g.var_nodes:
        for edge in vn.edges:
            if edge.var_node0 == vn:
                m_eta += list(edge.Message10.eta)
                m_lambda += list(edge.Message10.Lambda.flatten())
            elif edge.var_node1 == vn:
                m_eta += list(edge.Message01.eta)
                m_lambda += list(edge.Message01.Lambda.flatten())

    print('Messages')
    print(m_eta)
    print(m_lambda)

def pbeliefs(g):
    b_eta, b_lambda = [], []
    for vn in g.var_nodes:
        b_eta += list(vn.belief.eta)
        b_lambda += list(vn.belief.Lambda.flatten())
    print('Beliefs')
    print(b_eta)
    print(b_lambda, '\n')


""" Create graph """
g = Graph()
g.n_nodes = n_varnodes
g.n_edges = n_edges

for nodeID in range(n_varnodes):
    vn = VariableNode(nodeID, var_dofs)
    vn.prior.eta = np.array(priors_eta[nodeID * var_dofs: (nodeID + 1) * var_dofs])
    vn.prior.Lambda = np.array(np.reshape(priors_lambda[nodeID * var_dofs**2: (nodeID + 1) * var_dofs**2], [var_dofs, var_dofs]))
    g.var_nodes.append(vn)

for edgeID in range(n_edges):
    f_lamba = np.block([[np.reshape(factor_potentials_lambda[edgeID*(2*var_dofs)**2: edgeID*(2*var_dofs)**2 + var_dofs**2], [var_dofs,var_dofs]),
                         np.reshape(factor_potentials_lambda[edgeID*(2*var_dofs)**2 + var_dofs**2: edgeID*(2*var_dofs)**2 + 2*var_dofs**2], [var_dofs,var_dofs])],
                        [np.reshape(factor_potentials_lambda[edgeID*(2*var_dofs)**2 + 2*var_dofs**2: edgeID*(2*var_dofs)**2 + 3*var_dofs**2], [var_dofs,var_dofs]),
                         np.reshape(factor_potentials_lambda[edgeID*(2*var_dofs)**2 + 3*var_dofs**2: (edgeID+1) * (2*var_dofs)**2], [var_dofs,var_dofs])]])
    ed = Edge(edgeID, g.var_nodes[measurements_nodeIDs[edgeID*2]],
              g.var_nodes[measurements_nodeIDs[edgeID*2 + 1]],
              np.array(factor_potentials_eta[edgeID * 2*var_dofs: (edgeID + 1) * 2*var_dofs]), f_lamba)
    g.edges.append(ed)

    # Add edges to variable nodes
    g.var_nodes[measurements_nodeIDs[edgeID * 2]].edges.append(ed)
    g.var_nodes[measurements_nodeIDs[edgeID * 2 + 1]].edges.append(ed)


""" Full MAP solution """
beta = np.zeros(var_dofs * n_varnodes)
blambda = np.zeros([var_dofs * n_varnodes, var_dofs * n_varnodes])

for id, vn in enumerate(g.var_nodes):
    beta[id * var_dofs: (id + 1) * var_dofs] += vn.prior.eta
    blambda[id * var_dofs: (id + 1) * var_dofs, id * var_dofs: (id + 1) * var_dofs] += vn.prior.Lambda

for id, ed in enumerate(g.edges):
    beta[ed.var0ID * var_dofs: (ed.var0ID + 1) * var_dofs] += ed.potential.eta[:var_dofs]
    beta[ed.var1ID * var_dofs: (ed.var1ID + 1) * var_dofs] += ed.potential.eta[var_dofs:]

    blambda[ed.var0ID * var_dofs: (ed.var0ID + 1) * var_dofs, ed.var0ID * var_dofs: (ed.var0ID + 1) * var_dofs] += ed.potential.Lambda[:var_dofs, :var_dofs]
    blambda[ed.var1ID * var_dofs: (ed.var1ID + 1) * var_dofs, ed.var1ID * var_dofs: (ed.var1ID + 1) * var_dofs] += ed.potential.Lambda[var_dofs:, var_dofs:]
    blambda[ed.var0ID * var_dofs: (ed.var0ID + 1) * var_dofs, ed.var1ID * var_dofs: (ed.var1ID + 1) * var_dofs] += ed.potential.Lambda[:var_dofs, var_dofs:]
    blambda[ed.var1ID * var_dofs: (ed.var1ID + 1) * var_dofs, ed.var0ID * var_dofs: (ed.var0ID + 1) * var_dofs] += ed.potential.Lambda[var_dofs:, :var_dofs]

bsigma = np.linalg.inv(blambda)
bmu = np.dot(bsigma, beta)

def getmus(graph):
    b_mu = []
    for vn in g.var_nodes:
        b_mu += list(vn.mu)
    return b_mu


""" Do message passing """
# beg = time.time()
# g = all_beliefs(g)
# fin = time.time()
# print('\n', (fin - beg) / len(g.var_nodes))

# pbeliefs(g)

print(g.n_nodes + g.n_edges)

tot = 0
niters = 100
dists = []
for i in range(niters):
    g = sync_upt(g)
    pbeliefs(g)

    # Check we are tending to MAP solution
    b_mus = getmus(g)
    dist = np.linalg.norm(b_mus - bmu)
    print(i, dist)
    dists.append(dist)

    if i %2 == 0:
        plt.figure()
        plt.plot(dists)
        plt.ylabel('abs distance from MAP solution')
        plt.xlabel('niters')
        plt.show()
