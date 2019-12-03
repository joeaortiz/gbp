import numpy as np
import scipy
import time
# from src.utils import * 
from src.n2nutils import *


def synchronous_update(graph, local_relin=True):
    # print('\nSychronous message update')

    delta_mus, delta_reprojs = [], []
    # Factor nodes send outgoing messages along all of its edges
    for edge in graph.meas_edges:
        delta_mu, delta_reproj, ratio, mag = edge.computeMessages(local_relin=local_relin)
        delta_mus.append(delta_mu)
        delta_reprojs.append(delta_reproj)

    # Camera variable nodes send outgoing messages along all its edges
    for var_node in graph.cam_var_nodes + graph.landmark_var_nodes:
        var_node.updateBelief()

    return graph, delta_mus, delta_reprojs, ratio, mag


def convergeVariance(graph, update_tol=1e-3, maxniters=1000, err_print=False):
    # Synchronous message updates until variances have converged
    bigmu, bigSigma = solveMarginals(graph)
    marg_vars = []
    for c in range(len(graph.cam_var_nodes)):
        marg_vars += list(np.linalg.inv(bigSigma[c*6: (c+1)*6, c*6: (c+1)*6]).flatten())
    for l in range(len(graph.landmark_var_nodes)):
        marg_vars +=  list(np.linalg.inv(bigSigma[len(graph.cam_var_nodes) + l*3: len(graph.cam_var_nodes) + (l+1)*3, len(graph.cam_var_nodes) + l*3: len(graph.cam_var_nodes) + (l+1)*3]).flatten())


    converged = 0
    av_dist = []
    av_absdist = []
    Lambda_max_updates = []
    Lambda_dist = []
    for i in range(maxniters):

        Lambdas = []
        for var_node in graph.cam_var_nodes + graph.landmark_var_nodes:
            Lambdas += list(np.array(var_node.belief.Lambda)[0].flatten())

        graph = synchronous_update(graph)
        if err_print:
            print(f'\n{i} Reprojection Error: ', nn2_av_error(graph))

        newLambdas = []
        for var_node in graph.cam_var_nodes + graph.landmark_var_nodes:
            newLambdas += list(np.array(var_node.belief.Lambda)[0].flatten())

        mu = np.array([0])
        for var_node in graph.cam_var_nodes + graph.landmark_var_nodes:
            mu = np.concatenate((mu, np.array(var_node.mu)[0]))
        av_disti = np.mean(mu[1:] - bigmu)
        av_dist.append(av_disti)
        av_absdist.append(np.mean(abs(mu[1:] - bigmu)))
        # print(f'av dist {np.mean(mu[1:] - bigmu)}')

        max_update = np.max(abs(np.array(newLambdas) - np.array(Lambdas)))
        Ldist = np.mean(np.array(newLambdas) - np.array(Lambdas))
        Lambda_max_updates.append(max_update)
        Lambda_dist.append(Ldist)
        # print('Lambda max update:', max_update)
        # print('Lambda distance from marginal lambda:', Ldist)
        print(f'{i} -- av dist: {av_disti:.6f}, lambda update: {max_update:.6f}, lambda dist {Ldist:.5f}', end='\r')

        # Make sure that variances have converged and aren't at turning point. 
        if max_update < update_tol:
            converged += 1
            if converged == 10:
                print(f'\n Variances have converged to within {update_tol}!\n')
                break
        else:
            converged = 0

    if converged == 10:
        converged = True
    else:
        converged = False
        
    return graph, i, converged, av_dist, av_absdist, Lambda_dist, Lambda_max_updates


def reproj_cost(graph):
    """
    Returns the average reprojection error of an observation of a landmark in the image plane.
    The units are pixels.  
    """
    err = 0
    for edge in graph.meas_edges:
        err += 0.5 *  edge.reprojection_err() ** 2
    return err

def nn2_totnorm_error(graph):
    """
    Returns the average reprojection error of an observation of a landmark in the image plane.
    The units are pixels.  
    """
    err = 0
    for edge in graph.meas_edges:
        err += edge.reprojection_err() 
    return err


def prior_cost(graph):
    prior_cost = 0.0
    for n in graph.cam_var_nodes + graph.landmark_var_nodes:
        prior_cost += n.prior_cost()
    return prior_cost


# ------------------------------- Solve full system --------------------------------

def n2n_construct_Lambda(graph, dl=False):
    """
    Construct the big Lambda matrix for all variables from the factor nodes
    """
    n_cameras = len(graph.cam_var_nodes)
    n_landmarks = len(graph.landmark_var_nodes)

    cam_dofs = graph.cam_var_nodes[0].dofs

    n_variables = n_landmarks * 3 + n_cameras * cam_dofs

    Lambda = np.zeros([n_variables, n_variables])
    eta = np.zeros(n_variables)

    # Diagonal blocks of Lambda and eta
    # Depth priors arent used if landmark priors are used. 
    cam_timestamps = []
    for i, cam in enumerate(graph.cam_var_nodes):
        cam_timestamps.append(cam.timestamp)
        if dl:
            eta[i*cam_dofs:(i+1)*cam_dofs] = cam.priorpert.eta
            Lambda[ i*cam_dofs:(i+1)*cam_dofs, i*cam_dofs: (i+1)*cam_dofs] = cam.priorpert.Lambda            
        else:
            eta[i*cam_dofs:(i+1)*cam_dofs] = cam.prior.eta
            Lambda[ i*cam_dofs:(i+1)*cam_dofs, i*cam_dofs: (i+1)*cam_dofs] = cam.prior.Lambda

    landmark_pointIDs = []
    for i, lmark in enumerate(graph.landmark_var_nodes):
        landmark_pointIDs.append(lmark.pointID)
        if dl:
            eta[cam_dofs*n_cameras + i*3:cam_dofs*n_cameras + (i+1)*3] = lmark.priorpert.eta
            Lambda[cam_dofs*n_cameras + i*3:cam_dofs*n_cameras + (i+1)*3, cam_dofs*n_cameras + i*3:cam_dofs*n_cameras + (i+1)*3] = lmark.priorpert.Lambda    
        else:
            eta[cam_dofs*n_cameras + i*3:cam_dofs*n_cameras + (i+1)*3] = lmark.prior.eta
            Lambda[cam_dofs*n_cameras + i*3:cam_dofs*n_cameras + (i+1)*3, cam_dofs*n_cameras + i*3:cam_dofs*n_cameras + (i+1)*3] = lmark.prior.Lambda


    # Off diagonal blocks of Lambda. These are taken given current linearisation
    for edge in graph.meas_edges:
        factor_eta, factor_Lambda = edge.potential.eta.copy(), edge.potential.Lambda.copy()

        timestamp = edge.var_node0.timestamp
        pointID = edge.var_node1.pointID

        lmark_ix = landmark_pointIDs.index(pointID)
        cam_ix = cam_timestamps.index(timestamp)

        eta[cam_ix*cam_dofs : (cam_ix + 1)*cam_dofs] += np.array(factor_eta[0, 0:cam_dofs])[0]
        eta[cam_dofs*n_cameras + lmark_ix*3 : cam_dofs*n_cameras + (lmark_ix + 1)*3] += np.array(factor_eta[0, cam_dofs:])[0]

        Lambda[cam_ix*cam_dofs : (cam_ix + 1)*cam_dofs , cam_ix*cam_dofs : (cam_ix + 1)*cam_dofs] += factor_Lambda[:cam_dofs, :cam_dofs]
        Lambda[cam_dofs*n_cameras + lmark_ix*3 : cam_dofs*n_cameras + (lmark_ix + 1)*3, cam_dofs*n_cameras + lmark_ix*3 : cam_dofs*n_cameras + (lmark_ix + 1)*3] += factor_Lambda[cam_dofs:, cam_dofs:]
        Lambda[cam_ix*cam_dofs : (cam_ix + 1)*cam_dofs , cam_dofs*n_cameras + lmark_ix*3 : cam_dofs*n_cameras + (lmark_ix + 1)*3] += factor_Lambda[:cam_dofs, cam_dofs:]
        Lambda[cam_dofs*n_cameras + lmark_ix*3 : cam_dofs*n_cameras + (lmark_ix + 1)*3, cam_ix*cam_dofs : (cam_ix + 1)*cam_dofs] += factor_Lambda[cam_dofs:, :cam_dofs]

    return eta, Lambda

def solveMarginals(graph, dl=False):

    bigeta, bigLambda = n2n_construct_Lambda(graph, dl=dl)
    bigSigma = np.linalg.inv(bigLambda)
    bigmu = np.dot(bigSigma, bigeta)

    return bigmu, bigSigma

def totsqr_reprojection_err_from_bigmu(graph, bigmu):
    """ Compute the reprojection error from the mu vector for all variable nodes rather than the beliefs stored in the graph. 
        Also need to pass in graph so connectivity is known. """
    cam_dofs = graph.cam_var_nodes[0].dofs
    n_cameras = len(graph.cam_var_nodes)

    err = 0
    for edge in graph.meas_edges:
        camera_state  = bigmu[edge.var0ID * cam_dofs: (edge.var0ID + 1) * cam_dofs]
        lmark_state  = bigmu[n_cameras*cam_dofs + (edge.var1ID - n_cameras) * 3: n_cameras*cam_dofs + (edge.var1ID - n_cameras + 1) * 3]

        d = hfunc(camera_state, lmark_state, edge.K) - edge.z

        err += 0.5 * np.linalg.norm(d)**2
    # av_err = err / graph.n_edges
    return err

# ---------------------------------- Convergence checks ---------------------------------------------

def compose_v(graph):
    # Stack messages from variable nodes to factor nodes

    v = np.zeros(1)
    for var_node in graph.cam_var_nodes:
        for edge in var_node.edges:
            eta_message_var2factor = np.array(edge.Belief0.eta - edge.Message10.eta)[0]
            v = np.concatenate((v, eta_message_var2factor))
    for var_node in graph.landmark_var_nodes:
        for edge in var_node.edges:
            eta_message_var2factor = np.array(edge.Belief1.eta - edge.Message01.eta)[0]
            v = np.concatenate((v, eta_message_var2factor))

    return v[1:]

def compose_b_i2fk(graph, i, k):
    # Compose the b vector for the message from node i to factor node with index k at node i.

    var_nodes = graph.cam_var_nodes + graph.landmark_var_nodes
    var_node = var_nodes[i]

    b_i2fk = np.array(var_node.prior.eta)[0].copy()

    for j, edge in enumerate(var_node.edges):
        if j != k:

            if i < len(graph.cam_var_nodes): # Node i is a camera variable node.
                b_i2fk += np.array(edge.potential.eta[0, 0:var_node.dofs] - np.dot(np.dot(edge.potential.Lambda[0:var_node.dofs, var_node.dofs:], \
                            np.linalg.inv(edge.potential.Lambda[var_node.dofs:, var_node.dofs:] + edge.Belief1.Lambda - edge.Message01.Lambda) ), edge.potential.eta[0, var_node.dofs:].T).T)[0]

            else:  # Node i is a landmrk variable node.
                b_i2fk += np.array(edge.potential.eta[0, -var_node.dofs:] - np.dot(np.dot(edge.potential.Lambda[-var_node.dofs:, :-var_node.dofs], \
                            np.linalg.inv(edge.potential.Lambda[:-var_node.dofs, :-var_node.dofs] + edge.Belief0.Lambda - edge.Message10.Lambda) ), edge.potential.eta[0, :-var_node.dofs].T).T)[0]

    return b_i2fk

def compose_b(graph):

    for i, var_node in enumerate(graph.cam_var_nodes + graph.landmark_var_nodes):
        for k in range(len(var_node.edges)):
            if i==0 and k==0:
                b = compose_b_i2fk(graph, i, k)
            else:
                b = np.concatenate((b, compose_b_i2fk(graph, i, k)))

    return b


def compose_Q_i2fk(graph, i, k):
    # Block row matrix

    var_nodes = graph.cam_var_nodes + graph.landmark_var_nodes
    var_nodei = var_nodes[i]
    edgek = var_nodei.edges[k]

    Q_i2fk = np.zeros([var_nodei.dofs, 1])

    for j, nodej in enumerate(graph.cam_var_nodes + graph.landmark_var_nodes):
        for k, edge in enumerate(nodej.edges):

            # Block is non zero if node on other side of edge is var_nodei
            if j != i: 
                if edge.var_node0 == var_nodei and edge != edgek:
                    Qblock = np.dot(edge.potential.Lambda[0:var_nodei.dofs, -nodej.dofs:] , np.linalg.inv(edge.potential.Lambda[-nodej.dofs:, -nodej.dofs:] + \
                                edge.Belief1.Lambda - edge.Message01.Lambda))

                elif edge.var_node1 == var_nodei and edge != edgek:
                    Qblock = np.dot(edge.potential.Lambda[-var_nodei.dofs:, 0:nodej.dofs] , np.linalg.inv(edge.potential.Lambda[0:nodej.dofs, 0:nodej.dofs] + \
                                edge.Belief0.Lambda - edge.Message10.Lambda))

                else:
                    Qblock = np.zeros([var_nodei.dofs, nodej.dofs])
            else:
                Qblock = np.zeros([var_nodei.dofs, nodej.dofs])
            
            Q_i2fk = np.hstack((Q_i2fk, Qblock))

    return Q_i2fk[:, 1:]


def compose_Q(graph):

    for i, var_node in enumerate(graph.cam_var_nodes + graph.landmark_var_nodes):
        for k in range(len(var_node.edges)):
            if i==0 and k==0:
                Q = compose_Q_i2fk(graph, i, k)
            else:
                Q = np.vstack((Q, compose_Q_i2fk(graph, i, k)))

    return Q


def check_lin_system(graph, tol=1e-4):

    v_pred = nextv(graph)

    # One synchronous update and then compose next v
    graph = synchronous_update(graph)
    for node in graph.cam_var_nodes + graph.landmark_var_nodes:
        node.updateBelief()

    v_next = compose_v(graph)
    print('\nMax difference in predicted next vectors', np.max(abs(v_pred - v_next)))
    if np.max(abs(v_pred - v_next)) < tol:
        return True
    else:
        return False

def nextv(graph):
    # Compute v, b, and Q and compute the v for the next step
    v = compose_v(graph)
    b = compose_b(graph)
    Q = compose_Q(graph)
    v_pred = (- np.dot(Q, v) + b) * (1- graph.eta_damping) + graph.eta_damping * v

    return v_pred


def find_message_damping(graph, verbose=True):

    Q = -compose_Q(graph)
    if verbose:
        # print('\n\nSpectral radius of Q', spectral_radius(Q))
        print(f'Are Q and b values correct to within tolerance? {check_lin_system(graph, tol=1e-4)} \n')

    rhos = []
    ds = np.linspace(0, 0.99, 100)
    eigenvalues = []
    for d in ds:

        Qdash = Q * (1 - d) + d * np.eye(len(Q))
        rho_Qdash = spectral_radius(Qdash)
        rhos.append(rho_Qdash)
        if verbose:
            print(f'Damping: {d:.3f}     Spectral radius {rho_Qdash:.7f}')

        eigs, w = np.linalg.eig(Qdash)
        eigs = np.sort(eigs)
        # print(eigs)
        eigenvalues.append(list(eigs.real) + list(eigs.imag))
        # for e, r, a in zip(eigs, np.real(eigs), abs(eigs)):
        #     print(r, a)

        # argm = np.argmax(abs(eigs))
        # print(argm)
        # print(eigs[argm])

    graph.eta_damping = ds[np.argmin(rhos)]
    print(f'\n Message damping: {graph.eta_damping}')

    return graph, rhos, eigenvalues


def compare_global_local_QQ(graph):
    # Compare QQ.T at each variable node computed locally with the matrix computed globally.

    Q_centralised = compose_Q(graph)

    row_start = 0
    for i, var_node in enumerate(graph.cam_var_nodes + graph.landmark_var_nodes):
        Qvarnode = var_node.compose_Qvarnode()
        QQ_local = np.dot(Qvarnode, Qvarnode.T)

        Qvarnode_central = Q_centralised[row_start:row_start + Qvarnode.shape[0]]
        QQ_central  = np.dot(Qvarnode_central, Qvarnode_central.T)

        row_start += Qvarnode.shape[0]
        print(f'Max absolute difference between QQ.T matrices at node {i}: ', np.max(abs(QQ_local - QQ_central)))


# --------------------------------- GMRF convergence condition --------------------------------

def check_walk_summability(bigLambda):
    D = np.diag(np.diag(bigLambda))
    normalisedbigLambda = np.dot(np.linalg.inv(D), bigLambda)

    R = np.eye(len(bigLambda)) - normalisedbigLambda
    Rabs = abs(R)

    w,v = scipy.linalg.eigh(Rabs)
    max_eigvalue = np.max(abs(w))

    # print('max eig', max_eigvalue)

    if max_eigvalue < 1:
        walk_summable = True
    else:
        walk_summable = False

    return is_pos_def(np.eye(len(bigLambda)) - Rabs), walk_summable

# ------------------------------ Diagonal loading -----------------------------------------------

def dlChooseGamma(graph):
    
    bigeta, bigLambda = n2n_construct_Lambda(graph)

    R = np.eye(len(bigLambda)) - bigLambda
    Rabs = abs(R)

    gammalim = spectral_radius_sym(Rabs) - 1

    graph.gamma = gammalim + 1.

    for var_node in (graph.cam_var_nodes + graph.landmark_var_nodes):
        var_node.priorpert.Lambda = var_node.prior.Lambda + graph.gamma * np.eye(len(var_node.prior.Lambda))

    return graph


def dlDoubleLoop(graph, maxniters=500, update_tol=1e-3):
    """ Inner loop of diagonal loading. 
        First update the eta matrix with the most recent estimate of the means. 
        Then solve the linear system using BP. 
    """

    # update eta vectors at each node before message passing. 
    for var_node in (graph.cam_var_nodes + graph.landmark_var_nodes):
        var_node.priorpert.eta = var_node.prior.eta + graph.gamma * var_node.mu

    bigmu, bigSigma = solveMarginals(graph, dl=True)
    converged_reprojerr = reprojection_err_from_bigmu(graph, bigmu)
    # print(f'\nConverged reprojection error should be: {converged_reprojerr}')


    converged = 0
    for i in range(maxniters):

        etas = np.array([0])
        mus = np.array([0])
        for var_node in graph.cam_var_nodes + graph.landmark_var_nodes:
            etas = np.concatenate((etas, np.array(var_node.belief.eta)[0]))
            mus = np.concatenate((mus, np.array(var_node.mu)[0]))


        graph = synchronous_update(graph, dl=True)
        # print(f'\n{i} Reprojection Error: ', nn2_av_error(graph))

        newetas = np.array([0])
        for var_node in graph.cam_var_nodes + graph.landmark_var_nodes:
            newetas = np.concatenate((newetas, np.array(var_node.belief.eta)[0]))

        # print('Eta max update:', np.max(abs(newetas - etas)))        
        # Make sure that variances have converged and aren't at turning point. 
        if np.max(abs(newetas - etas)) < update_tol:
            converged += 1
            if converged == 10:
                # print(f'\n Means have converged to within {update_tol}!\n')
                break
        else:
            converged = 0

    # print(bigmu - mus[1:])

    return graph, mus[1:]



def dlSingleLoop(graph, s=0.5):
    """ Inner loop of diagonal loading. 
        First update the eta matrix with the most recent estimate of the means. 
        Then solve the linear system using BP. 
    """

    # update eta vectors at each node before message passing. 
    for var_node in (graph.cam_var_nodes + graph.landmark_var_nodes):
        var_node.priorpert.eta = var_node.priorpert.eta * (1 - s) + s * (var_node.prior.eta + graph.gamma * var_node.mu)

    # bigmu, bigSigma = solveMarginals(graph, dl=True)
    # converged_reprojerr = reprojection_err_from_bigmu(graph, bigmu)
    # # print(f'\nConverged reprojection error should be: {converged_reprojerr}')


    graph = synchronous_update(graph, dl=True)
    # print(f'\n{i} Reprojection Error: ', nn2_av_error(graph))

    mus = np.array([0])
    for var_node in graph.cam_var_nodes + graph.landmark_var_nodes:
        mus = np.concatenate((mus, np.array(var_node.mu)[0]))

    return graph, mus[1:]


