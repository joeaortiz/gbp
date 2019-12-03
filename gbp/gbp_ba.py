import numpy as np

from gbp.gbp import Graph, VariableNode, Edge

class baGraph(Graph):
    def __init__(self):
        PairwiseGraph.__init__(self)

        self.cam_var_nodes = []
        self.landmark_var_nodes = []
        self.var_nodes = self.cam_var_nodes + self.landmark_var_nodes

class LandmarkVariableNode(VariableNode):
    def __init__(self, variableID, dofs, pointID=None):
        VariableNode.__init__(self, variableID, dofs)
        self.pointID = pointID

class FrameVariableNode(VariableNode):
    def __init__(self, variableID, dofs, timestamp=None):
        VariableNode.__init__(self, variableID, dofs)
        self.timestamp = timestamp


class baEdge(Edge):
    def __init__(self, edgeID, var_node0, var_node1, z, K, Sigma_measurement):
        # Node 0 is camera, node 1 is landmark

        self.edgeID = edgeID

        self.var_node0 = var_node0
        self.var_node1 = var_node1
        self.var0ID = var_node0.variableID
        self.var1ID = var_node1.variableID
        self.n_vars0 = var_node0.dofs
        self.n_vars1 = var_node1.dofs

        self.potential = ndimGaussian(self.n_vars0 + self.n_vars1)
        self.linpoint = np.zeros(self.n_vars0 + self.n_vars1)

        self.Belief0 = ndimGaussian(self.n_vars0)
        self.Belief1 = ndimGaussian(self.n_vars1)

        self.Message01 = ndimGaussian(self.n_vars1)
        self.Message10 = ndimGaussian(self.n_vars0)

        # Related to measurement
        self.z = z
        self.K = K
        self.Sigma_measurement = Sigma_measurement

        self.eta_damping = 0.0
        self.num_undamped_iters = 13
        self.dampingcount = -self.num_undamped_iters
        self.max_etadamping = 0.4
        self.minlinearised_iters = 20

        self.reproj = 0.0
        self.mu = np.zeros(9)
        self.dmu_threshold = 1e-5

        self.loss = 'huber'
        self.Nstds = 3
        self.robust_flag = False
        self.Sigma_meas = -1
        self.err = -1

        self.inf = []
        self.inf1 =[]

    def reprojection_err(self):
        """ Calculate the reprojection error. """
        camera_state  = np.array(np.dot(np.linalg.inv(self.Belief0.Lambda), self.Belief0.eta.T).T)[0]
        lmark_state  = np.array(np.dot(np.linalg.inv(self.Belief1.Lambda), self.Belief1.eta.T).T)[0]
        d = hfunc(camera_state, lmark_state, self.K) - self.z
        return np.linalg.norm(d)

    def robustify_loss(self, pred_coords):
        """ Return Sigma_measurement which has been robustified if necessary using the chosen cost function."""
        if self.loss is None:
            Sigma_meas = self.Sigma_measurement

        elif self.loss == 'huber':  # Loss is linear after Nsts from expected value
            err = np.linalg.norm(self.z - pred_coords)
            self.err = err
            if err > self.Nstds * np.sqrt(self.Sigma_measurement):

                Sigma_meas = self.Sigma_measurement * err**2 / (2*(self.Nstds * np.sqrt(self.Sigma_measurement) * err - 0.5 * self.Nstds**2 * self.Sigma_measurement))
                # lmarkID = self.edges[0].var_node.variableID
                # camID = self.edges[1].var_node.variableID
                # print(f'camID {camID}, lmarkID {lmarkID}, num stds from expected: {np.linalg.norm(err) / np.sqrt(self.Sigma_measurement)}, new Sigma: {Sigma_meas}')
                self.robust_flag = True
            else:
                self.robust_flag = False
                Sigma_meas = self.Sigma_measurement

        elif self.loss == 'constant':  # Loss is constant after Nstds from expected value
            err = np.linalg.norm(self.z - pred_coords)
            if err > self.Nstds * np.sqrt(self.Sigma_measurement):
                Sigma_meas = (err / self.Nstds)**2
                self.robust_flag = True
            else:
                self.robust_flag = False
                Sigma_meas = self.Sigma_measurement

        return Sigma_meas

    def reprojection_cost(self):
        """ Calculate the reprojection error. """
        camera_state  = np.array(np.dot(np.linalg.inv(self.Belief0.Lambda), self.Belief0.eta.T).T)[0]
        lmark_state  = np.array(np.dot(np.linalg.inv(self.Belief1.Lambda), self.Belief1.eta.T).T)[0]
        d = hfunc(camera_state, lmark_state, self.K) - self.z

        return 0.5 * np.linalg.norm(d)**2 / self.Sigma_measurement

    def computePotential(self, camera_state=None, lmark_state=None):
        np.set_printoptions(precision=13)
        # print(self.Belief1.eta.T)

        if camera_state is None and lmark_state is None:
            # Linearisation point
            camera_state  = np.array(np.dot(np.linalg.inv(self.Belief0.Lambda), self.Belief0.eta.T).T)[0]
            lmark_state  = np.array(np.dot(np.linalg.inv(self.Belief1.Lambda), self.Belief1.eta.T).T)[0]
            # print('cam lin point', camera_state)
            # print('lmk lin point', lmark_state)

        cam_dofs = self.n_vars0

        J = Jac(camera_state, lmark_state, self.K[0,0], self.K[1,1])
        Jf = Jfd(camera_state, lmark_state, self.K, 1e-6)
        # print('\n Jac\n', J)

        # if np.max(abs(J - Jf)) > 0.01:
        #     print(self.edgeID)
        #     print(np.max(abs(J - Jf).flatten()))
        #     print(np.argmax(abs(J-Jf).flatten()))
            # print(camera_state, lmark_state)
            # print(f'z point in cf {np.dot(tranf_w2c(camera_state), np.concatenate((lmark_state, [1])) )[2]}')
            # print(J)
            # print(Jf)
            # print(J - Jf)

        pred_coords = hfunc(camera_state, lmark_state, self.K)
        self.Sigma_meas = self.robustify_loss(pred_coords)
        self.linpoint = np.concatenate((camera_state, lmark_state))
        Lambda_measurement = np.eye(2) / self.Sigma_meas #self.Sigma_measurement
        Lambda_factor = np.matrix(np.dot(np.dot(J.T, Lambda_measurement), J))
        eta_factor = np.matrix(np.dot(np.dot(J.T, Lambda_measurement), (np.dot(J, self.linpoint) + self.z - pred_coords)))

        # print('\n pre dividing ', np.dot(J.T, (np.dot(J, self.linpoint) + self.z - pred_coords)) )
        # print('\neta buffer', np.dot(J, self.linpoint) + self.z - pred_coords)
        # print('\npred_coords', pred_coords)

        return eta_factor, Lambda_factor

    def computeMessages(self, local_relin=True):

        cam_dofs = self.n_vars0

        if local_relin:
            newreproj = self.reprojection_err()
            delta_reproj = abs(newreproj - self.reproj)

            camera_state  = np.array(np.dot(np.linalg.inv(self.Belief0.Lambda), self.Belief0.eta.T).T)[0]
            lmark_state  = np.array(np.dot(np.linalg.inv(self.Belief1.Lambda), self.Belief1.eta.T).T)[0]
            newmu = np.concatenate((camera_state, lmark_state))
            delta_mu = np.linalg.norm(newmu - self.mu)

            self.reproj = newreproj
            self.mu = newmu

            # Condition to update potential
            if delta_mu < self.dmu_threshold and self.dampingcount > self.minlinearised_iters - self.num_undamped_iters:
                # print(f'Updating potential at edge {self.edgeID}')
                self.potential.eta, self.potential.Lambda = self.computePotential()
                self.eta_damping = 0.0
                self.dampingcount = -self.num_undamped_iters
            else:
                self.dampingcount += 1

            if self.dampingcount == 0:
                # print(f'changing edge damping factor from {self.eta_damping} to 0.4')
                self.eta_damping = self.max_etadamping
        else:
            delta_mu, delta_reproj = [], []

        eta_factor, Lambda_factor = self.potential.eta, self.potential.Lambda
        # print('\n eta factor \n', eta_factor)
        # print('\n Lambda factor \n', Lambda_factor)

        newMessage01_eta = eta_factor[0, cam_dofs:] - np.dot(np.dot(Lambda_factor[cam_dofs:, 0:cam_dofs], np.linalg.inv(Lambda_factor[0:cam_dofs, 0:cam_dofs] + \
                        self.Belief0.Lambda - self.Message10.Lambda) ) , (eta_factor[0, 0:cam_dofs] + self.Belief0.eta - self.Message10.eta).T).T
        newMessage01_Lambda = Lambda_factor[cam_dofs:, cam_dofs:] - np.dot(np.dot(Lambda_factor[cam_dofs:, 0:cam_dofs] , np.linalg.inv(Lambda_factor[0:cam_dofs, 0:cam_dofs] + \
                        self.Belief0.Lambda - self.Message10.Lambda) ) , Lambda_factor[0:cam_dofs, cam_dofs:])

        newMessage10_eta = eta_factor[0, 0:cam_dofs] - np.dot(np.dot(Lambda_factor[0:cam_dofs, cam_dofs:] , np.linalg.inv(Lambda_factor[cam_dofs:, cam_dofs:] + \
                        self.Belief1.Lambda - self.Message01.Lambda) ) , (eta_factor[0, cam_dofs:] + self.Belief1.eta - self.Message01.eta).T).T
        newMessage10_Lambda = Lambda_factor[0:cam_dofs, 0:cam_dofs] - np.dot(np.dot(Lambda_factor[0:cam_dofs, cam_dofs:] , np.linalg.inv(Lambda_factor[cam_dofs:, cam_dofs:] + \
                        self.Belief1.Lambda - self.Message01.Lambda) ) , Lambda_factor[cam_dofs:, 0:cam_dofs])

        # print(Lambda_factor[cam_dofs:, cam_dofs:] + self.Belief1.Lambda - self.Message01.Lambda)
        # Compute message using cov form
        # feta = np.copy(eta_factor)
        # flam = np.copy(Lambda_factor)
        # feta[:, 6:] += self.Belief1.eta - self.Message01.eta
        # flam[6:,6:] += self.Belief1.Lambda - self.Message01.Lambda


        # print(np.linalg.det(fcond), np.linalg.cond(fcond))
        # print('\n factor cov: \n', np.linalg.inv(flam))
        # print('\n factor mu: \n', np.dot(np.linalg.inv(flam), feta.T))


        # feta1 = np.copy(eta_factor)
        # flam1 = np.copy(Lambda_factor)
        # print(feta1)
        # # Add message from lmk node
        # feta1[:, :6] += self.Belief0.eta - self.Message10.eta
        # flam1[:6,:6] += self.Belief0.Lambda - self.Message10.Lambda
        # print('\n factor lambda to marginalise: \n', flam1)
        # print(np.linalg.det(flam1), np.linalg.cond(flam1))
        # print('\n factor cov: \n', np.linalg.inv(flam1))
        # print('\n factor mu: \n', np.dot(np.linalg.inv(flam1), feta1.T))


        # print('\n factor eta', eta_factor[0, 0:cam_dofs])
        # print('\n lambda factor ll: \n', Lambda_factor[0:cam_dofs, 0:cam_dofs]  )
        # print('\n lambda to invert is: \n', Lambda_factor[cam_dofs:, cam_dofs:] + self.Belief1.Lambda - self.Message01.Lambda)
        # print('\n lambda inverted is: \n', np.linalg.inv(Lambda_factor[cam_dofs:, cam_dofs:] + self.Belief1.Lambda - self.Message01.Lambda))
        # print('\n lambda product: \n', np.dot(Lambda_factor[0:cam_dofs, cam_dofs:] , np.linalg.inv(Lambda_factor[cam_dofs:, cam_dofs:] + self.Belief1.Lambda - self.Message01.Lambda) ))
        # print('\n eta not outedge factor : \n', eta_factor[0, cam_dofs:])
        # print('\n eta belief : \n', self.Belief1.eta)
        # print('\n eta terms summed: \n', (eta_factor[0, cam_dofs:] + self.Belief1.eta - self.Message01.eta))
        # print('\n eta to be added: \n', np.dot(np.dot(Lambda_factor[0:cam_dofs, cam_dofs:] , np.linalg.inv(Lambda_factor[cam_dofs:, cam_dofs:] + \
        #                 self.Belief1.Lambda - self.Message01.Lambda) ) , (eta_factor[0, cam_dofs:] + self.Belief1.eta - self.Message01.eta).T).T)
        # print('\n factor out edge eta: \n', eta_factor[0, 0:cam_dofs])
        ratio = []
        ratio10e, ratio10l, ratio01e, ratio01l = [], [], [], []
        mag_terms10e, mag_terms10l, mag_terms01e, mag_terms01l = [], [], [], []
        for i in range(6):
            mag_terms = eta_factor[0, i]
            mag_diff = newMessage10_eta[0,i]
            mag_terms10e.append(mag_diff)
            ratio10e.append(mag_diff / mag_terms)
        for i in range(6):
            for j in range(6):
                mag_terms = Lambda_factor[i, j]
                mag_diff = newMessage10_Lambda[i,j]
                mag_terms10l.append(mag_diff)
                ratio10l.append(mag_diff / mag_terms)
        for i in range(3):
            mag_terms = eta_factor[0, cam_dofs + i]
            mag_diff = newMessage01_eta[0,i]
            mag_terms01e.append(mag_diff)
            ratio01e.append(mag_diff / mag_terms)
        for i in range(3):
            for j in range(3):
                mag_terms = Lambda_factor[cam_dofs+i, cam_dofs+i]
                mag_diff = newMessage01_Lambda[i,j]
                mag_terms01l.append(mag_diff)
                ratio01l.append(mag_diff / mag_terms)
        ratio = [ratio10e, ratio01e, ratio10l, ratio01l]
        mag = [mag_terms10e, mag_terms01e, mag_terms10l, mag_terms01l]


        # print('\n lambda to invert is positive semi definite? ', is_pos_semi_def(Lambda_factor[0:cam_dofs, 0:cam_dofs] + self.Belief0.Lambda - self.Message10.Lambda)  )
        # print('\n lambda to invert is positive definite? ', is_pos_def(Lambda_factor[0:cam_dofs, 0:cam_dofs] + self.Belief0.Lambda - self.Message10.Lambda)  )
        # print('\n lambda to invert has rank ', np.linalg.matrix_rank(Lambda_factor[0:cam_dofs, 0:cam_dofs] + self.Belief0.Lambda - self.Message10.Lambda) )
        # print('\n lambda to invert has det ', np.linalg.det(Lambda_factor[0:cam_dofs, 0:cam_dofs] + self.Belief0.Lambda - self.Message10.Lambda) )
        # print('\n lambda to invert has condition number ', np.linalg.cond(Lambda_factor[0:cam_dofs, 0:cam_dofs] + self.Belief0.Lambda - self.Message10.Lambda) )
        # print('\n lambda to invert has eigenvalues ', np.linalg.eig(Lambda_factor[0:cam_dofs, 0:cam_dofs] + self.Belief0.Lambda - self.Message10.Lambda)[0] )
        # print(Lambda_factor[0:cam_dofs, 0:cam_dofs] + self.Belief0.Lambda - self.Message10.Lambda)
        # print(np.linalg.inv(Lambda_factor[0:cam_dofs, 0:cam_dofs] + self.Belief0.Lambda - self.Message10.Lambda))
        # print('\n lambda inverted', np.linalg.inv(Lambda_factor[cam_dofs:, cam_dofs:] + self.Belief1.Lambda - self.Message01.Lambda) )
        # print('\n lambda product', np.dot(Lambda_factor[0:cam_dofs, cam_dofs:] , np.linalg.inv(Lambda_factor[cam_dofs:, cam_dofs:] + self.Belief1.Lambda - self.Message01.Lambda) ))

        # print('\n\npre damping messages')
        # print(newMessage10_eta)
        # print(newMessage01_eta)
        # print(newMessage10_Lambda)
        # print(newMessage01_Lambda)

        old_lambda01 = self.Message01.Lambda.copy()

        # Update messages using damping
        eta_damping = self.eta_damping
        Lambda_damping = 0.
        self.Message01.eta = (1 - eta_damping) * newMessage01_eta + eta_damping * self.Message01.eta
        self.Message01.Lambda = (1 - Lambda_damping) * newMessage01_Lambda + Lambda_damping * self.Message01.Lambda
        self.Message10.eta = (1 - eta_damping) * newMessage10_eta + eta_damping * self.Message10.eta
        self.Message10.Lambda = (1 - Lambda_damping) * newMessage10_Lambda + Lambda_damping * self.Message10.Lambda


        if self.edgeID in [0, 1,2,3,4,5,6,7,8,9]:
            print(np.linalg.eig(self.Message01.Lambda)[0])
            print(np.linalg.eig(self.Message10.Lambda)[0])
            # self.inf.append(np.linalg.norm(self.Message01.Lambda))
            # self.inf1.append(np.linalg.norm(self.Message01.Lambda) - np.linalg.norm(old_lambda01))

        # print('post damping ', self.Message10.eta)

        return delta_mu, delta_reproj, ratio, mag