import argparse
from gbp import gbp_ba
import vis

parser = argparse.ArgumentParser()
parser.add_argument("--file", help="BAL style file with BA data")
args = parser.parse_args()

file = args.file
print(file)
file = 'data/fr1desk.txt'

configs = dict({
    'gauss_noise_std': 2,
    'loss': None,  # Can be Huber or constant,
    'Nstds': 3,
    'beta': 0.01,
    'num_undamped_iters': 10,
    'min_linear_iters': 8,
    'eta_damping': 0.4,
    'prior_std_weaker_factor': 100,
    # For implementation with floats
    'final_prior_std_weaker_factor': 100,
    'num_weakening_steps': 5
           })


graph = gbp_ba.create_ba_graph(file, configs)

print(f'Number of keyframes: {len(graph.cam_nodes)}')
print(f'Number of landmarks: {len(graph.lmk_nodes)}')
print(f'Number of measurement factors: {len(graph.factors)}\n')

graph.generate_priors_var(weaker_factor=configs['prior_std_weaker_factor'])  # Sets prior factors automatically to be much weaker than measurement factors.
graph.update_all_beliefs()

vis.vis_scene.view_from_graph(graph)

# weakening_factor = np.log10(configs['final_prior_std_weaker_factor'] / configs['prior_std_weaker_factor']) / configs['num_weakening_steps']

graph.eta_damping = 0.0
for i in range(200):
    # # To copy weakening of strong priors as must be done on IPU with float
    # if (i+1) % 2 == 0 and (i < configs['num_weakening_steps'] * 2) and weakening_factor != 0.:
    #     graph.weaken_priors(weakening_factor)

    if i == 8 or i == 17:
        for factor in graph.factors:
            factor.iters_since_relin = 1

    are = graph.are()
    energy = graph.reprojection_energy()
    n_factor_relins = 0
    for factor in graph.factors:
        if factor.iters_since_relin == 0:
            n_factor_relins += 1
    print(f'Iteration {i} // ARE {are:.4f} // Energy {energy:.4f} // Num factor relinearising {n_factor_relins}')

    graph.synchronous_iteration(robustify=True)

vis.vis_scene.view_from_graph(graph)
