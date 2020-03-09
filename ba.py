"""
Bundle Adjustment using GBP.
"""

import numpy as np
import argparse
from gbp import gbp_ba
import vis

parser = argparse.ArgumentParser()
parser.add_argument("--bal_file", required=True,
                    help="BAL style file with BA data")

parser.add_argument("--n_iters", type=int, default=200,
                    help="Number of iterations of GBP")

parser.add_argument("--gauss_noise_std", type=int, default=2,
                    help="Standard deviation of Gaussian noise of measurement model.")
parser.add_argument("--loss", default=None,
                    help="Loss function: None (squared error), huber or constant.")
parser.add_argument("--Nstds", type=float, default=3.,
                    help="If loss is not None, number of stds at which point the "
                         "loss transitions to linear or constant.")
parser.add_argument("--beta", type=float, default=0.01,
                    help="Threshold for the change in the mean of adjacent beliefs for "
                         "relinearisation at a factor.")
parser.add_argument("--num_undamped_iters", type=int, default=6,
                    help="Number of undamped iterations at a factor node after relinearisation.")
parser.add_argument("--min_linear_iters", type=int, default=8,
                    help="Minimum number of iterations between consecutive relinearisations of a factor.")
parser.add_argument("--eta_damping", type=float, default=0.4,
                    help="Max damping of information vector of messages.")

parser.add_argument("--prior_std_weaker_factor", type=float, default=50.,
                    help="Ratio of std of information matrix at measurement factors / "
                         "std of information matrix at prior factors.")

parser.add_argument("--float_implementation", action='store_true', default=False,
                    help="Float implementation, so start with strong priors that are weakened")
parser.add_argument("--final_prior_std_weaker_factor", type=float, default=100.,
                    help="Ratio of information at measurement factors / information at prior factors "
                         "after the priors are weakened (for floats implementation).")
parser.add_argument("--num_weakening_steps", type=int, default=5,
                    help="Number of steps over which the priors are weakened (for floats implementation)")

args = parser.parse_args()

print('Configs: \n', args)


configs = dict({
    'gauss_noise_std': args.gauss_noise_std,
    'loss': args.loss,
    'Nstds': args.Nstds,
    'beta': args.beta,
    'num_undamped_iters': args.num_undamped_iters,
    'min_linear_iters': args.min_linear_iters,
    'eta_damping': args.eta_damping,
    'prior_std_weaker_factor': args.prior_std_weaker_factor,
           })

if args.float_implementation:
    configs['final_prior_std_weaker_factor'] = args.final_prior_std_weaker_factor
    configs['num_weakening_steps'] = args.num_weakening_steps
    weakening_factor = np.log10(args.final_prior_std_weaker_factor) / args.num_weakening_steps


graph = gbp_ba.create_ba_graph(args.bal_file, configs)
print(f'\nData: {args.bal_file}\n')
print(f'Number of keyframes: {len(graph.cam_nodes)}')
print(f'Number of landmarks: {len(graph.lmk_nodes)}')
print(f'Number of measurement factors: {len(graph.factors)}\n')

# Sets prior factors automatically to be much weaker than measurement factors.
graph.generate_priors_var(weaker_factor=args.prior_std_weaker_factor)
graph.update_all_beliefs()

# Set up visualisation
scene = vis.ba_vis.create_scene(graph)
viewer = vis.ba_vis.TrimeshSceneViewer(scene=scene, resolution=scene.camera.resolution)
viewer.show()


for i in range(args.n_iters):
    # To copy weakening of strong priors as must be done on IPU with float
    if args.float_implementation and (i+1) % 2 == 0 and (i < args.num_weakening_steps * 2):
        print('Weakening priors')
        graph.weaken_priors(weakening_factor)

    # At the start, allow a larger number of iterations before linearising
    if i == 3 or i == 8:
        for factor in graph.factors:
            factor.iters_since_relin = 1

    are = graph.are()
    energy = graph.energy()
    n_factor_relins = 0
    for factor in graph.factors:
        if factor.iters_since_relin == 0:
            n_factor_relins += 1
    print(f'Iteration {i} // ARE {are:.4f} // Energy {energy:.4f} // Num factors relinearising {n_factor_relins}')

    viewer.update(graph)

    graph.synchronous_iteration(robustify=True, local_relin=True)

