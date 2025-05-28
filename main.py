from resopt.ntw.network import Network
from resopt.param.parameters import configs
from resopt.plot.plot import plot_convergence, plot_scatter_legend
from resopt.files.arrange import get_pareto_front_from_files
from resopt.files.file_utils import solutions_to_string
from resopt.analysis.analyze import generate_csv
from resopt.problem.solve_problem import solve
from resopt.problem.get_ref_points import solutions_to_ref_points, \
        lazy_ref_points
from resopt.problem.hybrid_solve import hybrid_solve

import random
import numpy as np
import sys

def paint_graph(ntw, seed=1):
    try:
        while True:
            print("Showing graph painted with seed = {}".format(seed))
            ntw.displayGraph(seed)
            seed += 1
    except KeyboardInterrupt:
        pass

# MAIN
# ==============================================================================

if __name__ == "__main__":

    random.seed(configs.seed)

    if configs.command == 'generate':
        # Generate the network
        ntw = Network(configs)

        if configs.print:
            print(ntw.getTotalNodeMemory())
            print(ntw.getTotalTaskMemory())
            print(ntw.memory)
            print(ntw.getMinimumNNodesNeeded())

        if configs.paint:
            paint_graph(ntw, configs.paint_seed)

        if configs.output:
            import pickle
            pickle.dump(ntw, configs.output)
            configs.output.close()

        if not ntw.checkMemoryRequirements():
            print('WARNING: Memory requirements could not be satisfied.')
            print('- Total node memory: {}'.format(
                    ntw.getTotalNodeMemory()))
            print('- Total task memory: {}'.format(
                    ntw.getTotalTaskMemory()))
            print('Change memory limit or amount of tasks and try again')
            sys.exit(1)

    elif configs.command == 'modify':
        import pickle
        ntw = pickle.load(configs.input)

    elif configs.command == 'solve':
        # Solve a problem using a network and an optimization algorithm
        import pickle
        ntw = pickle.load(configs.input)

        sol_f = solve(ntw, configs)
        if sol_f is None:
            if configs.output:
                configs.output.close()
            sys.exit(1)

        if configs.print:
            print(sol_f)

        if configs.output:
            configs.output.write(sol_f)
            configs.output.close()

    elif configs.command == 'hybrid_solve':
        # Hybrid solving dumping solutions
        import pickle
        ntw = pickle.load(configs.input)
        hybrid_solve(ntw, configs)

    elif configs.command == 'arrange':
        # Arrange files of solutions and prepare them for ploting
        array = get_pareto_front_from_files(configs)

        s = solutions_to_string(array)

        if configs.print:
            print(s)

        if configs.output:
            configs.output.write(s)
            configs.output.close()

    elif configs.command == 'get_ref_points':
        if configs.lazy:
            s = lazy_ref_points(configs).tolist().__repr__()
        else:
            s = solutions_to_ref_points(configs)

        if configs.output:
            configs.output.write(s)
            configs.output.close()
             
    elif configs.command == 'analyze':
        generate_csv(configs)
    
    elif configs.command == 'plot':
        # Plot files
        if configs.comparison:
            plot_scatter_legend(configs)
        if configs.history:
            plot_convergence(configs)



