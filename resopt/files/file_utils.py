import numpy as np

def parse_file(f, n_obj=2, n_algs=0, n_steps=0, timestamps=False, population=False, return_dict=False):
    """
    Possible formats for the columns:
        <ts_date> <ts_time> <isSolution> <generation> <o1> ... <on> (<a1> ... <am>)*n_steps
        <ts_date> <ts_time> <generation> <o1> ... <on> (<a1> ... <am>)*n_steps
        <isSolution> <generation> <o1> ... <on> (<a1> ... <am>)*n_steps
        <generation> <o1> ... <on> (<a1> ... <am>)*n_steps
        <o1> ... <on> (<a1> ... <am>)*n_steps
    """
    first = f.readline().split()
    columns = len(first)
    lst = first + f.read().split()
    f.close()

    o = [None] * n_obj
    a = [[None for _ in range(n_algs)] for _ in range(n_steps)] 
    generation = []

    first_column = columns - (n_obj + n_steps * n_algs)
    for i in range(n_obj):
        o[i] = [float(e) for e in lst[(first_column + i)::columns]]

    for i in range(n_steps):
        for j in range(n_algs):
            a[i][j] = [float(e) for e in lst[(first_column + n_obj + i * n_algs)::columns]]

    if first_column > 0:
        generation = [int(e) for e in lst[first_column-1::columns]]

    # Generate dictionary
    parsed = dict()

    if timestamps:
        parsed['timestamp'] = [lst[::columns], lst[1::columns]]

    if population:
        parsed['is_solution'] = [int(e) for e in lst[(first_column - 2)::columns]]

    if generation:
        parsed['generation'] = generation

    parsed['objective'] = o
    
    if n_algs > 0 and n_steps > 0:
        parsed['genetic_load'] = a

    if return_dict:
        return parsed
        
    # TODO: This is all trash. Return a dictionary. Or even better: a Pandas dataframe.
    if timestamps:
        if population:
            if n_algs > 0:
                return parsed['timestamp'], parsed['is_solution'], generation, o, a
            else:
                return parsed['timestamp'], parsed['is_solution'], generation, o
        else:
            if n_algs > 0:
                return parsed['timestamp'], generation, o, a
            else:
                return parsed['timestamp'], generation, o
    else:
        if population:
            if n_algs > 0:
                return parsed['is_solution'], generation, o, a
            else:
                return parsed['is_solution'], generation, o
        else:
            if n_algs > 0:
                return generation, o, a
            else:
                return generation, o

def get_solution_array(
        f, n_obj=2, n_algs=0, n_steps=0, timestamps=False, population=False, return_dict=False):

    solutions = []

    # FILE PARSE
    parsed = parse_file(
            f,
            n_obj=n_obj,
            n_algs=n_algs,
            n_steps=n_steps,
            timestamps=timestamps,
            population=population,
            return_dict=True
        )

    if timestamps:
        ts = parsed.get('timestamp', None)
        ts_date_uniq = []
        ts_time_uniq = []

    #generation = np.array(parsed.get('generation', []), dtype=np.uint32)
    generation = parsed.get('generation', [])
    o = np.array(parsed.get('objective', []), dtype=np.float32)
    is_solution = np.array(parsed.get('is_solution', [True] * len(o[0])), dtype=np.bool_)

    if population:
        pop_list = []

    if n_algs > 0:
        gen_load = parsed.get('genetic_load', None)
        gl_xgen_pop = [[] for _ in range(n_steps)]
        gl_xgen_sol = [[] for _ in range(n_steps)]

    if len(generation) > 0:
        last_gen = generation[-1]

        # To avoid ValueError when calling method index
        generation.append(last_gen + 1)
        
        o_slize = [[] for _ in range(n_obj)]

        if n_algs > 0:
            gl_slize = [[[] for _ in range(n_algs)] for _ in range(n_steps)]

        for g in range(1, last_gen+1):
            i, j = generation.index(g), generation.index(g+1)
            for k in range(n_obj):
                o_slize[k] = o[k][i:j]
            if n_algs > 0:
                for k in range(n_steps):
                    gl_slize = []
                    for l in range(n_algs):
                        gl_slize.append(gen_load[k][l][i:j])
                    gl_xgen_pop[k].append(np.array(gl_slize).T)
                    gl_xgen_sol[k].append(np.array(gl_slize).T[is_solution[i:j]])

            if timestamps:
                ts_date_uniq.append(ts[0][i])
                ts_time_uniq.append(ts[1][i])

            solutions.append(np.array(o_slize).T[is_solution[i:j]])

            if population:
                pop_list.append(np.array(o_slize).T)

    else:
        solutions.append(np.array(o).T)
        for k in range(n_steps):
            gl_xgen_pop[k].append(np.array(gen_load[k]).T)
            gl_xgen_sol[k].append(np.array(gen_load[k]).T[is_solution])

    if return_dict:
        parsed = dict()

        if timestamps:
            parsed['date'] = ts_date_uniq
            parsed['time'] = ts_time_uniq

        if population:
            parsed['population'] = pop_list

        parsed['solutions'] = solutions
        
        if n_algs > 0:
            parsed['genetic_load_pop'] = gl_xgen_pop
            parsed['genetic_load_sol'] = gl_xgen_sol

        return parsed
        
    else:
        if timestamps:
            return ts_date_uniq, ts_time_uniq, solutions
        else:
            return solutions

def solutions_to_string(solutions):
    s = ''
    for o in solutions:
        for o_n in o:
            s += "{} ".format(o_n)
        s += '\n'
    return s

if __name__ == '__main__':
    from resopt.param.parameters import configs

    n_obj = len(configs.objectives)
    n_algs = configs.n_algorithms
    is_pop = configs.entire_population
    n_steps = len(configs.gen_steps)

    for f in configs.input:
        parsed = get_solution_array(
                f,
                n_obj=n_obj,
                n_algs=n_algs,
                n_steps=n_steps,
                timestamps=True,
                population=is_pop,
                return_dict=True
            )

        for key, val in parsed.items():
            print(key, len(val))
            if key in ("population", "solutions"):
                for g in range(len(val)):
                    print("  gen", g+1, len(val[g]))
            elif key in ("genetic_load_sol", "genetic_load_pop"):
                for step in range(len(val)):
                    print("  step", step+1, len(val[step]))
                    for g in range(len(val[step])):
                        print("    gen", g+1, len(val[step][g]))
