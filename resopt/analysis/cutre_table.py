import argparse
import sys

parser = argparse.ArgumentParser(description="Arguments")

parser.add_argument('-i', '--input', nargs='+', type=argparse.FileType('r'), help='List of input file paths used for cutretabling the solutions')
parser.add_argument('-r', '--row_start', type=int, default=1, help='Start row')
parser.add_argument('-c', '--col_start', type=int, default=1, help='Start column')
parser.add_argument('--col_size', type=int, default=12, help='Size of a column in characters')

configs = parser.parse_args()

if __name__ == '__main__':
    tables = []
    for f in configs.input:
        lines = f.read().splitlines()
        f.close()
        tables.append([l.split() for l in lines])

    n_tables = len(tables)
    rs, re = configs.row_start, len(tables[0])
    cs, ce = configs.col_start, len(tables[0][0])

    # Cast string to float
    for t in range(n_tables):
        for i in range(rs,re):
            for j in range(cs,ce):
                tables[t][i][j] = float(tables[t][i][j])

    # Fill cutretable
    table_min = [[0. for _ in range(cs, ce)] for _ in range(rs, re)]
    table_max = [[0. for _ in range(cs, ce)] for _ in range(rs, re)]
    table_avg = [[0. for _ in range(cs, ce)] for _ in range(rs, re)]
    table_var = [[0. for _ in range(cs, ce)] for _ in range(rs, re)]
    for i in range(rs,re):
        for j in range(cs,ce):
            table_min[i-rs][j-cs] = min([t[i][j] for t in tables])
            table_max[i-rs][j-cs] = max([t[i][j] for t in tables])
            table_avg[i-rs][j-cs] = sum([t[i][j] for t in tables]) / n_tables
            table_var[i-rs][j-cs] = sum([
                    (t[i][j] - table_avg[i-rs][j-cs])**2
                    for t in tables
                ]) / n_tables

    # New cutretable
    s = ''
    new_rs, new_re = rs,     rs + (re - rs) * 4
    new_cs, new_ce = cs, ce + 1
    new_rows, new_cols = new_re, new_ce
    for i in range(new_rows):
        mrow  = (new_rs <= i < new_re)
        i_4   = (i - new_rs) % 4
        #empty = (mrow and i_4 != 0)
        empty = True
        if i < new_rs:
            old_i = i
        elif mrow:
            old_i = rs + ((i - new_rs) // 4)
            if i_4 == 0:
                s += '\n'
        else:
            old_i = re + i - new_re

        for j in range(new_cols):
            mcol = (j == new_cs)
            scol = new_cs <= j
            if not mcol and scol:
                old_j = j - 1
            else:
                old_j = j

            # Celda
            if not mrow:
                if not mcol:
                    s += '{: <15}'.format(tables[0][old_i][old_j])
                else:
                    if i == 0:
                        s += '{: <15}'.format('Metric')
                    else:
                        s += '{: <15}'.format('')
            else:
                if not scol:
                    if empty:
                        s += '{: <15}'.format('')
                    else:
                        s += '{: <15}'.format(tables[0][old_i][old_j])
                else:
                    if not mcol:
                        if i_4 == 0:
                            s += '{: <15}'.format('{:.10f}'.format(
                                table_min[old_i-rs][old_j-cs]))
                        if i_4 == 1:
                            s += '{: <15}'.format('{:.10f}'.format(
                                table_max[old_i-rs][old_j-cs]))
                        if i_4 == 2:
                            s += '{: <15}'.format('{:.10f}'.format(
                                table_avg[old_i-rs][old_j-cs]))
                        if i_4 == 3:
                            s += '{: <15}'.format('{:.10f}'.format(
                                table_var[old_i-rs][old_j-cs]))
                    else:
                        if i_4 == 0:
                            s += '{: <15}'.format('minimum')
                        if i_4 == 1:
                            s += '{: <15}'.format('maximum')
                        if i_4 == 2:
                            s += '{: <15}'.format('mean')
                        if i_4 == 3:
                            s += '{: <15}'.format('variance')
        s += '\n'

    print(s, end='')







