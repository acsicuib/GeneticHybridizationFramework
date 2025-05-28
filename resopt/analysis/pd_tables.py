import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

ALGORITHM = 'UNSGA3'
POP_SIZE = 300
N_GEN = 400
METRIC = 'GD' # change with the desired metric


def my_boxplot(df, ax, qry = '', metric = '', gby = ''):
    print(df)
    df.query(qry).boxplot(metric, by=gby, ax=ax)
    ax.set_title('Metric: ' + metric + ', ' + qry)

def my_surface(df, ax, x, y, z):
    df2 = pd.DataFrame(df.query(f'Algorithm == "{ALGORITHM}"').groupby([x, y])[z].mean()).reset_index()
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.plot_trisurf(df2[x], df2[y], df2[z])

def df_min_max_mean_variance(df):
    df2 = pd.DataFrame(
            df.groupby(
                    ['Algorithm', 'Population', 'Generation']
                ).agg(
                    ['min', 'max', 'mean', 'var']
                    ).reset_index())
    
    return df2.drop('Seed', axis=1)

def compareMetricsPopulationAndGeneration(df):
    #df2 = df_min_max_mean_variance(df)

    #print(df2) # use less -S command
    #print(df2.to_html()) # redirect output to .html file
    #df2.to_csv('table_grouped.csv', index=False)

    #my_boxplot(
    #        df,
    #        qry = f'Algorithm == "NSGA2" and Generation == {N_GEN}',
    #        metric = METRIC,
    #        gby = 'Population')

    fig1 = plt.figure()
    fig2 = plt.figure()

    i = 1
    for METRIC in ('Solutions','GD','IGD','HV','S','STE'):
        ax1 = fig1.add_subplot(2, 3, i)
        my_boxplot(
                df, ax1,
                qry = f'Algorithm == "{ALGORITHM}" and Population == {POP_SIZE}',
                metric = METRIC,
                gby = 'Generation')

        ax2 = fig2.add_subplot(2, 3, i, projection='3d')
        my_surface(df, ax2, 'Population', 'Generation', METRIC)

        i += 1

    #plt.show()

def hybridAlgorithmMetrics(df, gen_steps=[], hybrid=True):
    fig = plt.figure("Metric comparison between algorithms ({}hybrid)".format(
        '' if hybrid else 'non-'))
    fig.suptitle("Metric comparison between algorithms ({}hybrid)".format(
        '' if hybrid else 'non-'))

    i = 1
    df_group = df.groupby(['Algorithm'])
    for METRIC in ('Solutions','GD','IGD','HV','S','STE'):
        ax = fig.add_subplot(2, 3, i)
        ax.set_ylabel(METRIC)

        for key, grp in df_group:
            grp.plot(ax=ax, kind='line', x='Generation', y=METRIC, label=key[0])
            ax.set_xticks(gen_steps)

        ax.grid(which='both', axis='x', linestyle='--')

        i += 1

    #plt.show()

def hybridNoHybridAlgorithmComparison(df1, df2, gen_steps=[]):
    algorithms = df1['Algorithm'].unique()

    fig = {key: plt.figure(
            "Metric comparison between {} hybrid and non-hybrid".format(key)
        ) for key in algorithms}

    for key in algorithms:
        fig[key].suptitle("Metric comparison between {} hybrid and non-hybrid".format(key))
        label1 = "{} hybrid".format(key)
        label2 = "{} non-hybrid".format(key)

        i = 1
        for METRIC in ('Solutions','GD','IGD','HV','S','STE'):
            ax = fig[key].add_subplot(2, 3, i)
            ax.set_title("Metric {}".format(METRIC))
            ax.set_ylabel(METRIC)

            df1[df1['Algorithm'] == key].plot(
                    ax=ax, kind='line', x='Generation', y=METRIC, label=label1)
            df2[df2['Algorithm'] == key].plot(
                    ax=ax, kind='line', x='Generation', y=METRIC, label=label2)
            ax.set_xticks(gen_steps)

            ax.grid(which='both', axis='x', linestyle='--')
            i += 1

    #plt.show()

if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    df = pd.read_csv(sys.argv[1])
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    if len(sys.argv) > 2:
        df2 = pd.read_csv(sys.argv[2])
        df2 = df2.loc[:, ~df2.columns.str.contains('^Unnamed')]
        gen_steps = [int(arg) for arg in sys.argv[3:]]

    compareMetricsPopulationAndGeneration(df)
    #hybridAlgorithmMetrics(df, gen_steps, hybrid=True)
    #hybridAlgorithmMetrics(df2, gen_steps, hybrid=False)
    #hybridNoHybridAlgorithmComparison(df, df2, gen_steps)

    plt.show()
    






