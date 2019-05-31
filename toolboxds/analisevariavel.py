import pandas as pd
import numpy as np
import scipy as sp
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import matplotlib.pyplot as plt
init_notebook_mode(connected=True)


def prepara_base_agrupada(df, var, by_var, chave='chave'):

    a = df.groupby([var, by_var], as_index=False).agg({chave: np.size})
    a['index'] = a[by_var].astype(str)  # + '\n' +  a[by_var].astype(str)
    b = df.groupby([by_var], as_index=False).agg({chave: np.size}).rename(columns={chave: 'count'})
    b['index'] = b[by_var].astype(str)  # + '\n' +  b[by_var].astype(str)
    a = a.merge(b, on='index', suffixes=('', '_'))
    a['%'] = a[chave] * 100 / a['count']
    return a.loc[:, [by_var, var, '%']]


def plotly_stackedbar(df_, var, by_var=None, title='', X_label=None, fl_geral=False):

    W = .25
    D = .25
    if by_var is not None:
        df_aux = df_.loc[:, [var, by_var]].copy()
    else:
        df_aux = df_.loc[:, [var]].copy()

    if by_var is None:
        by_var = var + '_'
    if by_var not in df_aux.columns:
        df_aux[by_var] = ' '

    for col in [var, by_var]:
        if df_aux.dtypes[col] == 'object':
            MISSING_VALUE = 'Missing'
        else:
            MISSING_VALUE = -1
        df_aux[col].fillna(MISSING_VALUE, inplace=True)

    if fl_geral & (by_var is not None):
        df_aux2 = df_aux.copy()
        df_aux2[by_var] = ' Geral'
        df_aux = pd.concat([df_aux, df_aux2])

    df_aux['chave'] = 1
    df = prepara_base_agrupada(df_aux, var, by_var, 'chave')
    df_pivot = df.rename(columns={by_var: by_var.split(' ')[0]}).pivot_table(
        index=var, columns=by_var.split(' ')[0]).fillna(0)
    display(df_pivot.loc[:, sorted(df_pivot.columns, key=lambda x:str(x))])  # [ str(col) for col in
    x_axis = by_var
    y_axis = var
    df[y_axis] = df[var].astype(str)
    df[x_axis] = df[by_var].astype(str)
    traces = []

    X = sorted(df[x_axis].unique())
    index = [(W + D)*i for i in range(len(X))]
    for i, cat in enumerate(sorted(df[y_axis].unique())):
        lista_pc = []
        for x in X:
            try:
                lista_pc.append(df.loc[(df[y_axis] == cat) & (df[x_axis] == x)]['%'].values[0])
            except:
                lista_pc.append(0)

        traces.append(go.Bar(
            x=index,
            y=lista_pc,
            width=W,
            name=cat))

    data = traces
    layout = go.Layout(
        barmode='stack',
        title='Distribuição da característica {}'.format(var),
        xaxis=dict(
            title=by_var,
            tickvals=index,
            ticktext=X
        ),
        yaxis=dict(
            title='% '
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='stacked-bar')


def plotly_boxplot(df_, var, by_var=None, title='', X_label=None, prefixo='', fl_geral=False):

    W = .25
    D = .7

    df = df_.copy()
    if by_var is None:
        by_var = var + '_'
    if by_var not in df.columns:
        df[by_var] = ' '

    for col in [by_var]:
        if df.dtypes[col] == 'object':
            MISSING_VALUE = 'Missing'
        else:
            MISSING_VALUE = -1
        df[col].fillna(MISSING_VALUE, inplace=True)
        df[col] = prefixo + df[col].astype(str)

    if fl_geral & (by_var is not None):
        df_aux2 = df.copy()
        df_aux2[by_var] = ' Geral'
        df = pd.concat([df, df_aux2])

    x_var = by_var
    y_var = var
    df[x_var] = df[by_var].astype(str)
    df[y_var] = df[var].astype(str)
    traces = []

    X = sorted(df[x_var].unique())
    index = [(W + D)*i for i in range(len(X))]

    for x, k in zip(X, index):
        y = df.loc[df[x_var] == x][y_var].dropna()
        traces.append(go.Box(
            y=y,
            name=str(x),
            boxmean=True))

    data = traces
    layout = go.Layout(
        barmode='stack',
        title='Distribuição da característica {}'.format(var),
        xaxis=dict(
            title=by_var,
        ),
        yaxis=dict(
            title=y_var
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='stacked-bar')


def describe_agrupado(df_, var, by_var, fl_geral=False, prefixo=''):

    df = df_.copy()
    if by_var is None:
        by_var = var + '_'
    if by_var not in df.columns:
        df[by_var] = ' '

    for col in [by_var]:
        if df.dtypes[col] == 'object':
            MISSING_VALUE = 'Missing'
        else:
            MISSING_VALUE = -1
        df[col].fillna(MISSING_VALUE, inplace=True)
        df[col] = prefixo + df[col].astype(str)

    if fl_geral & (by_var is not None):
        df_aux2 = df.copy()
        df_aux2[by_var] = ' Geral'
        df = pd.concat([df, df_aux2])
    x_var = by_var
    X = sorted(df[x_var].unique())
    for k, x in enumerate(X):
        if k == 0:
            df_describe = df.loc[df[x_var] == x][var].describe(
                percentiles=[.1, .25, .75, .99]).to_frame()
        else:
            df_describe[var + ' \n' + x] = df.loc[df[x_var] ==
                                                  x][var].describe(percentiles=[.1, .25, .75, .99])
    pd.options.display.float_format = '{:.2f}'.format
    display(df_describe)


def plotly_stackedbar_simple(df, col):

    if df.dtypes[col] == 'object':
        MISSING_VALUE = 'Missing'
    else:
        MISSING_VALUE = -99
    dist_ = df[col].fillna(MISSING_VALUE).value_counts(normalize=True)*100

    traces = []
    for cat in dist_.index:
        x = [col]
        y = [dist_[cat]]
        traces.append(go.Bar(
            x=x,
            y=y,
            width=0.1,
            name=cat))

    data = traces
    layout = go.Layout(
        barmode='stack',
        title='Distribuição da característica {}'.format(col),
        xaxis=dict(
            title=col
        ),
        yaxis=dict(
            title='% '
        ),
    )

    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='stacked-bar')


def plotly_hist(df, col):

    data = [go.Histogram(x=df[col],
                         )]
    layout = go.Layout(
        title='Histograma {}'.format(col),
        xaxis=dict(
            title=col
        ),
        yaxis=dict(
            title='# registros'
        ),
    )

    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='basic histogram')


def analise_univariada_num(df, var, by_var, fl_geral=False):

    print('------------------------------------------------------------------------------------------------------------')
    print('\t\t\t\t\t    Análise univariada')
    print('------------------------------------------------------------------------------------------------------------')
    print('Análise da variável: {}'.format(var))
    print('\tChecagem inicial: ')
    print('\t\t# registros total: {:.0f}'.format(df.shape[0]))
    print('\t\t# de nulos: {:.0f}'.format(
        df[var].replace([-np.inf, np.inf], np.nan).isnull().sum()))
    print('\t\t% de nulos: {:.2f}%'.format(df[var].isnull().sum()*100 / df.shape[0]))
    print('\t\t# valores distintos: {:.0f}'.format(df[var].nunique()))
    print('\t\t% valores distintos: {:.2f}%'.format(df[var].nunique()*100 / df.shape[0]))
    print('\n')
    print('\tDistribuição: ')
    describe_var = df[var].describe(percentiles=[.1, .25, .75, .99])
    print('\t\t#  não nulos\t: {:.0f}'.format(describe_var['count']))
    print('\t\tmédia \t\t: {:.2f}'.format(describe_var['mean']))
    print('\t\tstd   \t\t: {:.2f}'.format(describe_var['std']))
    print('\t\tmin   \t\t: {:.2f}'.format(describe_var['min']))
    print('\t\tp10   \t\t: {:.2f}'.format(describe_var['10%']))
    print('\t\tp25   \t\t: {:.2f}'.format(describe_var['25%']))
    print('\t\tp50   \t\t: {:.2f}'.format(describe_var['50%']))
    print('\t\tp75   \t\t: {:.2f}'.format(describe_var['75%']))
    print('\t\tp99   \t\t: {:.2f}'.format(describe_var['99%']))
    print('\t\tmax   \t\t: {:.2f}'.format(describe_var['max']))
    plotly_hist(df, var)
    if by_var is not None:
        print('\n')
        print('Análise da variável {} quebrada por {}'.format(var, by_var))
        describe_agrupado(df, var, by_var, prefixo='Intensidade :', fl_geral=fl_geral)
        plotly_boxplot(df, var, by_var, prefixo='Intensidade:\n', fl_geral=fl_geral)


def analise_univariada_cat(df, var, by_var, fl_geral=False):

    print('------------------------------------------------------------------------------------------------------------')
    print('\t\t\t\t\t    Análise univariada')
    print('------------------------------------------------------------------------------------------------------------')
    print('Análise da variável: {}'.format(var))
    print('\tChecagem inicial: ')
    print('\t\t# registros total: {:.0f}'.format(df.shape[0]))
    print('\t\t# de nulos: {:.0f}'.format(
        df[var].replace([-np.inf, np.inf], np.nan).isnull().sum()))
    print('\t\t% de nulos: {:.2f}%'.format(df[var].isnull().sum()*100 / df.shape[0]))
    print('\t\t# valores distintos: {:.0f}'.format(df[var].nunique()))
    print('\t\t% valores distintos: {:.2f}%'.format(df[var].nunique()*100 / df.shape[0]))
    print('\n')
    print('\tDistribuição: ')
    plotly_stackedbar(df, var, fl_geral=False)
    if by_var is not None:
        print('\n')
        print('Análise da variável {} quebrada por {}'.format(var, by_var, fl_geral=fl_geral))
        plotly_stackedbar(df, var, by_var, fl_geral=fl_geral)


def analise_cat_1(df, var, y, label='', fl_ordena=0, num=False, q=0, q2=0):

    if label == '':
        label = var
    df_ = df.loc[:, [var, y]].copy()
    if num:
        if q == 0:
            if q2 == 0:
                df_[var+'_cat'] = pd.qcut(df_[var], q=[0.0, 0.2, 0.5, 0.8, 1.0],
                                          retbins=True, duplicates='drop')[0].cat.add_categories('missing')
                var = var+'_cat'
            else:
                df_[var+'_cat'] = pd.qcut(df_[var], q=q2, retbins=True,
                                          duplicates='drop')[0].cat.add_categories('missing')
                var = var+'_cat'
        else:
            df_[var+'_cat'] = pd.cut(df_[var], bins=q).cat.add_categories('missing')
            var = var+'_cat'
    print('\n')
    print('Análise da variável {}'.format(var))
    print('# valores distintos {}'.format(df_[var].nunique()))
    try:
        df_[var] = df_[var].fillna('missing')
    except:
        df_[var] = df_[var].cat.add_categories('missing').fillna('missing')

    total_maus = df_[y].sum()
    total_bons = df_.shape[0] - df_[y].sum()

    def tx(x):
        return sum(x)*100/sum(1 - x + x)

    def pct(x):
        return sum(1-x+x)*1.0 / (df_.shape[0])

    def pct_maus(x):
        return (sum(x)/total_maus)

    def rr(x):
        return (sum(x)/total_maus)/(sum(1-x)/total_bons)

    def WoE(x):
        return np.log(rr(x))

    def IV(x):
        return -(sum(1-x)/total_bons - sum(x)/total_maus)*WoE(x)
    s = df_.groupby(var).agg({y: [np.size, pct, pct_maus, tx, rr, WoE, IV]})[y]

    def color_negative_green(val):
        color = 'green' if val < 0 else 'black'
        return 'color: %s' % color
    print('IV : {:.3f}'.format(s.IV.sum()))
    #print('Beta : {:.3f}'.format(dic_beta[var]))
    t = s.style.applymap(color_negative_green)
    display(pd.DataFrame(df_[var].value_counts()).join(pd.DataFrame(df_[var].value_counts(
        normalize=True)*100), lsuffix='k').rename(columns={var+'k': '#', var: '%'}))
    display(t)
    by_var = df_.groupby(var).agg({y: [np.size, np.mean]})[y]
    if fl_ordena == 1:
        by_var = by_var.sort_values(by='mean')
    Y1 = by_var['size']
    Y2 = by_var['mean']
    Y_mean = np.ones(shape=(len(Y1.index))) * df_[y].mean()
    index = np.arange(len(Y1.index))
    if True:
        plt.bar(index, Y1, alpha=0.3, color='gray')
        plt.grid(False)
        plt.xticks(rotation=20 if var[:2] not in ('fl', 'cd') else 0)
        plt.ylabel('# registros')
        plt.twinx()
        plt.gca().set_xticks(index)
        plt.gca().set_xticklabels([Y1.index[i] for i in index], rotation=40)
        plt.plot(index, Y_mean, label='tx. média evento')
        plt.plot(index, Y2, marker='o', label='tx. evento')
        plt.gca().set_yticklabels([' {:.2f}%'.format(i*100) for i in plt.gca().get_yticks()])
        plt.grid(False)
        plt.title('Bivariada  {}'.format(label))
        plt.ylabel('tx. evento')
        plt.xlabel(label)
        if var[:2] in ('fl', 'cd'):
            plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
        else:
            plt.legend(loc=9, bbox_to_anchor=(0.5, -0.35))
        plt.show()
        print(s.IV.sum())
        return (var, s.IV.sum())
        print('\n')


def plotly_analise_cat_1(df, var, y, label='', fl_ordena=0, num=False, q=0, q2=0):

    if label == '':
        label = var
    df_ = df.loc[:, [var, y]].copy()
    if num:
        if q == 0:
            if q2 == 0:
                df_[var+'_cat'] = pd.qcut(df_[var], q=[0.0, 0.2, 0.5, 0.8, 1.0],
                                          retbins=True, duplicates='drop')[0].cat.add_categories('missing')
                var = var+'_cat'
            else:
                df_[var+'_cat'] = pd.qcut(df_[var], q=q2, retbins=True,
                                          duplicates='drop')[0].cat.add_categories('missing')
                var = var+'_cat'
        else:
            df_[var+'_cat'] = pd.cut(df_[var], bins=q).cat.add_categories('missing')
            var = var+'_cat'
    print('\n')
    print('Análise da variável {}'.format(var))
    print('# valores distintos {}'.format(df_[var].nunique()))
    try:
        df_[var] = df_[var].fillna('missing')
    except:
        df_[var] = df_[var].cat.add_categories('missing').fillna('missing')

    total_maus = df_[y].sum()
    total_bons = df_.shape[0] - df_[y].sum()

    def tx(x):
        return sum(x)*100/sum(1 - x + x)

    def pct(x):
        return sum(1-x+x)*1.0 / (df_.shape[0])

    def pct_maus(x):
        return (sum(x)/total_maus)

    def rr(x):
        return (sum(x)/total_maus)/(sum(1-x)/total_bons)

    def WoE(x):
        return np.log(rr(x))

    def IV(x):
        return -(sum(1-x)/total_bons - sum(x)/total_maus)*WoE(x)
    s = df_.groupby(var).agg({y: [np.size, pct, pct_maus, tx, rr, WoE, IV]})[y]

    def color_negative_green(val):
        color = 'green' if val < 0 else 'black'
        return 'color: %s' % color
    print('IV : {:.3f}'.format(s.IV.sum()))
    #print('Beta : {:.3f}'.format(dic_beta[var]))
    t = s.style.applymap(color_negative_green)
    display(pd.DataFrame(df_[var].value_counts()).join(pd.DataFrame(df_[var].value_counts(
        normalize=True)*100), lsuffix='k').rename(columns={var+'k': '#', var: '%'}))
    display(t)
    by_var = df_.groupby(var).agg({y: [np.size, np.mean]})[y]
    if fl_ordena == 1:
        by_var = by_var.sort_values(by='mean')
    Y1 = by_var['size']
    Y2 = by_var['mean']*100
    Y_mean = np.ones(shape=(len(Y1.index))) * df_[y].mean()*100
    index = [i for i in np.arange(len(Y1.index))]

    trace1 = go.Bar(
        x=index,
        y=Y1,
        name='# registros'
    )
    trace2 = go.Scatter(
        x=index,
        y=Y2,
        name='tx. média evento',
        yaxis='y2'
    )

    trace3 = go.Scatter(
        x=index,
        y=Y_mean,
        name='tx. evento',
        yaxis='y2'
    )
    data = [trace1, trace2, trace3]
    layout = go.Layout(
        title='Bivariada  {}'.format(label),
        yaxis=dict(
            title='# registros'
        ),
        xaxis=dict(
            # title=by_var,
            tickvals=index,
            ticktext=[str(label) for label in Y2.index.tolist()]
        ),
        yaxis2=dict(
            title='% evento',
            titlefont=dict(
                color='rgb(148, 103, 189)'
            ),
            tickfont=dict(
                color='rgb(148, 103, 189)'
            ),
            overlaying='y',
            side='right'
        )
    )
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='multiple-axes-double')

    print(s.IV.sum())
    return (var, s.IV.sum())
    print('\n')
