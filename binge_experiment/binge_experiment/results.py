from datetime import datetime
import sqlite3

import pandas as pd


class Results:

    def __init__(self, fname):

        self._fname = fname
        self._conn = sqlite3.connect(fname,
                                     detect_types=sqlite3.PARSE_DECLTYPES)
        self._conn.row_factory = sqlite3.Row

        self._setup()

    def _setup(self):

        cur = self._conn.cursor()

        cur.execute('CREATE TABLE IF NOT EXISTS results '
                    '(loss TEXT, '
                    'embedding_dim INTEGER, '
                    'n_iter INTEGER, '
                    'batch_size INTEGER, '
                    'l2 REAL, '
                    'learning_rate REAL, '
                    'use_cuda BOOLEAN, '
                    'xnor BOOLEAN, '
                    'mean_mrr REAL, '
                    'time TIMESTAMP) ')
        cur.execute('CREATE TABLE IF NOT EXISTS benchmark '
                    '(embedding_dim INTEGER, '
                    'xnor BOOLEAN, '
                    'duration REAL, '
                    'memory INTEGER, '
                    'time TIMESTAMP)')

    def save(self, hyperparameters, mrrs):

        data = hyperparameters.copy()
        data['mean_mrr'] = mrrs.mean()
        data['time'] = datetime.now()

        cur = self._conn.cursor()

        cur.execute('INSERT INTO results '
                    'VALUES (:loss, :embedding_dim, :n_iter, '
                    ':batch_size, :l2, :learning_rate, :use_cuda, :xnor, '
                    ':mean_mrr, :time)', data)
        self._conn.commit()

    def save_benchmark(self, embedding_dim, xnor, duration, memory):

        cur = self._conn.cursor()

        cur.execute('INSERT INTO benchmark '
                    'VALUES (:embedding_dim, '
                    ':xnor, :duration, :memory, :time)',
                    {'embedding_dim': embedding_dim,
                     'xnor': xnor,
                     'duration': duration,
                     'memory': memory,
                     'time': datetime.now()})

        self._conn.commit()

    def clear_benchmarks(self):

        cur = self._conn.cursor()

        cur.execute('DELETE FROM benchmark')

        self._conn.commit()

    def __contains__(self, hyperparameters):

        cur = self._conn.cursor()

        cur.execute('SELECT COUNT(*) FROM results '
                    'WHERE loss=:loss AND embedding_dim=:embedding_dim '
                    'AND n_iter=:n_iter AND batch_size=:batch_size '
                    'AND l2=:l2 AND learning_rate=:learning_rate '
                    'AND use_cuda=:use_cuda AND xnor=:xnor', hyperparameters)

        return cur.fetchone()[0]

    def load_best(self, embedding_dim, xnor):

        cur = self._conn.cursor()

        cur.execute('SELECT loss, embedding_dim, n_iter, '
                    'batch_size, l2, learning_rate, use_cuda, xnor FROM results '
                    'WHERE embedding_dim = :embedding_dim '
                    'AND xnor = :xnor '
                    'ORDER BY mean_mrr DESC LIMIT 1',
                    {'embedding_dim': embedding_dim,
                     'xnor': xnor})

        return dict(cur.fetchone())

    def __del__(self):

        self._conn.close()

    def load(self, best_only=False):

        cur = self._conn.cursor()

        cur.execute('SELECT loss, results.embedding_dim AS embedding_dim, n_iter, '
                    'batch_size, l2, learning_rate, use_cuda, results.xnor AS xnor, mean_mrr, '
                    'COALESCE(duration, 0.0) AS duration, '
                    '0.001 / COALESCE(duration, 0.0) AS qpms, '
                    'memory '
                    'FROM results '
                    'LEFT JOIN benchmark ON ('
                    'results.embedding_dim = benchmark.embedding_dim '
                    'AND results.xnor = benchmark.xnor) '
                    'ORDER BY results.embedding_dim, results.xnor ASC')

        data = [dict(x) for x in cur.fetchall()]

        if not data:
            raise Exception('No data to be read')

        if best_only:
            return (pd.DataFrame(data)
                    .sort_values('mean_mrr', ascending=False)
                    .groupby(['embedding_dim', 'xnor'], as_index=False)
                    .first()
                    .sort_values(['embedding_dim', 'xnor']))
        else:
            return pd.DataFrame(data)

    def plot(self, fname):

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("ticks")

        data = self.load()

        xnor_data = data[data['xnor'] == 1]
        float_data = data[data['xnor'] == 0]

        fig, ax = plt.subplots()
        ax.set(xscale='log',
               xlabel='Predictions per millisecond',
               ylabel='MRR')
        ax.plot(xnor_data['qpms'],
                xnor_data['mean_mrr'],
                label='XNOR')
        ax.plot(float_data['qpms'],
                float_data['mean_mrr'],
                label='Real-valued')
        ax.legend()
        fig.suptitle('MRR-Scores Per Millisecond Tradeoff')
        fig.savefig(fname)
