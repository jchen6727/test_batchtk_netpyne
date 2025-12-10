import uuid

from pymoo.algorithms.soo.nonconvex.ga import GA

from batchtk.runtk.trial import trial

from batchtk import runtk
from batchtk.runtk import constructors
from batchtk.runtk.trial import LABEL_POINTER, DIR_POINTER
from batchtk.algos import Trial

from pymoo.optimize import minimize

from pymoo.core.problem import Problem
import numpy
from batchtk.utils import expand_path

class TrialProblem(Problem, Trial):
    def __init__(self, label: str, params: dict[str, tuple[float, float]],
                 metrics: dict, n_ieq_constr = 0, n_eq_constr = 0,
                 dispatcher_constructor = None, project_dir = None,
                 output_dir = None, submit_constructor = None,
                 storage_dir = None, dispatcher_kwargs = None,
                 submit_kwargs = None, interval = 60,
                 storage_constructor = constructors.SQLiteStorage,
                 storage_kwargs = None,
                 log_constructor = constructors.BatchtkLogger,
                 log_kwargs = None, report = ('path', 'config', 'data'),
                 cleanup = (runtk.SGLOUT, runtk.MSGOUT),
                 check_storage = True,
                 ):
        n_var = len(params)
        n_obj = len(metrics)
        self.metrics = sorted(metrics)
        self.params, xb = zip(*params)
        xl, xu = zip(*xb)
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr,
                         n_eq_constr=n_eq_constr, xl=xl, xu=xu)
        self._fixed_trial_args = dict()
        self.set_fixed_trial_args(
            dispatcher_constructor=dispatcher_constructor, project_dir=project_dir,
            output_dir=output_dir, submit_constructor=submit_constructor,
            storage_dir=storage_dir, dispatcher_kwargs=dispatcher_kwargs,
            submit_kwargs=submit_kwargs, interval=interval,
            storage_constructor=storage_constructor, storage_kwargs=storage_kwargs,
            log_constructor=log_constructor, log_kwargs=log_kwargs, report=report,
            cleanup=cleanup, check_storage=check_storage,
        )
        self.label = label
    def _evaluate(self, x, out, *args, **kwargs):
        config = {param: x for param, x in zip(self.params, x)}
        tid = self.compute_id_from_dict(self.label, config)
        results = self.run_trial(
            config=config,
            label=self.label,
            tid=tid
        )
        out["F"] = [results[metric] for metric in self.metrics]



dispatcher_constructor = constructors.LocalDispatcher
project_dir = expand_path('../runner_scripts', create_dirs=True)
output_dir = expand_path('./output', create_dirs=True)
submit_constructor = constructors.LocalSubmit
storage_dir = expand_path('./output', create_dirs=True)
submit_kwargs = {'command': 'python rosenbrock.py'}
storage_constructor = constructors.SQLiteStorage
log_constructor = constructors.BatchtkLogger

problem = TrialProblem(
    label='rosenbrock',
    params={'x0': (-2.048, 2.048), 'x1': (-2.048, 2.048)},
    metrics={'obj': 'minimize'},
    dispatcher_constructor=dispatcher_constructor,
    project_dir=project_dir,
    output_dir=output_dir,
    submit_constructor=submit_constructor,
    storage_dir=storage_dir,
    submit_kwargs=submit_kwargs,
    storage_constructor=storage_constructor,
    log_constructor=log_constructor,
)


algorithm = GA(
    pop_size=33,
    eliminate_duplicates=True)

res = minimize(problem,
               algorithm,
               seed=1,
               verbose=False)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))

"""
results with pop_size = 100
(dev) ➜  algo_scratch git:(main) ✗ time python pymoo_test.py
Best solution found:
X = [1.         1.         1.         1.         1.         1.
 1.         1.         1.         2.99221525 2.9943655  2.99054298
 1.        ]
F = [-14.97712372]
python pymoo_test.py  0.69s user 0.06s system 83% cpu 0.900 total



class ElementwiseEvaluationFunction:

    def __init__(self, problem, args, kwargs) -> None:
        super().__init__()
        self.problem = problem
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        out = dict()
        self.problem._evaluate(x, out, *self.args, **self.kwargs)
        return out

"""