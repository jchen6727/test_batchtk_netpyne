import uuid

from pymoo.algorithms.soo.nonconvex.ga import GA

from batchtk.runtk.trial import trial

from batchtk.runtk import constructors
from batchtk.runtk.trial import LABEL_POINTER, DIR_POINTER


from pymoo.optimize import minimize

from pymoo.core.problem import Problem

class TrialProblem(Problem):
    def __init__(self, label: str, params: dict[str, tuple[float, float]], n_obj, n_constr

    dispatcher_constructor: callable = None, project_dir: str = None,
    output_dir: str = None, submit_constructor: callable = None,
    storage_dir: str = None, dispatcher_kwargs: Optional[dict] = None,
    submit_kwargs: Optional[dict] = None, interval: Optional[int] = 60,
    storage_constructor: Optional[callable] = constructors.SQLiteStorage,
    storage_kwargs: Optional[dict] = None,
    log_constructor: Optional[callable] = constructors.BatchtkLogger,
    log_kwargs: Optional[dict] = None, report: Optional[list] = ('path', 'config', 'data'),
    cleanup: Optional[bool | list | tuple] = (runtk.SGLOUT, runtk.MSGOUT),
    check_storage: Optional[bool] = True, **kwargs) -> pandas.DataFrame:


                 ):
        n_var = len(params)
        self.params, xb = zip(*params)
        xl, xu = zip(*xb)
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)
        self.label = label
    def _evaluate(self, x, out, *args, **kwargs):
        tid = uuid.uuid4()
        results = trial(
            config={f'x{i}': x[i] for i in range(self.n_var)},
            label=self.label,
            tid=tid,
            dispatcher_constructor=constructors.LocalDispatcher,
            project_dir='.',
            output_dir='./output',
            storage_dir='./storage',
            interval=1,
            submit_constructor=constructors.LocalSubmit,
        )
        out["F"] = results['objective']


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