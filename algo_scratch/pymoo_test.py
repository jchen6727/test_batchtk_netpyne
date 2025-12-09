from pymoo.algorithms.soo.nonconvex.ga import GA

from batchtk.runtk.trial import trial

from batchtk.runtk import constructors
from batchtk.runtk.trial import LABEL_POINTER, DIR_POINTER


from pymoo.optimize import minimize

from pymoo.core.problem import Problem

class TrialProblem(Problem):

    def __init__(self, trial, n_var, n_obj, n_constr, xl, xu):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)
        self.trial = trial

    def _evaluate(self, x, out, *args, **kwargs):
        results = trial(
            config={f'x{i}': x[i] for i in range(self.n_var)},
            label='single_trial',
            tid=0,
            dispatcher_constructor=constructors.LocalDispatcher,
            project_dir='.',
            output_dir='./output',
            storage_dir='./storage',
            interval=1,
            submit_constructor=constructors.LocalSubmit,
        )
        out["F"] = results['objective']

problem = get_problem("g1")

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