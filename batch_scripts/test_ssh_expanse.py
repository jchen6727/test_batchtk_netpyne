from netpyne.batchtools.search import generate_constructors
from batchtk.algos import optuna_search
from batchtk.utils import TOTPConnection
import tomllib as toml

Dispatcher, Submit = generate_constructors('ssh_slurm', 'sftp')

with open('../secret_key.toml', 'rb') as fptr:
    secret_key = toml.load(fptr)['SECRET_KEY']

slurm_args = {
    'allocation': 'csd403',
    'realtime': '00:30:00',
    'nodes': '1',
    'coresPerNode': '1',
    'mem': '4G',
    'partition': 'shared',
    'email': 'jchen.6727@gmail.com',
    'custom': '',
    'comand': 'python single_opt.py',
}

results = optuna_search(
    study_label='rosenbrock',
    param_space={'x0': (-5, 5), 'x1': (-5, 5)},
    param_space_samplers=['int', 'int'],  # specify integer sampling for both parameters
    metrics={'fx': 'minimize'},
    num_trials=12, num_workers=3,
    dispatcher_constructor=Dispatcher,
    dispatcher_kwargs = {'connection': TOTPConnection('expanse0', secret_key )},
    submit_constructor=Submit,
    submit_kwargs=slurm_args, # normal run
    interval=10,
    project_path='.',
    output_path=expand_path('./optimization', create_dirs=True),
)
