from netpyne.batchtools.search import generate_constructors
from batchtk.algos import cmaes_search
from batchtk.utils import TOTPConnection, expand_path
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
    'custom': ('source ~/.bashrc\n'
               'conda activate netpyne\n'),
    'command': 'python rosenbrock.py',
}

results = cmaes_search(
    study_label='rosenbrock',
    param_space={'x0': (-5, 5), 'x1': (-5, 5)},
    param_space_samplers=['int', 'int'],  # specify integer sampling for both parameters
    metrics={'fx': 'minimize'},
    num_trials=12, num_workers=3,
    dispatcher_constructor=Dispatcher,
    dispatcher_kwargs = {'connection_constructor': TOTPConnection,
                         'connection_kwargs': {'host': 'expanse0',
                                               'key': secret_key}},
    submit_constructor=Submit,
    submit_kwargs=slurm_args, # normal run
    interval=10,
    project_dir='/home/jchen12/dev/test_batchtk_netpyne/sim_scripts',
    output_dir='/home/jchen12/dev/test_batchtk_netpyne/output_cmaes',
    storage_dir=expand_path('/Users/jchen/dev/test_batchtk_netpyne/checkpoint_cmaes', create_dirs=True),
)
