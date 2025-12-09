from netpyne.batchtools.search import generate_constructors
from batchtk.algos import optuna_search
from batchtk.utils import TOTPConnection, TomlParser, expand_path
import tomllib as toml


Dispatcher, Submit = generate_constructors('ssh_slurm', 'sftp')

del Submit

with open('../secret_key.toml', 'rb') as fptr:
    secret_key = toml.load(fptr)['SECRET_KEY']

parser = TomlParser(file_path='../hpc_toml/expanse.toml')
Submit = parser.get_submit_class()
print(Submit())
results = optuna_search(
    study_label='rosenbrock',
    param_space={'x0': (-5, 5), 'x1': (-5, 5)},
    param_space_samplers=['int', 'int'],  # specify integer sampling for both parameters
    metrics={'fx': 'minimize'},
    num_trials=3, num_workers=3,
    dispatcher_constructor=Dispatcher,
    dispatcher_kwargs={'connection_constructor': TOTPConnection,
                       'connection_kwargs': {'host': 'expanse0',
                                             'key': secret_key}},
    submit_constructor=Submit,
    submit_kwargs={}, # normal run
    interval=10,
    project_dir='/home/jchen12/dev/test_batchtk_netpyne/sim_scripts',
    output_dir='/home/jchen12/dev/test_batchtk_netpyne/output_optuna',
    storage_dir=expand_path('/Users/jchen/dev/test_batchtk_netpyne/checkpoint_optuna', create_dirs=True),
)