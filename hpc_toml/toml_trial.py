#from batchtk.algos import optuna_search
from batchtk.utils import TOTPConnection, TomlParser, expand_path
from batchtk.runtk import SSHDispatcher
from batchtk.utils import expand_path
from batchtk.runtk.trial import trial, LABEL_POINTER, DIR_POINTER
import tomllib as toml

with open('../secret_key.toml', 'rb') as fptr:
    secret_key = toml.load(fptr)['SECRET_KEY']

parser = TomlParser(file_path='expanse.toml')
Submit = parser.get_submit_class()
print(Submit())
config = {'x0': 5,
          'x1': 5,
          '_batchtk_label_pointer': LABEL_POINTER,
          '_batchtk_path_pointer': DIR_POINTER}

results = trial(
    config=config,
    label='single_trial',
    tid=0,
    dispatcher_constructor=SSHDispatcher,
    dispatcher_kwargs={'connection_constructor': TOTPConnection,
                       'connection_kwargs': {'host': 'expanse0',
                                             'key': secret_key}},
    submit_constructor=Submit,
    interval=10,
    project_dir='/home/jchen12/dev/test_batchtk_netpyne/sim_scripts',
    output_dir='/home/jchen12/dev/test_batchtk_netpyne/output_optuna',
    storage_dir=expand_path('/Users/jchen/dev/test_batchtk_netpyne/checkpoint_optuna', create_dirs=True),
)
