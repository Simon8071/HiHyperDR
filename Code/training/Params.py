import argparse
import json
import os
import sys


TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(TRAINING_DIR)


def ParseArgs():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument(
        '--dataset',
        default='GDSC',
        choices=['GDSC', 'DrugBank', 'Drugbank', 'PDTC', 'TCGA', 'SingleCell'],
        help='dataset adapter to use',
    )
    parser.add_argument(
        '--data_path',
        default=os.path.join(CODE_ROOT, "data"),
        help='common data root or a dataset-specific directory',
    )
    parser.add_argument(
        '--config',
        default=None,
        help='JSON parameter file; explicit command-line options take precedence',
    )
    parser.add_argument(
        '--checkpoint_dir',
        default=None,
        help='optional checkpoint directory; defaults to a dataset-specific path',
    )
    parser.add_argument(
        '--test_size',
        default=0.2,
        type=float,
        help='test fraction used when PDTC or TCGA must be split at runtime',
    )
    parser.add_argument(
        '--random_feature_dim',
        default=128,
        type=int,
        help='reproducible random gene-feature dimension for DrugBank',
    )
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--tstBat', default=100000, type=int, help='number of interactions in a testing batch')
    parser.add_argument('--epoch', default=3000, type=int, help='number of epochs')
    parser.add_argument('--latdim', default=128, type=int, help='embedding size')
    parser.add_argument('--hyperNum', default=64, type=int, help='number of hyperedges')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='Number of output classes (2 for binary classification)')
    parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
    parser.add_argument('--keepRate', default=0.7, type=float, help='ratio of edges to keep')
    parser.add_argument('--temp', default=0.5, type=float, help='temperature')
    parser.add_argument('--mult', default=1e-1, type=float, help='multiplication factor')
    parser.add_argument('--ssl_reg', default=0.01, type=float, help='weight for ssl loss')
    parser.add_argument('--tstEpoch', default=5, type=int, help='number of epoch to test while training')
    parser.add_argument('--gpu', default=0, type=int, help='indicates which gpu to use')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed')
    parser.add_argument('--dense', action='store_true', default=True, help='dense')
    parser.add_argument('--global_cl_reg', default=0.01, type=float, help='weight for global-level supervised contrastive loss')
    parser.add_argument('--local_cl_reg', default=0.01, type=float, help='weight for local-level supervised contrastive loss')
    parser.add_argument('--proto_t', default=0.1, type=float, help='temperature for prototype-based contrastive learning')
    parser.add_argument('--reg', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--use_contrastive_loss', action='store_true', default=False,
                    help='whether to use global and local contrastive loss')
    if os.environ.get('HIHYPERDR_SINGLE_CELL_DDP') == '1':
        distributed = parser.add_argument_group('single-cell distributed training')
        distributed.add_argument(
            '--gpus', default=4, type=int,
            help='number of local GPUs used by the SingleCell distributed trainer',
        )
        distributed.add_argument(
            '--master_addr', default='127.0.0.1',
            help='DDP rendezvous address for single-node training',
        )
        distributed.add_argument(
            '--master_port', default=12355, type=int,
            help='DDP rendezvous port',
        )
        distributed.add_argument(
            '--single_cell_group',
            default='ALL',
            choices=['ALL', 'cancer', 'drug', 'tissue'],
            help='single-cell experiment group under Code/data/single cell',
        )
        distributed.add_argument(
            '--single_cell_subset',
            default=None,
            help=(
                'subset name inside cancer/drug/tissue, for example '
                '"Breast cancer" or "Chemotherapy"; ALL uses the ALL cohort'
            ),
        )
    evaluation = parser.add_argument_group('prediction evaluation')
    evaluation.add_argument(
        '--split',
        default=None,
        help='evaluation CSV filename for datasets with a predefined split',
    )
    evaluation.add_argument(
        '--model',
        choices=['focal'],
        default='focal',
        help='retained trained model used for prediction evaluation',
    )
    evaluation.add_argument(
        '--checkpoint',
        default=None,
        help='custom checkpoint path; overrides --model when provided',
    )
    evaluation.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='classification threshold used for prediction evaluation',
    )
    args, _ = parser.parse_known_args()

    if args.config:
        with open(args.config, 'r', encoding='utf-8') as handle:
            config = json.load(handle)
        explicit_options = {
            token[2:].split('=', 1)[0]
            for token in sys.argv[1:]
            if token.startswith('--')
        }
        for key, value in config.items():
            if key.startswith('_') or not hasattr(args, key):
                continue
            if key not in explicit_options:
                setattr(args, key, value)
    return args


args = ParseArgs()
