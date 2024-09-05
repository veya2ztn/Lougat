
import argparse
def _print_args(args):
    """Print arguments."""
    if args.rank == 0:
        print('-------------------- arguments --------------------', flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (32 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print('---------------- end of arguments ----------------', flush=True)


def _check_arg_is_not_none(args, arg):
    assert getattr(args, arg) is not None, '{} argument is None'.format(arg)

import dataclasses
#--------------------------------------------------#
def check_is_args(args):
    if dataclasses.is_dataclass(args):return True
    if isinstance(args, argparse.Namespace):return True
    if hasattr(args, 'to_fields_dict'):return True
    return False




def get_print_namespace_tree(namespace, indent=0):
    result = ""
    if check_is_args(namespace):
        if hasattr(namespace, 'to_fields_dict'):
            namespace = namespace.to_fields_dict()
        else:
            namespace = vars(namespace) 
    for key, value in namespace.items():
        if key.startswith('_'):continue
        line = ' ' * indent
        if isinstance(value, dict) or check_is_args(value):
            line += key + "\n"
            line += get_print_namespace_tree(value, indent + 4)
        else:
            line += f"{key:30s} ---> {value}\n"

        result += line
    return result 

def print_namespace_tree(namespace):
    print(get_print_namespace_tree(namespace))
    # namespace = vars(namespace) if check_is_args(namespace) else namespace
    # for key, value in namespace.items():
    #     print(' ' * indent, end='')
    #     if isinstance(value, dict) or check_is_args(value):
    #         print(key)
    #         print_namespace_tree(value, indent + 4)
    #     else:
    #         print(f"{key:30s} ---> {value}")

def get_compare_namespace_trees(namespace1, namespace2, indent=0):
    result = ""
    namespace1 = vars(namespace1) if check_is_args(namespace1) else namespace1
    namespace2 = vars(namespace2) if check_is_args(namespace2) else namespace2

    for key in set(list(namespace1.keys()) + list(namespace2.keys())):
        line = ' ' * indent
        value1 = namespace1.get(key)
        value2 = namespace2.get(key)

        if ((isinstance(value1, dict) or check_is_args(value1)) and
           (isinstance(value2, dict) or check_is_args(value2))):
            line += key + "\n"
            line += get_compare_namespace_trees(value1, value2, indent + 4)
        else:
            if value1 is None:
                line += f"{key:30s} ---> [None]<=>[{value2}]"
            elif value2 is None:
                line += f"{key:30s} ---> [{value1}]<=>[None]"
            else:
                line += f"{key:30s} ---> [{value1}]<=>[{value2}]"
            if value1 != value2:
                line += "<======="
            line+="\n"
        result += line
    return result

def compare_namespace_trees(namespace1, namespace2):
    print(get_compare_namespace_trees(namespace1, namespace2))

def convert_namespace_tree(namespace):
    namespace = vars(namespace) if check_is_args(namespace) else namespace
    if isinstance(namespace,dict):
        return dict([(key, convert_namespace_tree(val)) for key, val in namespace.items()])
    else:
        return namespace

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
from dataclasses import fields
def get_default_config(cls):
    # Create a new instance without calling __init__ or __post_init__
    default_config = object.__new__(cls)
    
    # Populate the instance with default values
    for field in fields(cls):
        setattr(default_config, field.name, field.default)

    return default_config


def build_kv_list(d, parent_key='', sep='.',exclude_key=['downstream_pool']):
    items = []
    for k, v in d.items():
        new_key =  f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) and k not in exclude_key:
            items.extend(build_kv_list(v, new_key, sep=sep, exclude_key=exclude_key))
        else:
            items.append([k, v, new_key])
    return items


def retrieve_dict(d,exclude_key=['downstream_pool']):
    kvfk = build_kv_list(d, exclude_key=exclude_key)
    kkmap = {}
    for key, v, full_key in kvfk:
        if key not in kkmap:kkmap[key]=[]
        kkmap[key].append([full_key,v])

    
    pool={}
    for key, full_key_pool in kkmap.items():
        if len(full_key_pool)>1:

            if len(set([(tuple(v) if isinstance(v, list) else v) for k,v in full_key_pool]))==1:
                k, v = full_key_pool[0]
                pool[key] = v 
            else:
                print(f"key={key} conflict. Can be {[k for k,v in full_key_pool]}")
                
                assert key in ['strategy_name','early_warning', '_type_'], f"key={key} conflict. please check your config file"
                for k, v in full_key_pool:
                    new_name = '.'.join(k.split('.')[-2:])
                    print(f"So far it should be the strategy_name/early_warning, we will use the last two:{new_name}")
                    pool[new_name] = v
        else:
            k, v = full_key_pool[0]
            pool[key] = v 
    return pool

def dict_to_arglist(d):
    arglist = []
    for k, v in d.items():
        arglist.append('--' + str(k))
        if v is not None:
            if isinstance(v, (list,tuple)):
                for vvv in v:
                    arglist.append(str(vvv))
            else:
                arglist.append(str(v))


        else:
            arglist.pop(-1)
    return arglist