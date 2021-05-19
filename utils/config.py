import os.path as osp
import sys
from argparse import ArgumentParser
from collections import Iterable
from importlib import import_module
from easydict import EasyDict as edict


def add_args(parser, cfg, prefix=''):
    for k, v in cfg.items():
        if isinstance(v, str):
            parser.add_argument('--' + prefix + k)
        elif isinstance(v, int):
            parser.add_argument('--' + prefix + k, type=int)
        elif isinstance(v, float):
            parser.add_argument('--' + prefix + k, type=float)
        elif isinstance(v, bool):
            parser.add_argument('--' + prefix + k, action='store_true')
        elif isinstance(v, dict):
            add_args(parser, v, k + '.')
        elif isinstance(v, Iterable):
            parser.add_argument('--' + prefix + k, type=type(v[0]), nargs='+')
        else:
            print('connot parse key {} of type {}'.format(prefix + k, type(v)))
    return parser


class Config(object):
    @staticmethod
    def from_file(filename):
        if filename.endswith('.py'):
            sys.path.append(osp.dirname(filename))
            module_name = osp.basename(filename).rstrip('.py')
            cfg = import_module(module_name)
            config_dict = edict({
                name: value
                for name, value in cfg.__dict__.items()
                if not name.startswith(('__', '_'))
            })
        else:
            raise IOError('only py type are supported as config files')
        return Config(config_dict, filename=filename)

    @staticmethod
    def auto_argparser(description=None):
        partial_parser = ArgumentParser(description=description)
        partial_parser.add_argument('config', help='config file path')
        cfg_file = partial_parser.parse_known_args()[0].config
        cfg = Config.from_py(cfg_file)
        parser = ArgumentParser(description=description)
        parser.add_argument('config', help='config file path')
        add_args(parser, cfg)
        return parser, cfg

    def __init__(self, config_dict, filename=None):
        assert isinstance(config_dict, dict)
        self._config_dict = config_dict
        self._default_dict = {}
        self.filename = filename
        if filename:
            with open(filename, 'r') as f:
                self._text = f.read()

    def __getattr__(self, key):
        try:
            val = self._config_dict[key]
        except KeyError:
            if key in self._default_dict:
                val = self._default_dict[key]
            else:
                raise
        return val

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __iter__(self):
        return self.keys()

    def __contains__(self, key):
        if key in self._config_dict or key in self._default_dict:
            return True
        else:
            return False

    @property
    def text(self):
        return self._text

    def keys(self):
        for key in self._config_dict:
            yield key
        for key in self._default_dict:
            if key not in self._config_dict:
                yield key

    def values(self):
        for key in self.keys():
            yield self.__getattr__(key)

    def items(self):
        for key in self.keys():
            yield key, self.__getattr__(key)

    def set_default(self, default_dict):
        assert isinstance(default_dict, dict)
        self._default_dict.update(default_dict)

    def update_with_args(self, args):
        for k, v in vars(args).items():
            if v is not None:
                if '.' not in k:
                    self._config_dict[k] = v
                else:
                    tree = k.split('.')
                    tmp = self._config_dict
                    for key in tree[:-1]:
                        tmp = tmp[key]
                    tmp[tree[-1]] = v