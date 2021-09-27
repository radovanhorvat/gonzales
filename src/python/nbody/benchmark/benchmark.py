import platform
import psutil
import timeit
import logging
import numpy as np
import functools
import json

from nbody.simulator.space import Space
import nbody.lib.brute_force as bf
import nbody.lib.octree as oct


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)


def vel_func(pos_vec):
    return np.array((0., 0., 0.))


def mass_func(pos_vec):
    return 1.0


class BenchmarkConfigBase:
    def __init__(self, particle_nums, num_exec=5):
        self.particle_nums = particle_nums
        self.num_exec = num_exec


class PPBenchmarkConfig(BenchmarkConfigBase):
    """
        Benchmark configuration for brute force acceleration calculation
    """
    def do_timing(self, space):
        t = timeit.Timer(functools.partial(bf.calculate_accs_pp, space.r, space.m, 1.0, 0.))
        res = t.timeit(self.num_exec) / self.num_exec
        return res


class BHBenchmarkConfig(BenchmarkConfigBase):
    """
        Benchmark configuration for Barnes-hut acceleration calculation
    """
    def __init__(self, particle_nums, theta=0.75, num_exec=5):
        super().__init__(particle_nums, num_exec)
        self.theta = theta

    def do_timing(self, space):
        t = timeit.Timer(functools.partial(oct.calc_accs_octree, 1., 0., 0., 0., space.r, space.m, 1.0, 0., self.theta))
        res = t.timeit(self.num_exec) / self.num_exec
        return res


class BenchmarkSuite:
    def __init__(self, output_filename=None):
        self.output_filename = output_filename if output_filename is not None else 'benchmark_results.json'
        self.benchmarks = {}
        self._results = {}

    @staticmethod
    def _get_system_info():
        return {'platform': platform.system(),
                'platform_release': platform.release(),
                'platform_version': platform.version(),
                'processor': platform.processor(),
                'cores': psutil.cpu_count(),
                'memory': round(psutil.virtual_memory().total / (1024.0**3))}

    def add_benchmark(self, bench_name, bench_config):
        self.benchmarks[bench_name] = bench_config

    @staticmethod
    def generate_space(num):
        space = Space()
        space.add_cuboid(num, np.array((0., 0., 0.)), 1., 1., 1., vel_func, mass_func)
        return space

    def run(self):
        logging.info('Gathering system information')
        self._results['sys_info'] = self._get_system_info()
        self._results['benchmarks'] = dict()
        logging.info('Start benchmark suite')
        for bench_name, bench_config in self.benchmarks.items():
            self._results['benchmarks'][bench_name] = dict()
            for num in bench_config.particle_nums:
                space = self.generate_space(num)
                logging.info('Running benchmark: {}, N={}'.format(bench_name, num))
                self._results['benchmarks'][bench_name][num] = bench_config.do_timing(space)
        logging.info('Dumping results to json')
        self._dump_to_json()
        logging.info('End benchmark suite')

    def _dump_to_json(self):
        with open(self.output_filename, 'w') as fp:
            json.dump(self._results, fp)


def run_default_benchmark(output_filename='benchmark_results.json'):
    bf_config = PPBenchmarkConfig((1000, 10000, 20000, 30000))
    bh_config = BHBenchmarkConfig((10000, 100000, 500000, 1000000))
    suite = BenchmarkSuite(output_filename=output_filename)
    suite.add_benchmark('brute_force', bf_config)
    suite.add_benchmark('barnes_hut', bh_config)
    suite.run()
