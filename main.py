import argparse
import concurrent.futures
import json
import logging
import math
import os
import shutil
import socket
import struct
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sklearn.model_selection import train_test_split

from scipy.spatial import distance

import embedding_computation
import shuffler

class Benchmark:
    def __init__(self, name, path, build_str, run_script, repeats, samples):
        self.name = name
        self.path = path
        self.build_str = build_str
        self.run_script = run_script
        self.repeats = repeats
        self.samples = samples

        for sample in samples.values():
            sample.benchmark = self

    def consolidate_experiments_results(self):
        total = Counter()
        results = {}
        for sample in self.samples.values():
            sample.consolidate_experiments_results(total, results)

        return total, results

class Sample:
    def __init__(self, symbol):
        self.symbol = symbol
        self.embedding = None
        self.baseline_size = math.inf
        self.baseline_runtime = Runtime(0.0, 0.0)
        self.benchmark = None
        self.experiments = None

    def consolidate_experiments_results(self, total, results):
        experiments = []
        for experiment in self.experiments:
            experiments.append({
                'pass_list': experiment.pass_list,
                'size': experiment.size,
                'evaluation': experiment.evaluation
            })
            total[experiment.evaluation] += 1

        results[self.symbol] = {
            'embedding': self.embedding,
            'baseline_size': self.baseline_size,
            'baseline_runtime': {'percent': self.baseline_runtime.percent, 'time': self.baseline_runtime.time},
            'experiments': experiments,
        }

class Experiment:
    def __init__(self):
        self.pass_list = None
        self.size = None
        self.runtime = None
        self.evaluation = None

class Runtime:
    def __init__(self, percent, time):
        self.percent = percent
        self.time = time


class Result:
    def __init__(self, size, runtime):
        self.size = size
        self.runtime = runtime

class OptCacheEntry:
    @staticmethod
    def dist(a, b):
        return distance.cosine(a, b)

    def __init__(self, embedding, opts):
        self.embedding = embedding
        self.opts = opts

class OptCache:
    def __init__(self, opts_count):
        self.cache = []
        self.opts_count = opts_count

    def fit(self, samples):
        for _, sample in samples:
            # suitable_experiments = list(filter(lambda experiment: experiment['evaluation'] == 'better', sample.experiments))
            suitable_experiments = sample['experiments']
            suitable_experiments.sort(reverse=True, key=lambda experiment: (sample['baseline_size'] - experiment['size']) / sample['baseline_size'])
            suitable_experiments = suitable_experiments[:self.opts_count]
            self.cache.append(OptCacheEntry(sample['embedding'], list(map(lambda experiment: experiment['pass_list'], suitable_experiments))))

    def predict(self, embedding):
        min_dist, closest_entry = math.inf, self.cache[0]
        for entry in self.cache:
            dist = OptCacheEntry.dist(embedding, entry.embedding)
            if dist < min_dist:
                min_dist, closest_entry = dist, entry
        return closest_entry.opts

class Coordinator:
    @staticmethod
    def truncate_symbol_name(symbol):
        return symbol[:48]

    @staticmethod
    def parse_bench_info(bench_dir):
        lines = [line.strip() for line in Path(bench_dir, "benchmark_info.txt").read_text().splitlines()]

        if "build:" in lines:
            build_str = lines[lines.index("build:") + 1]
        else:
            build_str = ""

        if "bench_repeats:" in lines:
            repeats = int(lines[lines.index("bench_repeats:") + 1])
        else:
            repeats = 1

        run_indices = [i for i, line in enumerate(lines) if line == "run:"]
        run_strs = []
        for i in run_indices:
            run_strs.append(lines[i + 1])
        if not run_strs:
            run_strs = ['', '']
            repeats = max(repeats // 2, 1)
        elif len(run_strs) == 1:
            run_strs.append(run_strs[0])
            repeats = max(repeats // 2, 1)

        run_script = Coordinator.generate_run_script(repeats, run_strs)

        functions_index = lines.index("functions:") + 1
        functions = lines[functions_index:]
        try:
            long_functions_index = lines.index("long_functions:") + 1
            samples = lines[long_functions_index:lines.index("functions:") - 1]
            samples = [sample for sample in samples if sample in functions]
        except ValueError:
            samples = functions
        samples = list(map(Coordinator.truncate_symbol_name, samples))
        samples = {symbol: Sample(symbol) for symbol in samples}

        return Benchmark(bench_dir.name, bench_dir, build_str, run_script, repeats, samples)

    @staticmethod
    def get_sub_dirs(base_dir):
        return [Path(base_dir, subdir) for subdir in os.listdir(base_dir) if os.path.isdir(Path(base_dir, subdir))]

    @staticmethod
    def collect_benchmarks(benchmarks_suite_dir):
        logging.info("Collecting benchmarks")

        benchmarks = {}
        benchmarks_dirs = Coordinator.get_sub_dirs(benchmarks_suite_dir)
        while benchmarks_dirs:
            benchmark_dir = benchmarks_dirs.pop()
            if os.path.exists(os.path.join(benchmark_dir, "benchmark_info.txt")):
                benchmarks[benchmark_dir.name] = Coordinator.parse_bench_info(benchmark_dir)
            elif os.path.exists(os.path.join(benchmark_dir, "benchmark_info.txt.disabled")):
                continue
            else:
                benchmarks_dirs.extend(Coordinator.get_sub_dirs(benchmark_dir))
        return benchmarks

    @staticmethod
    def create_benchmark_working_dir(benchmark, working_dir):
        if os.path.exists(working_dir):
            return
        os.makedirs(working_dir, exist_ok=True)
        shutil.copytree(benchmark.path, working_dir, dirs_exist_ok=True)

    @staticmethod
    def create_benchmark_working_dir_future(thread_pool, benchmark, working_dir):
        return thread_pool.submit(Coordinator.create_benchmark_working_dir, benchmark, working_dir)

    @staticmethod
    def get_plugin_socket_path(working_dir):
        return os.path.join(working_dir, "gcc_plugin.soc")

    @staticmethod
    def get_coordinator_socket_path(working_dir):
        return os.path.join(working_dir, "s")

    @staticmethod
    def create_coordinator_socket(socket_path):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM, 0)
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        sock.bind(socket_path)
        return sock

    @staticmethod
    def generate_pass_list():
        return shuffler.get_shuffled_list(Coordinator.shuffler_lib, 2).copy()

    @staticmethod
    def create_gcc_instance(build_str, working_dir, plugin_socket_path):
        if os.path.exists(plugin_socket_path):
            os.unlink(plugin_socket_path)
        return subprocess.Popen(build_str, shell=True, cwd=working_dir, stdout=sys.stdout, stderr=sys.stdout)

    @staticmethod
    def setup_gcc_communication(working_dir, build_str_fmt, benchmark):
        coordinator_socket_path = Coordinator.get_coordinator_socket_path(working_dir)
        coordinator_socket = Coordinator.create_coordinator_socket(coordinator_socket_path)
        build_str = build_str_fmt.format(socket_path=coordinator_socket_path, build_str=benchmark.build_str)
        plugin_socket_path = Coordinator.get_plugin_socket_path(working_dir)
        gcc = Coordinator.create_gcc_instance(build_str, working_dir, plugin_socket_path)
        while not os.path.exists(plugin_socket_path):
            gcc.poll()
            if gcc.returncode is not None:
                print(f"gcc failed: return code {gcc.returncode}\n", file=sys.stderr)

        return coordinator_socket, plugin_socket_path, gcc

    @staticmethod
    def compile_sample(working_dir, build_str_fmt, sample):
        coordinator_socket, plugin_socket_path, gcc = Coordinator.setup_gcc_communication(working_dir, build_str_fmt, sample.benchmark)

        while gcc.poll() is None:
            try:
                symbol = Coordinator.truncate_symbol_name(coordinator_socket.recv(4096, socket.MSG_DONTWAIT).decode("utf-8"))
                if symbol == sample:
                    coordinator_socket.sendto(("\n".join(sample.pass_list) + "\n").encode("utf-8"), plugin_socket_path)
                else:
                    list_msg = bytes(1)
                    coordinator_socket.sendto(list_msg, plugin_socket_path)
                coordinator_socket.recv(1024 * Coordinator.embed_len_multiplier)
            except BlockingIOError:
                pass

        if gcc.wait() != 0:
            print(f"gcc failed: return code {gcc.returncode}\n", file=sys.stderr)

    @staticmethod
    def compute_embedding(embedding):
        autophase = embedding[:47]
        cfg_len = embedding[47]
        cfg = embedding[48: 48 + cfg_len]
        val_flow = embedding[48 + cfg_len:]

        cfg_embedding = list(embedding_computation.get_flow2vec_embed(cfg, 25))
        val_flow_embedding = list(embedding_computation.get_flow2vec_embed(val_flow, 25))

        return autophase + cfg_embedding + val_flow_embedding

    @staticmethod
    def consolidate_embeddings(benchmark):
        embeddings = {}
        for sample in benchmark.samples.values():
            embeddings[sample.symbol] = sample.embedding
        return embeddings

    @staticmethod
    def compile_baseline_for_runtime(working_dir, benchmark):
        build_str = Coordinator.compile_baseline_for_runtime_build_str_fmt.format(build_str=benchmark.build_str)
        try:
            subprocess.run(build_str, shell=True, check=True, cwd=working_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            logging.error(f"Got error when compiling baseline from {working_dir} working directory")

    @staticmethod
    def generate_symtab(working_dir):
        try:
            subprocess.run("${AARCH_PREFIX}nm --extern-only --defined-only -v --print-file-name pg_main.elf > symtab", shell=True, check=True, cwd=working_dir,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            logging.error(f"Got error when generating symbol table in {working_dir} working directory")

    @staticmethod
    def compile_baseline_for_size_wrapper(coordinator, benchmark):
        coordinator.compile_baseline_for_size(coordinator.get_compile_baseline_for_size_working_dir(benchmark), benchmark)

    @staticmethod
    def compile_baseline_for_runtime_wrapper(coordinator, benchmark):
        working_dir = coordinator.get_compile_baseline_for_runtime_working_dir(benchmark)
        coordinator.compile_baseline_for_runtime(working_dir, benchmark)
        Coordinator.generate_symtab(working_dir)
        try:
            subprocess.run(benchmark.run_script, shell=True, check=True, cwd=working_dir, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            logging.error(f"Got error when running baseline benchmark in {working_dir} working directory")

    @staticmethod
    def compile_sample_for_size_wrapper(coordinator, sample, experiment_no):
        coordinator.compile_sample_for_size(coordinator.get_compile_sample_for_size_working_dir(sample, experiment_no), sample)
        coordinator.dump_sample_pass_list(sample, experiment_no)

    @staticmethod
    def compile_sample_for_runtime_wrapper(coordinator, sample, experiment_no):
        working_dir = coordinator.get_compile_sample_for_runtime_working_dir(sample, experiment_no)
        coordinator.compile_sample_for_runtime(working_dir, sample)
        Coordinator.generate_symtab(working_dir)
        try:
            subprocess.run(sample.benchmark.run_script, shell=True, check=True, cwd=working_dir, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            logging.error(f"Got error when running sample benchmark in {working_dir} working directory")

    @staticmethod
    def generate_run_script(repeats, run_strs):
        sum_exists = False
        run_script = f"for i in {{1..{repeats}}}; do "
        assert (len(run_strs) >= 2)
        for run_str in run_strs:
            run_script += f"qemu-aarch64 -L /usr/aarch64-linux-gnu ./pg_main.elf {run_str}; "
            run_script += f"${{AARCH_PREFIX}}gprof -s -Ssymtab pg_main.elf gmon.out{' gmon.sum; ' if sum_exists else '; '}"
            sum_exists = True
        run_script += "done"
        return run_script

    @staticmethod
    def get_runtime_data(working_dir):
        try:
            return subprocess.run(f"${{AARCH_PREFIX}}gprof -bp --no-demangle pg_main.elf gmon.sum", shell=True, check=True, capture_output=True, cwd=working_dir).stdout.decode("utf-8").splitlines()
        except subprocess.CalledProcessError:
            logging.error(f"Got error when collecting runtime data from {working_dir} working directory")
            return [" no time accumulated"]

    @staticmethod
    def evaluate_experiment(result, baseline):
        if result.size > baseline.size or result.size == math.inf:
            return 'worse'

        if result.runtime.percent < 1 and baseline.runtime.percent < 1:
            is_runtime_worse = False
        elif result.runtime.time <= baseline.runtime.time:
            is_runtime_worse = False
        elif baseline.runtime.time == 0:
            is_runtime_worse = True
        elif (result.runtime.time - baseline.runtime.time) / baseline.runtime.time < 0.05:
            is_runtime_worse = False
        else:
            is_runtime_worse = True

        if is_runtime_worse:
            return 'worse'

        if result.size == baseline.size:
            return 'equal'
        else:
            return 'better'

    @staticmethod
    def process_experiment(sample, experiment_no):
        experiment = Result(sample.experiments[experiment_no].size, sample.experiments[experiment_no].runtime)
        baseline = Result(sample.baseline_size, sample.baseline_runtime)
        sample.experiments[experiment_no].evaluation = Coordinator.evaluate_experiment(experiment, baseline)

    shuffler_lib = shuffler.setuplib("./shuffler/libactions.so")

    embed_len_multiplier = 200

    baseline_size_build_str_fmt_wo_plugin_path = (
        "custom-gcc -fplugin={plugin_path} -O2 -fplugin-arg-plugin-dyn_replace=learning "
        "-fplugin-arg-plugin-remote_socket={{socket_path}} {{build_str}} -o main.elf"
    )
    compile_baseline_for_runtime_build_str_fmt = "custom-gcc -O2 {build_str} -pg -o pg_main.elf"

    sample_runtime_build_str_fmt_wo_plugin_path = (
        "custom-gcc -fplugin={plugin_path} -O2 -fplugin-arg-plugin-dyn_replace=learning "
        "-fplugin-arg-plugin-remote_socket={{socket_path}} -pg {{build_str}} -o pg_main.elf"
    )

    embeddings_file_name = "embeddings.json"
    pass_list_file_name = "pass_list.json"

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-w", "--working-dir", dest="working_dir", action="store", type=str, required=True)
        parser.add_argument("-b", "--benchmarks", dest="benchmarks_suite_path", action="store", required=True)
        parser.add_argument("-p", "--plugin", dest="plugin_path", action="store", required=True)
        parser.add_argument("-e", "--experiments", dest="experiments_count", action="store", type=int, default=1)
        args = parser.parse_args()

        self.working_dir = args.working_dir
        self.experiments_dir = os.path.join(self.working_dir, "e")

        self.experiments_count = args.experiments_count

        self.benchmarks = Coordinator.collect_benchmarks(args.benchmarks_suite_path)

        self.baseline_size_build_str_fmt = Coordinator.baseline_size_build_str_fmt_wo_plugin_path.format(plugin_path=args.plugin_path)
        self.sample_size_build_str_fmt = self.baseline_size_build_str_fmt
        self.sample_runtime_build_str_fmt = Coordinator.sample_runtime_build_str_fmt_wo_plugin_path.format(plugin_path=args.plugin_path)

        self.experiments_results = None
        self.opt_cache = None

    def get_baseline_working_dir(self, benchmark):
        return os.path.join(self.experiments_dir, benchmark.name, "baseline")

    def get_compile_baseline_for_size_working_dir(self, benchmark):
        return os.path.join(self.get_baseline_working_dir(benchmark), "size")

    def get_compile_baseline_for_runtime_working_dir(self, benchmark):
        return os.path.join(self.get_baseline_working_dir(benchmark), "runtime")

    def get_sample_working_dir(self, sample, experiment_no):
        return os.path.join(self.experiments_dir, sample.benchmark.name, sample.symbol, experiment_no)

    def get_compile_sample_for_size_working_dir(self, sample, experiment_no):
        return os.path.join(self.get_sample_working_dir(sample, experiment_no), "size")

    def get_compile_sample_for_runtime_working_dir(self, sample, experiment_no):
        return os.path.join(self.get_sample_working_dir(sample, experiment_no), "runtime")

    def make_benchmark_working_dirs(self, benchmarks, experiments_count):
        thread_pool = ThreadPoolExecutor(os.cpu_count())

        futures = []
        logging.info(f"Creating benchmark working directories")
        for benchmark in benchmarks.values():
            logging.info(f"Creating '{benchmark.name}' benchmark working directories")
            futures.append(Coordinator.create_benchmark_working_dir_future(thread_pool, benchmark, self.get_compile_baseline_for_size_working_dir(benchmark)))
            futures.append(Coordinator.create_benchmark_working_dir_future(thread_pool, benchmark, self.get_compile_baseline_for_runtime_working_dir(benchmark)))

            for sample in benchmark.samples.values():
                logging.info(f"Creating '{benchmark.name}' benchmark '{sample.symbol}' sample working directories")
                for experiment_no in range(experiments_count):
                    futures.append(Coordinator.create_benchmark_working_dir_future(thread_pool, benchmark, self.get_compile_sample_for_size_working_dir(sample, str(experiment_no))))
                    futures.append(Coordinator.create_benchmark_working_dir_future(thread_pool, benchmark, self.get_compile_sample_for_runtime_working_dir(sample, str(experiment_no))))

        logging.info(f"Waiting for {len(futures)} benchmark working directory creation tasks to complete")
        completed_futures_counter = 1
        for future in concurrent.futures.as_completed(futures):
            if completed_futures_counter % 20 == 0:
                logging.info(f"{completed_futures_counter} benchmark working directory creation tasks completed out of {len(futures)}")
            completed_futures_counter += 1

            try:
                future.result()
            except Exception as exc:
                logging.info(f"Task generated an exception: {exc}")

        thread_pool.shutdown()

    def dump_benchmark_embeddings(self, benchmark):
        with open(os.path.join(self.get_compile_baseline_for_size_working_dir(benchmark), Coordinator.embeddings_file_name), "w") as embeddings_file:
            json.dump(Coordinator.consolidate_embeddings(benchmark), embeddings_file)

    def compile_baseline_for_size(self, working_dir, benchmark):
        coordinator_socket, plugin_socket_path, gcc = Coordinator.setup_gcc_communication(working_dir, self.baseline_size_build_str_fmt, benchmark)

        while gcc.poll() is None:
            try:
                symbol = Coordinator.truncate_symbol_name(coordinator_socket.recv(4096, socket.MSG_DONTWAIT).decode("utf-8"))
                list_msg = bytes(1)
                coordinator_socket.sendto(list_msg, plugin_socket_path)
                embedding_msg = coordinator_socket.recv(1024 * Coordinator.embed_len_multiplier)
                if symbol in benchmark.samples:
                    benchmark.samples[symbol].embedding = Coordinator.compute_embedding([x[0] for x in struct.iter_unpack("i", embedding_msg)])
            except BlockingIOError:
                pass

        if gcc.wait() != 0:
            print(f"gcc failed: return code {gcc.returncode}\n", file=sys.stderr)

        self.dump_benchmark_embeddings(benchmark)

    def dump_sample_pass_list(self, sample, experiment_no):
        with open(os.path.join(self.get_sample_working_dir(sample, experiment_no), Coordinator.pass_list_file_name), "w") as pass_list_file:
            json.dump({'pass_list': sample.pass_list}, pass_list_file)

    def compile_sample_for_size(self, working_dir, sample):
        Coordinator.compile_sample(working_dir, self.sample_size_build_str_fmt, sample)

    def compile_sample_for_runtime(self, working_dir, sample):
        Coordinator.compile_sample(working_dir, self.sample_runtime_build_str_fmt, sample)

    def create_compile_baseline_for_size_future(self, thread_pool, benchmark):
        return thread_pool.submit(Coordinator.compile_baseline_for_size_wrapper, self, benchmark)

    def create_compile_baseline_for_runtime_future(self, thread_pool, benchmark):
        return thread_pool.submit(Coordinator.compile_baseline_for_runtime_wrapper, self, benchmark)

    def create_compile_sample_for_size_future(self, thread_pool, sample, experiment_no):
        return thread_pool.submit(Coordinator.compile_sample_for_size_wrapper, self, sample, experiment_no)

    def create_compile_sample_for_runtime_future(self, thread_pool, sample, experiment_no):
        return thread_pool.submit(Coordinator.compile_sample_for_runtime_wrapper, self, sample, experiment_no)

    def run_experiment(self, thread_pool, futures, sample, experiment_no):
        futures.append(self.create_compile_sample_for_size_future(thread_pool, sample, experiment_no))
        futures.append(self.create_compile_sample_for_runtime_future(thread_pool, sample, experiment_no))

    def run_sample(self, thread_pool, futures, sample):
        logging.info(f"Running '{sample.benchmark.name}' benchmark '{sample.symbol}' sample experiments")

        for experiment_no in range(self.experiments_count):
            sample.pass_list = Coordinator.generate_pass_list()
            self.run_experiment(thread_pool, futures, sample, str(experiment_no))

    def run_benchmark(self, thread_pool, futures, benchmark):
        logging.info(f"Running '{benchmark.name}' benchmark experiments")

        futures.append(self.create_compile_baseline_for_size_future(thread_pool, benchmark))
        futures.append(self.create_compile_baseline_for_runtime_future(thread_pool, benchmark))

        for sample in benchmark.samples.values():
            self.run_sample(thread_pool, futures, sample)

    def run_experiments(self):
        logging.info("Running experiments")

        self.make_benchmark_working_dirs(self.benchmarks, self.experiments_count)

        thread_pool = ThreadPoolExecutor(os.cpu_count())
        futures = []
        for benchmark in self.benchmarks.values():
            self.run_benchmark(thread_pool, futures, benchmark)

        logging.info(f"Waiting for {len(futures)} benchmark processing tasks to complete")
        completed_futures_counter = 1
        for future in concurrent.futures.as_completed(futures):
            if completed_futures_counter % 10 == 0:
                logging.info(f"{completed_futures_counter} benchmark processing tasks completed out of {len(futures)}")
            completed_futures_counter += 1
            try:
                future.result()
            except Exception as exc:
                logging.info(f"Task generated an exception: {exc}")

        thread_pool.shutdown()

    def get_sample_size(self, sample, experiment_no):
        working_dir = self.get_compile_sample_for_size_working_dir(sample, str(experiment_no))
        try:
            size = int(subprocess.run(f"${{AARCH_PREFIX}}nm --print-size --size-sort --radix=d main.elf | grep {sample.symbol}",
                                      shell=True, check=True, capture_output=True, cwd=working_dir).stdout.decode("utf-8").splitlines()[0].split()[1])
        except subprocess.CalledProcessError:
            logging.error(f"Got error when collecting sample size from {working_dir} working directory")
            size = math.inf
        sample.experiments[experiment_no].size = size

    def get_baseline_sizes(self, benchmark):
        working_dir = self.get_compile_baseline_for_size_working_dir(benchmark)
        try:
            size_info = subprocess.run("${AARCH_PREFIX}nm --print-size --size-sort --radix=d main.elf", shell=True, check=True, capture_output=True,
                                       cwd=working_dir).stdout.decode("utf-8").splitlines()
        except subprocess.CalledProcessError:
            logging.error(f"Got error when collecting baseline sizes from {working_dir} working directory")
            size_info = []
        for line in size_info:
            pieces = line.split()
            symbol = Coordinator.truncate_symbol_name(pieces[3])
            if symbol in benchmark.samples:
                benchmark.samples[symbol].baseline_size = int(pieces[1])

    def get_baseline_runtimes(self, benchmark):
        runtime_data = Coordinator.get_runtime_data(self.get_compile_baseline_for_runtime_working_dir(benchmark))

        if " no time accumulated" in runtime_data:
            logging.warning(f"no time accumulated for baseline of {benchmark.name} benchmark")
        else:
            runtime_data = runtime_data[5:]
            for line in runtime_data:
                pieces = line.split()
                symbol = Coordinator.truncate_symbol_name(pieces[-1])
                if symbol in benchmark.samples:
                    benchmark.samples[symbol].baseline_runtime = Runtime(float(pieces[0]), float(pieces[2]))

    def get_sample_runtime(self, sample, experiment_no):
        runtime_data = Coordinator.get_runtime_data(self.get_compile_sample_for_runtime_working_dir(sample, str(experiment_no)))

        runtime = Runtime(0.0, 0.0)
        if " no time accumulated" in runtime_data:
            logging.warning(f"no time accumulated for '{sample.symbol}' sample of '{sample.benchmark.name}' benchmark")
        else:
            runtime_data = runtime_data[5:]
            for line in runtime_data:
                pieces = line.split()
                symbol = Coordinator.truncate_symbol_name(pieces[-1])
                if symbol == sample.symbol:
                    runtime = Runtime(float(pieces[0]), float(pieces[2]))
                    break
        sample.experiments[experiment_no].runtime = runtime

    def get_sample_pass_list(self, sample, experiment_no):
        with open(os.path.join(self.get_sample_working_dir(sample, str(experiment_no)), Coordinator.pass_list_file_name), "r") as pass_list_file:
            sample.experiments[experiment_no].pass_list = json.load(pass_list_file)['pass_list']

    def consolidate_experiment_results(self):
        benchmarks = {}
        total = Counter()
        for benchmark in self.benchmarks.values():
            total, benchmark_results = benchmark.consolidate_experiments_results()
            benchmarks[benchmark.name] = benchmark_results
            for evaluation, count in total.items():
                total[evaluation] += count

        self.experiments_results = {'benchmarks': benchmarks, 'total': total}

    def collect_experiment_result(self, sample, experiment_no):
        self.get_sample_size(sample, experiment_no)
        self.get_sample_runtime(sample, experiment_no)
        self.get_sample_pass_list(sample, experiment_no)
        Coordinator.process_experiment(sample, experiment_no)

    def collect_sample_results(self, sample):
        logging.info(f"Collecting results for '{sample.symbol}' sample of '{sample.benchmark.name}'")

        sample.experiments = [Experiment()] * self.experiments_count
        for experiment_no in range(self.experiments_count):
            self.collect_experiment_result(sample, experiment_no)

    def collect_benchmark_results(self, benchmark):
        logging.info(f"Collecting results for {benchmark.name} baseline")

        self.get_baseline_sizes(benchmark)
        self.get_baseline_runtimes(benchmark)

        for sample in benchmark.samples.values():
            self.collect_sample_results(sample)

    def load_benchmark_embeddings(self, benchmark):
        with open(os.path.join(self.get_compile_baseline_for_size_working_dir(benchmark), Coordinator.embeddings_file_name), "r") as embeddings_file:
            for sample, embedding in json.load(embeddings_file).items():
                benchmark.samples[sample].embedding = embedding

    def collect_experiments_results(self, results_file_name):
        logging.info("Collecting results")

        for benchmark in self.benchmarks.values():
            self.collect_benchmark_results(benchmark)
            self.load_benchmark_embeddings(benchmark)

        self.consolidate_experiment_results()

        with open(os.path.join(self.working_dir, results_file_name), "w") as results_file:
            json.dump(self.experiments_results, results_file)

    def load_experiments_results(self, results_file_name):
        with open(os.path.join(self.working_dir, results_file_name), "r") as results_file:
            self.experiments_results = json.load(results_file)

    def collect_samples_from_experiments_results(self, results_file_name):
        self.load_experiments_results(results_file_name)

        samples = []
        for benchmark, benchmark_result in self.experiments_results['benchmarks'].items():
            for sample, sample_result in benchmark_result.items():
                samples.append((sample, {'benchmark': self.benchmarks[benchmark], **sample_result}))

        return samples

    def evaluate_opt_cache(self, experiments_results_file_name, opts_count):
        train, test = train_test_split(self.collect_samples_from_experiments_results(experiments_results_file_name), test_size=0.5)

        with open(os.path.join(self.working_dir, "test_samples.json"), "w") as test_samples_file:
            json.dump({'test_samples': [symbol for symbol, _ in test]}, test_samples_file)

        opt_cache = OptCache(opts_count)
        opt_cache.fit(train)

        benchmarks = {sample['benchmark'].name: sample['benchmark'] for _, sample in test}
        self.make_benchmark_working_dirs(benchmarks, opts_count)

        thread_pool = ThreadPoolExecutor(os.cpu_count())
        futures = []
        test_samples = {}
        for symbol, test_sample in test:
            logging.info(f"Testing '{test_sample['benchmark'].name}' benchmark '{symbol}' sample prediction")

            opts = opt_cache.predict(test_sample['embedding'])

            sample = self.benchmarks[test_sample['benchmark'].name].samples[symbol]
            sample.baseline_size = test_sample['baseline_size']
            sample.baseline_runtime = Runtime(test_sample['baseline_runtime']['percent'], test_sample['baseline_runtime']['time'])

            test_samples[symbol] = sample
            for experiment_no in range(len(opts)):
                sample.pass_list = opts[experiment_no]
                self.run_experiment(thread_pool, futures, sample, str(experiment_no))

        logging.info(f"Waiting for {len(futures)} sample prediction testing tasks to complete")
        completed_futures_counter = 1
        for future in concurrent.futures.as_completed(futures):
            if completed_futures_counter % 10 == 0:
                logging.info(f"{completed_futures_counter} sample prediction testing tasks completed out of {len(futures)}")
            completed_futures_counter += 1

            try:
                future.result()
            except Exception as exc:
                logging.info(f"Task generated an exception: {exc}")

        thread_pool.shutdown()

        total = Counter()
        results = {}
        for sample in test_samples.values():
            self.collect_sample_results(sample)
            sample.consolidate_experiments_results(total, results)

        with open(os.path.join(self.working_dir, "opt_cache_results.json"), "w") as results_file:
            json.dump({'samples': results, 'total': total}, results_file)


if __name__ == "__main__":
    logging.basicConfig(force=True, level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    Coordinator().run_experiments()
