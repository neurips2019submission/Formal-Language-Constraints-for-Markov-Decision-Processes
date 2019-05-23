import luigi

import atari
import pipeline.node2vec_task as node2vec_task

class TestTask(luigi.Task):
    env_name = luigi.Parameter('BreakoutDeterministic-v4', False, True)
    contract = luigi.Parameter('{lr}(2)', False, True)
    steps = luigi.IntParameter(100000, False, True)
    architecture = luigi.Parameter('contract', False, True)
    contract_mode = luigi.Parameter('punish', False, True)
    train_seed = luigi.IntParameter(123, False, True)
    test_seed = luigi.IntParameter(321, False, True)
    enforce_contract = luigi.BoolParameter(False, False, True)
    doom_scenario = luigi.Parameter(None, False, True)
    log_root = luigi.Parameter('.', False, True)


    def requires(self):
#        weights = luigi.LocalTarget(
#            test.filename_root_fn(self.env_name, self.contract, self.steps,
#                                   self.architecture, self.contract_mode,
#                                   self.train_seed) + '_weights.h5f')
        node2vec = node2vec_task.Node2VecTask(
            env_name=self.env_name, contract=self.contract)
        return node2vec

    def output(self):
        return None
        # # TODO: the following code neither works nor affects the results
        # names = ['{}_{}_history_{}'.format(self.log_root, n, self.steps) for n in ['action', 'reward', 'violation', 'done']]
        # npys = [luigi.LocalTarget(
        #     atari.filename_prefix_fn(self.env_name, self.contract, self.steps,
        #                            self.architecture, self.contract_mode,
        #                            self.train_seed, self.test_seed) + '_' + x + '.npy') for x in names]
        # return npys

    def run(self):
        atari.test(
            self.env_name,
            self.contract,
            self.architecture,
            self.contract_mode,
            self.steps,
            self.train_seed,
            self.test_seed,
            self.input().path,
            self.enforce_contract,
            self.doom_scenario)