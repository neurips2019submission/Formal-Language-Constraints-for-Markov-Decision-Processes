import luigi

import train_task

class TrainSeedTask(luigi.Task):
    env_name = luigi.Parameter('BreakoutDeterministic-v4', False, True)
    contract = luigi.Parameter('{lr}(2)', False, True)
    steps = luigi.IntParameter(100000, False, True)
    contract_mode = luigi.Parameter('punish', False, True)
    seed = luigi.IntParameter(123, False, True)

    def requires(self):
        architectures = ['contract', 'contract_action_history', 'contract_dfa_state']
        return {arch: train_task.TrainTask(env_name=self.env_name, contract=self.contract, steps=self.steps, architecture=arch, contract_mode=self.contract_mode, seed=self.seed) for arch in architectures}

    def output(self):
        return self.input()

