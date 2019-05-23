import yaml
import luigi
import os

import regex2dfa_task


config = yaml.load(open('./pipeline/config.yaml'))

class Node2VecTask(luigi.Task):
    env_name = luigi.Parameter()
    contract = luigi.Parameter()

    def requires(self):
        return regex2dfa_task.Regex2DFATask(
            env_name=self.env_name, contract=self.contract)

    def output(self):
        out_dir = os.path.join(config['embedding']['output_folder'],
                               self.env_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, self.contract + '.node2vec')

        return luigi.LocalTarget(out_path)

    def complete(self):
        parent = config['embedding']['output_folder']
        node2vec = os.path.exists(os.path.join(parent, self.env_name, '{}.node2vec'.format(self.contract)))

        return node2vec


    def run(self):
        cmd = 'python \'{}\' \'{}\' \'{}\''.format(
            config['embedding']['exec'],
            self.input()['node2vec'].path,
            self.output().path)
        print(cmd)
        os.system(cmd)

if __name__ == '__main__':
    luigi.run()
