import os
import sys
import yaml

import luigi


config = yaml.load(open('./pipeline/config.yaml'))

class Regex2DFATask(luigi.Task):
    env_name = luigi.Parameter()
    contract = luigi.Parameter()

    def output(self):
        out_folder = os.path.join(config['dfa']['output_folder'],
                                  self.env_name)
        targets = {
            k: luigi.LocalTarget(
                os.path.join(out_folder, '{}_{}.dot'.format(k, self.contract)))
            for k in config[self.env_name][self.contract].keys()
        }
        return targets

    def requires(self):
        return None

    def complete(self):
        parent = config['dfa']['output_folder']
        regular = os.path.exists(
            os.path.join(
                parent, self.env_name,
                'regular_{}.dot'.format(self.contract)
            )
        )
        node2vec = os.path.exists(
            os.path.join(
                parent, self.env_name,
                'node2vec_{}.dot'.format(self.contract)
            )
        )
        return regular and node2vec

    def run(self):
        outfolder = config['dfa']['output_folder']
        outfolder = os.path.join(outfolder, self.env_name)

        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        for k in config[self.env_name][self.contract].keys():
            cmd = 'java -jar {} \"{}\" {}_{}.dot'.format(
                config['dfa']['exec'], config[self.env_name][self.contract][k],
                os.path.join(outfolder, k), self.contract)

            os.system(cmd)

if __name__ == '__main__':
    luigi.run()
