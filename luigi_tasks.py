import requests
import subprocess
import tarfile
from pathlib import Path

import luigi
from luigi.util import inherits, requires


class LocalFilesTarget(luigi.target.Target):
    def __init__(self, paths):
        self.paths = [Path(path) for path in paths]

    
    def exists(self):
        return all(path.is_file() for path in self.paths)


class DownloadStsbenchmark(luigi.Task):
    data_dir = luigi.Parameter(default='./data')


    def requires(self):
        pass


    def output(self):
        path = str(Path(self.data_dir) / 'Stsbenchmark.tar.gz')
        return luigi.LocalTarget(path, format=luigi.format.Nop)


    def run(self):
        url = "http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz"
        data = requests.get(url).content
        with self.output().open('wb') as f:
            f.write(data)


@requires(DownloadStsbenchmark)
class ExtractStsbenchmark(luigi.Task):
    data_dir = luigi.Parameter(default='./data')


    def output(self):
        data_dir = Path(self.data_dir) / 'stsbenchmark/'
        paths = [
            data_dir / 'sts-train.tsv',
            data_dir / 'sts-dev.tsv',
            data_dir / 'sts-test.tsv'
        ]
        return LocalFilesTarget(paths)


    def run(self):
        filenames = [
            'stsbenchmark/sts-train.csv',
            'stsbenchmark/sts-dev.csv',
            'stsbenchmark/sts-test.csv'
        ]
        tar_path = self.input().open().name
        tar = tarfile.open(tar_path, 'r:gz')
        members = [tar.getmember(filename) for filename in filenames]
        
        for member in members:
            member.name = Path(member.name).with_suffix('.tsv')
        
        tar.extractall(self.data_dir, members)


class DownloadPretrainedBert(luigi.Task):
    bert_pretrained_dir = luigi.Parameter(default='./model_pretrained/BERT')


    def requires(self):
        pass

    
    def output(self):
        filenames = [
            'bert_model.ckpt.data-00000-of-00001',
            'bert_model.ckpt.index',
            'bert_model.ckpt.meta',
            'bert_config.json',
            'vocab.txt'
        ]
        bert_dir = Path(self.bert_pretrained_dir)
        paths = [bert_dir / filename for filename in filenames]
        return LocalFilesTarget(paths)


    def run(self):
        raise NotImplementedError()


class DownloadPretrainedBertJa(luigi.Task):
    bert_pretrained_dir = luigi.Parameter(default='./model_pretrained/BERT_ja')


    def requires(self):
        pass

    
    def output(self):
        filenames = [
            'model.ckpt-1400000.data-00000-of-00001',
            'model.ckpt-1400000.index',
            'model.ckpt-1400000.meta',
            'wiki-ja.model',
            'wiki-ja.vocab',
            'graph.pbtxt'
        ]
        bert_dir = Path(self.bert_pretrained_dir)
        paths = [bert_dir / filename for filename in filenames]
        return LocalFilesTarget(paths)


    def run(self):
        raise NotImplementedError()


@inherits(ExtractStsbenchmark)
@inherits(DownloadPretrainedBert)
class FinetuneBertForSts(luigi.Task):
    data_dir = luigi.Parameter(default='./data')
    bert_pretrained_dir = luigi.Parameter(default='./model_pretrained/BERT')
    bert_finetuned_dir = luigi.Parameter(default='./model_finetuned/BERT_STS')


    def requires(self):
        yield self.clone(ExtractStsbenchmark)
        yield self.clone(DownloadPretrainedBert)


    def output(self):
        path = str(Path(self.bert_finetuned_dir) / 'eval_results.txt')
        return luigi.LocalTarget(path)

    
    def run(self):
        cmd = [
            "python",
            "./run_reg.py",
            "--task_name=sts-b",
            "--do_train=false",
            "--do_eval=true",
            f"--data_dir={Path(self.data_dir)}",
            f"--vocab_file={Path(self.bert_pretrained_dir) / 'vocab.txt'}",
            f"--bert_config_file={Path(self.bert_pretrained_dir) / 'bert_config.json'}",
            f"--init_checkpoint={Path(self.bert_finetuned_dir) / 'model.ckpt-28745'}",
            "--max_seq_length=512",
            "--train_batch_size=2",
            "--learning_rate=2e-5",
            "--num_train_epochs=10",
            f"--output_dir={Path(self.bert_finetuned_dir)}"
        ]
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    luigi.run()
