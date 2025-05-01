from collections import OrderedDict
from jobs import BaseJob
from toolkit.extension import get_all_extensions_process_dict
from collections import OrderedDict
from datetime import datetime

class BenchmarkJob(BaseJob):
    def __init__(self, config: OrderedDict):
        super().__init__(config)
        self.training_folder = self.get_conf('training_folder', required=True)
        self.is_v2 = self.get_conf('is_v2', False)
        self.log_dir = self.get_conf('log_dir', None)
        self.device = self.get_conf('device', 'cpu')
        self.process_dict = get_all_extensions_process_dict()
        self.load_processes(self.process_dict)

    def run(self):
        super().run()

        print("")
        print(f"Running  {len(self.process)} process{'' if len(self.process) == 1 else 'es'}")

        for process in self.process:
            process.run()
