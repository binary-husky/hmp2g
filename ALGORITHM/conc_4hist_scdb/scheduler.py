import numpy as np
# code changed from site-packages/torch/optim/lr_scheduler.py
class FeedBackPolicyResonance:
    def __init__(self, mcv) -> None:
        self.last_epoch = 0
        self.best = -np.inf
        self.num_bad_epochs = 0
        self.patience = 50
        self.cooldown_counter = 0
        self.cooldown = 50
        self.threshold = 0.01
        self.recommanded_yita = 0
        self.recommanded_yita_max_limit = 0.75
        self.recommanded_yita_inc_unit = 0.05
        self.mcv = mcv

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        # when win rate tooo low, be infinitely patient
        if metrics > 0:
            epoch = self.last_epoch + 1
            self.last_epoch = epoch

            if self.is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.in_cooldown:
                self.cooldown_counter -= 1
                self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

            if self.num_bad_epochs > self.patience:
                self._do_some_adjustment(epoch)
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0

        self.mcv.rec(self.cooldown_counter, 'cooldown_counter')
        self.mcv.rec(self.num_bad_epochs, 'num_bad_epochs')

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _do_some_adjustment(self, epoch):
        self.recommanded_yita += self.recommanded_yita_inc_unit
        if self.recommanded_yita > self.recommanded_yita_max_limit:
            self.recommanded_yita = self.recommanded_yita_max_limit

    def is_better(self, a, best):
        return a > best + self.threshold
