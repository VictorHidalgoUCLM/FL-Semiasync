class EarlyStoppingTriggered(Exception):
    pass

class EMAEarlyStopping:
    def __init__(self, alpha=0.1, patience=3, min_delta=0.0):
        self.alpha = alpha
        self.patience = patience
        self.min_delta = min_delta

        self.ema = None
        self.best_ema = float('inf')
        self.bad_epochs = 0
        self.epoch = 0
        self.improved = False

        self.end_flag = False

    def update(self, val_loss: float) -> bool:        
        self.epoch += 1
        self.improved = False

        if self.ema is None:
            self.ema = val_loss
        else:
            self.ema = self.alpha * val_loss + (1 - self.alpha) * self.ema

        if self.best_ema - self.ema > self.min_delta:
            self.best_ema = self.ema
            self.bad_epochs = 0
            self.improved = True
        else:
            self.bad_epochs += 1

        return self.bad_epochs >= self.patience
    
    def restart(self):
        self.bad_epochs = 0
        self.end_flag = True