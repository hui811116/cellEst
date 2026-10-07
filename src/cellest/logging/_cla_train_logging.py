import logging
import json


class TrainLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.propagate = False
        self._handlers: list[logging.Handler] = []
        self.configure_logging()

    def configure_logging(self, args=None):
        verbose = getattr(args, "verbose", False)
        log_path = getattr(args, "log", None)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)

        for handler in self._handlers:
            self.logger.removeHandler(handler)
            handler.close()
        self._handlers.clear()

        handlers: list[logging.Handler] = [logging.StreamHandler()]
        if log_path:
            handlers.append(logging.FileHandler(log_path))
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        for handler in handlers:
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self._handlers.append(handler)

    def log_dataset_summary(self, train_set, validation_set, test_set, class_to_idx, device):
        self.logger.info("Using device: %s", device)
        self.logger.info(
            "Dataset sizes: train=%d validation=%d test=%d",
            len(train_set),
            len(validation_set),
            len(test_set) if test_set is not None else 0,
        )
        self.logger.info("Class-to-index mapping: %s", class_to_idx)

    def log_epoch_summary(self, epoch, total_epochs, train_loss, train_acc, val_loss, val_acc, val_auc):
        self.logger.info(
            "Epoch %d/%d | train loss %.4f acc %.4f | validation loss %.4f acc %.4f AUROC %.4f",
            epoch,
            total_epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_auc,
        )

    def log_run_summary(self, output_dir, summary):
        self.logger.info("Training outputs saved to %s", output_dir)
        self.logger.info("Run summary:\n%s", json.dumps(summary, indent=2))

    def info(self, message, *args, **kwargs):
        self.logger.info(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self.logger.warning(message, *args, **kwargs)


