import torch
from tqdm import tqdm


class Logger:
    def __init__(self, metrics):
        self.metrics = {name: [0] for name in metrics}

    def add(self, name):
        self.metrics[name] = [0]

    def update(self, name, value):
        self.metrics[name].append(value)

    def get(self, name):
        return self.metrics[name]

    def print_last(self):
        for name in self.metrics:
            print(f"{name} : {self.metrics[name][-1]}")

    def reset_all(self):
        for m in self.metrics:
            self.metrics[m] = [0]

    def reset(self, m):
        self.metrics[m] = [0]

    def avg(self, name, size):
        return sum(self.metrics[name][-size:]) / (len(self.metrics[name][-size:]))


class LooperCls:
    def __init__(self, device):
        self.device = device
        self.logger = Logger(["loss"])
        self.logger.add("num_labels")

    def train_step(self, model, optimizer, loss_fn, train_set):

        model.train()
        for batch in train_set:
            optimizer.zero_grad()
            data, labels = batch
            data = data.to(self.device)
            labels = labels.to(self.device)
            output = model(data)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            self.logger.update("loss", loss.item())

    def make_pseudo_labels(self, strong_preds, true_labels, th, unk):
        strong_preds = torch.nn.functional.softmax(strong_preds, dim=1)
        strong_preds, pseudo_labels = strong_preds.max(dim=1)
        non_zero_labels = torch.where(
            strong_preds > torch.tensor(th), torch.tensor(1), torch.tensor(0)
        )
        pseudo_labels = torch.where(
            non_zero_labels == 1, pseudo_labels, torch.tensor(0)
        )
        pseudo_labels[true_labels != unk] = true_labels[true_labels != unk]
        non_zero_labels = torch.where(
            pseudo_labels == 0, torch.tensor(0), torch.tensor(1)
        )
        return pseudo_labels, non_zero_labels

    def train_step_self(self, model, optimizer, loss_fn, train_set, th, unk=777):

        model.train()
        for batch in train_set:
            optimizer.zero_grad()

            weak_aug, strong_aug, true_labels = batch
            weak_aug = weak_aug.to(self.device)
            strong_aug = strong_aug.to(self.device)

            strong_preds = model(weak_aug).detach()
            pseudo_labels, non_zero_labels = self.make_pseudo_labels(
                strong_preds.cpu(), true_labels.cpu(), th, unk
            )
            weak_preds = model(strong_aug)
            pseudo_labels = pseudo_labels.to(self.device)
            loss = loss_fn(weak_preds, pseudo_labels)
            loss = (non_zero_labels.to(self.device) * loss).mean()
            loss.backward()
            optimizer.step()
            self.logger.update("num_labels", sum(non_zero_labels == 1).item())
            self.logger.update("loss", loss.item())

    def val_step(self, model, val_funcs, val_set):

        model.eval()
        with torch.no_grad():
            for batch in val_set:
                data, labels = batch
                data = data.to(self.device)
                labels = labels.to(self.device)
                B = labels.shape[0]
                output = model(data)
                for val_fn in val_funcs:
                    name = val_fn.__name__
                    value = val_fn(output, labels)
                    if name in self.logger.metrics:
                        self.logger.update(name, value)
                    else:
                        self.logger.add(name)
                        self.logger.update(name, value)
        return B

    def train(
        self,
        model,
        epochs,
        train_set,
        val_set,
        optimizer,
        loss_fn,
        val_funcs,
        semi_self=False,
        th=None,
    ):

        model.to(self.device)
        pbar = tqdm(total=epochs)
        for i in range(epochs):
            B = self.val_step(model, val_funcs, val_set)
            if semi_self:
                self.train_step_self(model, optimizer, loss_fn, train_set, th)
            else:
                self.train_step(model, optimizer, loss_fn, train_set)

            metrics = {m: self.logger.avg(m, B) for m in self.logger.metrics}
            pbar.set_postfix(metrics)
            pbar.update(1)

    def history(self):
        return self.logger.metrics
