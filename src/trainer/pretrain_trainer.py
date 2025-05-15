import torch
import os


class Trainer:
    def __init__(self, args, optimizer, lr_scheduler, gmc_loss_fn, summary_writer, device, model_name):
        self.args = args
        self.model_name = model_name
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.gmc_loss_fn = gmc_loss_fn
        self.summary_writer = summary_writer
        self.device = device

    def _forward_epoch(self, model, batched_data):
        (idx, direction, x_tabular, x_img) = batched_data
        x_tabular = x_tabular.to(self.device)
        x_img = x_img.to(self.device)
        batch_repr = model.forward_unsupervised(x_tabular, x_img)
        return batch_repr

    def train_epoch(self, model, train_loader, epoch_idx):
        model.train()
        for batch_idx, batched_data in enumerate(train_loader):
            self.optimizer.zero_grad()
            batch_repr = self._forward_epoch(model, batched_data)
            joint_mod_loss_sum = self.gmc_loss_fn(batch_repr, temperature=0.1, batch_size=32)
            loss = torch.mean(joint_mod_loss_sum)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()
            self.lr_scheduler.step()
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('Loss/train', loss, (epoch_idx - 1) * len(train_loader) + batch_idx + 1)

    def fit(self, model, train_loader, test_loader):
        test_result, train_result = None, None
        for epoch in range(1, self.args.n_epochs + 1):
            self.train_epoch(model, train_loader, epoch)
            test_result = self.eval(model, test_loader)
            train_result = self.eval(model, train_loader)
            print(f"Epoch: {epoch}\ttrain loss: {train_result}")

        if self.args.save:
            os.makedirs(self.args.save_path, exist_ok=True)
            torch.save(model.state_dict(), self.args.save_path+f"/{self.args.config_sgpt}.pth")

        return train_result, test_result

    def eval(self, model, dataloader):
        model.eval()
        loss_all = []
        with torch.no_grad():
            for batched_data in dataloader:
                batch_repr = self._forward_epoch(model, batched_data)
                batch_size = batch_repr[0].size()[0]
                joint_mod_loss_sum = self.gmc_loss_fn(batch_repr, temperature=0.1, batch_size=batch_size)  # 0.1
                loss = torch.mean(joint_mod_loss_sum)
                loss_all.append(loss.item())

        return sum(loss_all) / len(loss_all)
