import sys

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig

from neuralop.models import GINO
from neuralop.training import AdamW
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss
import torch


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        sample['x'] = sample['x'].reshape(1, -1).permute(1, 0)
        sample['y'] = sample['y'].reshape(1, -1).permute(1, 0)
        return sample


if __name__ == '__main__':

    pipe = ConfigPipeline(
        [
            YamlConfig(
                'gino_darcy_1.yaml', config_name='local', config_folder='../../network/configs/'
            ),
            ArgparseConfig(infer_types=True, config_name=None, config_file=None),
            YamlConfig(config_folder='../../network/configs/'),
        ]
    )
    params = pipe.read_conf()
    device = 'cuda' if params.opt.cuda is True else 'cpu'

    train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000, batch_size=5,
        test_resolutions=[16, 32], n_tests=[100, 50],
        test_batch_sizes=[32, 32],
    )

    data_processor = data_processor.to(device)

    base_dataset = train_loader.dataset
    input_geom = torch.randn(16 * 16, 2)
    latent_queries = torch.rand(64, 64, 2)
    output_queries = torch.rand(16 * 16, 2)

    augmented_dataset = AugmentedDataset(base_dataset)
    augmented_loader = torch.utils.data.DataLoader(
        augmented_dataset,
        batch_size=train_loader.batch_size,
        shuffle=train_loader.sampler is not None,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory
    )

    base_dataset = test_loaders[16].dataset
    input_geom = torch.randn(16 * 16, 2)
    latent_queries = torch.rand(64, 64, 2)
    output_queries = torch.rand(16 * 16, 2)
    augmented_dataset_test = AugmentedDataset(base_dataset)

    augmented_loader_test16 = torch.utils.data.DataLoader(
        augmented_dataset_test,
        batch_size=test_loaders[16].batch_size,
        shuffle=test_loaders[16].sampler is not None,  # 保持与原始 DataLoader 一致
        num_workers=test_loaders[16].num_workers,
        pin_memory=test_loaders[16].pin_memory
    )

    base_dataset = test_loaders[32].dataset  # 原始 Dataset 对象
    input_geom = torch.randn(32 * 32, 2)  # 示例输入
    latent_queries = torch.rand(64, 64, 2)  # 示例潜在查询
    output_queries = torch.rand(32 * 32, 2)  # 示例输出查询
    augmented_dataset_test = AugmentedDataset(base_dataset)

    augmented_loader_test32 = torch.utils.data.DataLoader(
        augmented_dataset_test,
        batch_size=test_loaders[32].batch_size,
        shuffle=test_loaders[32].sampler is not None,  # 保持与原始 DataLoader 一致
        num_workers=test_loaders[32].num_workers,
        pin_memory=test_loaders[32].pin_memory
    )

    test_loaders[16] = augmented_loader_test16
    test_loaders[32] = augmented_loader_test32

    model = GINO(in_channels=params.gino.in_channels,
                 out_channels=params.gino.out_channels,
                 projection_channels=params.gino.projection_channels,
                 gno_coord_dim=params.gino.gno_coord_dim,
                 gno_radius=params.gino.gno_radius,
                 fno_n_modes=params.gino.fno_n_modes,
                 fno_hidden_channels=params.gino.fno_hidden_channels,
                 lifting_channels=params.gino.fno_lifting_channels).to(device)

    n_params = count_model_params(model)
    print(f'\nOur model has {n_params} parameters.')
    sys.stdout.flush()

    optimizer = AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)

    train_loss = h1loss
    eval_losses = {'h1': h1loss, 'l2': l2loss}

    print('\n### MODEL ###\n', model)
    print('\n### OPTIMIZER ###\n', optimizer)
    print('\n### SCHEDULER ###\n', scheduler)
    print('\n### LOSSES ###')
    print(f'\n * Train: {train_loss}')
    print(f'\n * Test: {eval_losses}')
    sys.stdout.flush()

    # train the network without using trainer
    u_range = torch.linspace(0, 1, 16)
    v_range = torch.linspace(0, 1, 16)
    uu, vv = torch.meshgrid(u_range, v_range, indexing='xy')
    uv_grid = torch.cat([uu.unsqueeze(0), vv.unsqueeze(0)]).permute(1, 2, 0)
    uv_gino = uv_grid.flatten(0, 1).to(device)

    input_geom = uv_gino.unsqueeze(0)
    latent_queries = torch.stack(
        torch.meshgrid([torch.linspace(0, 1, 64)] * 2, indexing='xy')
    )
    latent_queries = latent_queries.permute(*list(range(1, 2 + 1)), 0).to(
        device)  # (1, grid_size, grid_size, ..., grid_size, gno_coord_dim)
    output_queries = uv_gino

    # Training
    model.train()
    for epoch in range(20):
        for i, data in enumerate(augmented_loader):
            x, y = data['x'].to(device), data['y'].to(device)
            optimizer.zero_grad()
            y_pred = model(x, input_geom, latent_queries.unsqueeze(0), output_queries)
            loss = train_loss(y_pred, y)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f'Epoch {epoch + 1}, iter {i}, loss: {loss.item()}')
                sys.stdout.flush()

        # save model
        if epoch == 0 or (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'./model_epoch_{epoch}.pth')
        scheduler.step()
