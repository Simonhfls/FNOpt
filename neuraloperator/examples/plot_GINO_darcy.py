
from matplotlib import pyplot as plt

from examples.train_GINO_darcy import AugmentedDataset
from neuralop.models import GINO
from neuralop.data.datasets import load_darcy_flow_small
import torch

device = 'cpu'

def get_uv_grid(resolution, indexing='xy'):
    u_range = torch.linspace(0, 1, resolution)
    v_range = torch.linspace(0, 1, resolution)
    uu, vv = torch.meshgrid(u_range, v_range, indexing=indexing)
    uv_grid = torch.cat([uu.unsqueeze(0), vv.unsqueeze(0)]).permute(1, 2, 0)
    uv_gino = uv_grid.flatten(0, 1).to(device)
    return uv_gino

def get_latent_queries(latent_size):
    latent_queries = torch.stack(torch.meshgrid([torch.linspace(0, 1, latent_size)] * 2, indexing='xy'))
    latent_queries = latent_queries.permute(*list(range(1, 2 + 1)), 0).to(device)  # (1, grid_size, grid_size, ..., grid_size, gno_coord_dim)
    return latent_queries


def gino_test_resolution():
    n_col = 5
    fig = plt.figure(figsize=(10, 7))
    uv_gino = get_uv_grid(16)
    input_geom = uv_gino.unsqueeze(0)
    latent_queries = get_latent_queries(64)  # 64x64
    output_queries = uv_gino
    uv32_gino = get_uv_grid(32)
    n = 64  # resolution
    uvn_gino = get_uv_grid(n)

    input_geom32 = uv32_gino.unsqueeze(0)
    # latent_queries32 = torch.stack(torch.meshgrid([torch.linspace(0, 1, 128)] * 2, indexing='xy'))
    # latent_queries32 = latent_queries32.permute(*list(range(1, 2 + 1)), 0).to(device)  # (1, grid_size, grid_size, ..., grid_size, gno_coord_dim)
    output_queries32 = uv32_gino

    # interpolate input_geom32 to get input_geomn
    input_geomn = uvn_gino.unsqueeze(0)
    output_queriesn = uvn_gino
    with torch.no_grad():
        for index in range(3):
            data = test_samples[index]
            data32 = test_samples32[index]
            # Input x
            x = data['x'].to(device)
            x32 = data32['x'].to(device)
            # Ground-truth
            y = data['y'].to(device)

            # interpolate x32 to xn
            xn = torch.nn.functional.interpolate(x32.reshape(1, 32, 32).unsqueeze(0), size=(n, n), mode='bilinear', align_corners=False).squeeze(0)
            xn = xn.reshape(-1, 1)

            out = model(x.unsqueeze(0), input_geom, latent_queries.unsqueeze(0), output_queries)
            out32 = model(x32.unsqueeze(0), input_geom32, latent_queries.unsqueeze(0), output_queries32)
            outn = model(xn.unsqueeze(0), input_geomn, latent_queries.unsqueeze(0), output_queriesn)

            ax = fig.add_subplot(3, n_col, index*n_col + 1)
            ax.imshow(x.reshape(16, 16, 1), cmap='gray')
            if index == 0:
                ax.set_title('Input x')
            plt.xticks([], [])
            plt.yticks([], [])

            ax = fig.add_subplot(3, n_col, index*n_col + 2)
            ax.imshow(y.reshape(16, 16, 1).squeeze())
            if index == 0:
                ax.set_title('Ground-truth y')
            plt.xticks([], [])
            plt.yticks([], [])

            ax = fig.add_subplot(3, n_col, index*n_col + 3)
            ax.imshow(out.reshape(16, 16, 1).squeeze())
            if index == 0:
                ax.set_title('Prediction')
            plt.xticks([], [])
            plt.yticks([], [])

            ax = fig.add_subplot(3, n_col, index*n_col + 4)
            ax.imshow(out32.reshape(32, 32, 1).squeeze())
            if index == 0:
                ax.set_title('Prediction (32x32)')
            plt.xticks([], [])
            plt.yticks([], [])

            ax = fig.add_subplot(3, n_col, index * n_col + 5)
            ax.imshow(outn.reshape(n, n, 1).squeeze())
            if index == 0:
                ax.set_title(f'Prediction ({n}x{n})')
            plt.xticks([], [])
            plt.yticks([], [])


    fig.suptitle('Inputs, ground-truth output and prediction (16x16).', y=0.98)
    plt.tight_layout()
    fig.show()

def gino_test_latent_queries(predict_res=16):
    latent_queries128 = get_latent_queries(128)
    latent_queries64 = get_latent_queries(64)
    latent_queries32 = get_latent_queries(32)
    latent_queries16 = get_latent_queries(16)
    n_col = 6
    fig = plt.figure(figsize=(12, 7))
    test_samples = test_loaders[predict_res].dataset
    uv_gino = get_uv_grid(predict_res)
    input_geom = uv_gino.unsqueeze(0)
    output_queries = uv_gino

    with torch.no_grad():
        for index in range(3):
            data = test_samples[index]
            # Input x
            x = data['x'].to(device)
            # Ground-truth
            y = data['y'].to(device)
            out128 = model(x.unsqueeze(0), input_geom, latent_queries128.unsqueeze(0), output_queries)
            out64 = model(x.unsqueeze(0), input_geom, latent_queries64.unsqueeze(0), output_queries)
            out32 = model(x.unsqueeze(0), input_geom, latent_queries32.unsqueeze(0), output_queries)
            out16 = model(x.unsqueeze(0), input_geom, latent_queries16.unsqueeze(0), output_queries)

            ax = fig.add_subplot(3, n_col, index*n_col + 1)
            ax.imshow(x.reshape(predict_res, predict_res, 1), cmap='gray')
            if index == 0:
                ax.set_title('Input x')
            plt.xticks([], [])
            plt.yticks([], [])

            ax = fig.add_subplot(3, n_col, index*n_col + 2)
            ax.imshow(y.reshape(predict_res, predict_res, 1).squeeze())
            if index == 0:
                ax.set_title('Ground-truth y')
            plt.xticks([], [])
            plt.yticks([], [])


            ax = fig.add_subplot(3, n_col, index*n_col + 3)
            ax.imshow(out128.reshape(predict_res, predict_res, 1).squeeze())
            if index == 0:
                ax.set_title('Prediction (lq128)')
            plt.xticks([], [])
            plt.yticks([], [])

            ax = fig.add_subplot(3, n_col, index*n_col + 4)
            ax.imshow(out64.reshape(predict_res, predict_res, 1).squeeze())
            if index == 0:
                ax.set_title('Prediction (lq64*)')
            plt.xticks([], [])
            plt.yticks([], [])

            ax = fig.add_subplot(3, n_col, index*n_col + 5)
            ax.imshow(out32.reshape(predict_res, predict_res, 1).squeeze())
            if index == 0:
                ax.set_title('Prediction (lq32)')
            plt.xticks([], [])
            plt.yticks([], [])

            ax = fig.add_subplot(3, n_col, index * n_col + 6)
            ax.imshow(out16.reshape(predict_res, predict_res, 1).squeeze())
            if index == 0:
                ax.set_title('Prediction (lq16)')
            plt.xticks([], [])
            plt.yticks([], [])

    fig.suptitle(f'Inputs, ground-truth output and prediction ({predict_res}x{predict_res}).', y=0.98)
    plt.tight_layout()
    plt.show()
    # fig.show()
    pass


if __name__ == '__main__':
    model_path = 'model_epoch_15_ok.pth'

    model = GINO(in_channels=1,
                 out_channels=1,
                 projection_channels=256,
                 gno_coord_dim=2,
                 gno_radius=0.032,
                 fno_n_modes=(16, 16),
                 fno_hidden_channels=32,
                 lifting_channels=256).to(device)

    # load weights to model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # model2 = GINO(in_channels=1,
    #              out_channels=1,
    #              projection_channels=256,
    #              gno_coord_dim=2,
    #              gno_radius=0.032,
    #              fno_n_modes=(16, 16),
    #              fno_hidden_channels=32,
    #              lifting_channels=256).to(device)
    #
    # # load weights to model
    # model2.load_state_dict(torch.load(model_path, map_location=device))
    # model2.eval()

    train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000, batch_size=5,
        test_resolutions=[16, 32], n_tests=[100, 50],
        test_batch_sizes=[32, 32],
    )
    data_processor = data_processor.to(device)
    base_dataset = train_loader.dataset
    augmented_dataset = AugmentedDataset(base_dataset)
    augmented_loader = torch.utils.data.DataLoader(
        augmented_dataset,
        batch_size=train_loader.batch_size,
        shuffle=train_loader.sampler is not None,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory
    )

    base_dataset = test_loaders[16].dataset

    augmented_dataset_test = AugmentedDataset(base_dataset)

    augmented_loader_test16 = torch.utils.data.DataLoader(
        augmented_dataset_test,
        batch_size=test_loaders[16].batch_size,
        shuffle=test_loaders[16].sampler is not None,  # keep consistent with original DataLoader
        num_workers=test_loaders[16].num_workers,
        pin_memory=test_loaders[16].pin_memory
    )

    base_dataset = test_loaders[32].dataset

    augmented_dataset_test = AugmentedDataset(base_dataset)

    augmented_loader_test32 = torch.utils.data.DataLoader(
        augmented_dataset_test,
        batch_size=test_loaders[32].batch_size,
        shuffle=test_loaders[32].sampler is not None,  # keep consistent with original DataLoader
        num_workers=test_loaders[32].num_workers,
        pin_memory=test_loaders[32].pin_memory
    )

    test_loaders[16] = augmented_loader_test16
    test_loaders[32] = augmented_loader_test32



    test_samples = test_loaders[16].dataset
    test_samples32 = test_loaders[32].dataset


    gino_test_resolution()
    gino_test_latent_queries(predict_res=32)
    pass