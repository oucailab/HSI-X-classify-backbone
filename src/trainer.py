from torch.backends import cudnn

from .data import *

import time
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from . import config
from .model_registry import get_model_adapter

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def getLog(log_path, str):
    import os
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, 'a+') as log:
        log.write('{}'.format(str))
        log.write('\n')


def train(epochs, lr, model, cuda, train_loader, test_loader, out_features, model_savepath, log_path, hsi_pca_wight, datasetType):
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    cudnn.benchmark = True

    hsi_pca_wight_tensor = torch.from_numpy(hsi_pca_wight).to(device)

    model_adapter = get_model_adapter(model)
    model_bundle = model_adapter.build_model(config, datasetType, device)
    net = model_bundle["net"]
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    max_acc = 0
    sum_time = 0

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    current_time_log = 'start time: {}'.format(current_time)
    getLog(log_path, config.get_taskInfo())
    getLog(log_path, '-------------------Started Training-------------------')
    getLog(log_path, current_time_log)

    for epoch in range(epochs):
        since = time.time()
        net.train()

        try:
            iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch')
        except Exception:
            iterator = train_loader
        for i, (hsi_pca, hsi, sar, tr_labels) in enumerate(iterator):
            batch = {
                "hsi_pca": hsi_pca.to(device),
                "hsi": hsi.to(device),
                "aux": sar.to(device),
                "label": tr_labels.to(device),
            }

            optimizer.zero_grad()
            outputs = model_adapter.forward_train(model_bundle, batch)
            loss = criterion(outputs, batch["label"])

            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            net.eval()
            count = 0

            for hsi_pca, hsi, sar, gtlabels in test_loader:
                batch = {
                    "hsi_pca": hsi_pca.to(device),
                    "hsi": hsi.to(device),
                    "aux": sar.to(device),
                    "label": gtlabels,
                }

                with torch.no_grad():
                    outputs = model_adapter.forward_eval(model_bundle, batch)

                outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                if count == 0:
                    y_pred_test = outputs
                    gty = gtlabels
                    count = 1
                else:
                    y_pred_test = np.concatenate((y_pred_test, outputs))
                    gty = np.concatenate((gty, gtlabels))

            acc1 = accuracy_score(gty, y_pred_test)

            if acc1 > max_acc:
                os.makedirs(os.path.dirname(model_savepath), exist_ok=True)
                torch.save(net, model_savepath)
                max_acc = acc1

            time_elapsed = time.time() - since
            sum_time += time_elapsed
            rest_time = (sum_time / (epoch + 1)) * (epochs - epoch - 1)
            currentTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            log = currentTime + ' [Epoch: %d] [%.0fs, %.0fh %.0fm %.0fs] [current loss: %.4f] acc: %.4f' %(epoch + 1, time_elapsed, (rest_time // 60) // 60, (rest_time // 60) % 60, rest_time % 60, loss.item(), acc1)
            print(log)
            getLog(log_path, log)

    print('max_acc: %.4f' %(max_acc))
    finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    finish_time_log = 'finish time: {} '.format(finish_time)
    mac_acc_log = 'max_acc: {} '.format(max_acc)
    getLog(log_path, mac_acc_log)
    getLog(log_path, finish_time_log)
    getLog(log_path, '-------------------Finished Training-------------------')


def myTrain(datasetType, model):
    """训练函数，从配置系统获取参数"""
    channels = config.get_value('channels')
    windowSize = config.get_value('windowSize')
    out_features = config.get_value('out_features')
    cuda = config.get_value('cuda')
    lr = config.get_value('lr')
    epoch_nums = config.get_value('epoch_nums')
    batch_size = config.get_value('batch_size')
    num_workers = config.get_value('num_workers')
    random_seed = config.get_value('random_seed')
    model_savepath = config.get_value('model_savepath')
    log_path = config.get_value('log_path')

    print(f"训练参数: model={model}, epochs={epoch_nums}, lr={lr}, cuda={cuda}")
    set_random_seed(random_seed)
    train_loader, test_loader, trntst_loader, all_loader, hsi_pca_wight = getMyData(datasetType, channels, windowSize, batch_size, num_workers)

    train(epoch_nums, lr, model, cuda, train_loader, test_loader, out_features[datasetType], model_savepath[datasetType], log_path[datasetType], hsi_pca_wight, datasetType)
