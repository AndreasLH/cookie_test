import torch
import hydra
import logging
import wandb
import numpy as np
from hydra.utils import get_original_cwd

from src.data.load_dataset import mnist
from src.models.model import MyAwesomeModel

log = logging.getLogger(__name__)
wandb.init(project='MNIST_Cookie_test')


@hydra.main(version_base=None, config_path="../../conf", config_name="config.yaml")
def evaluate(cfg):
    # log.info("Working directory : {}".format(os.getcwd()))
    # log.info(f"configuration: \n{OmegaConf.to_yaml(cfg)}")
    hparams_model = cfg.model.hparams
    hparams_ex = cfg.experiment.hparams

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = hparams_ex.batch_size
    x_dim = hparams_model.x_dim
    hidden_dim = hparams_model.hidden_dim
    hidden_dim2 = hparams_model.hidden_dim2
    latent_dim = hparams_model.latent_dim

    model_checkpoint = get_original_cwd() + "/models/checkpoint.pth"
    log.info("Evaluating until hitting the ceiling")
    log.info(model_checkpoint)

    class Color:
        red = "\033[31m"
        green = "\033[32m"

    color = Color.green
    # evaluation logic here
    state_dict = torch.load(model_checkpoint)
    _, test_set = mnist(data_path=get_original_cwd() + "/data/processed/",
                        eval=False, batch_size=batch_size)
    model = MyAwesomeModel(x_dim, hidden_dim, hidden_dim2, latent_dim)

    accuracy = test(model.to(DEVICE), test_set, state_dict)
    if accuracy.item() * 100 < 85:
        color = Color.red

    log.info(f"{color}Accuracy: {accuracy.item()*100:.3f}%")


def test(model, test_set, state_dict):
    """Test (evaluation) function for the model

    Args:
        model (nn.Module): model to test
        test_set (torch.utils.data.DataLoader): test set
        state_dict (dict): state dict of the model
    Returns:
        accuracy (torch.Tensor): accuracy of the model
    """
    # TODO: might get an error if the model is trained on gpu but inference is done on cpu
    model.load_state_dict(state_dict)

    y_true = []
    y_pred = []

    with torch.no_grad():
        model.eval()
        accuracy = 0
        for i, (images, labels) in enumerate(test_set):
            log_ps = model(images)
            ps = torch.exp(log_ps)
            _, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            # print("labels: \t", labels, "\n", "prediction: \t", top_class.flatten())
            accuracy += torch.mean(equals.type(torch.FloatTensor))

            y_true.append(list(labels.numpy()))
            y_pred.append(list(top_class.flatten().numpy()))

            # wandb.log({"pr": wandb.plot.pr_curve(labels.numpy(), top_class.flatten().numpy())})
        accuracy /= len(test_set)

        def flatten_list(list_):
            return [item for sublist in list_ for item in sublist]
        y_true = flatten_list(y_true)
        y_pred = flatten_list(y_pred)
        wandb.log({"accuracy": accuracy})
        cm = wandb.plot.confusion_matrix(
            y_true=np.array(y_true),
            preds=np.array(y_pred),
            class_names=list(range(10)))

        wandb.log({"conf_mat": cm})

    return accuracy


if __name__ == "__main__":
    evaluate()
