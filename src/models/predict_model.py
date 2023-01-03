import click
import torch

from src.data.load_dataset import mnist
from src.models.model import MyAwesomeModel


@click.group()
def cli():
    pass


@click.command()
@click.argument("model_checkpoint")
@click.argument("data_path")
def evaluate(model_checkpoint, data_path):
    model_checkpoint = "models/checkpoint.pth"
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    class Color:
        red = "\033[31m"
        green = "\033[32m"

    color = Color.green
    # evaluation logic here
    state_dict = torch.load(model_checkpoint)
    _, test_set = mnist(data_path, eval=True)
    model = MyAwesomeModel()

    accuracy = test(model, test_set, state_dict)
    if accuracy.item() * 100 < 85:
        color = Color.red

    print(color, f"Accuracy: {accuracy.item()*100:.3f}%")


def test(model, test_set, state_dict):
    """Test (evaluation) function for the model

    Args:
        model (nn.Module): model to test
        test_set (torch.utils.data.DataLoader): test set
        state_dict (dict): state dict of the model
    Returns:
        accuracy (torch.Tensor): accuracy of the model
    """
    model.load_state_dict(state_dict)

    with torch.no_grad():
        model.eval()
        accuracy = 0
        for images, labels in test_set:
            log_ps = model(images)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            print("labels: \t", labels, "\n", "prediction: \t", top_class.flatten())
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        accuracy /= len(test_set)
    return accuracy


cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
