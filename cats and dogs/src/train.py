import torch
from tqdm import tqdm


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    loss_fn,
    device
):

    model.train()

    running_loss = 0

    train_bar = tqdm(dataloader)

    for images, labels in train_bar:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        train_bar.set_postfix(
            loss=loss.item()
        )

    return running_loss / len(dataloader)


def validate(
    model,
    dataloader,
    loss_fn,
    device
):

    model.eval()

    running_loss = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = loss_fn(outputs, labels)

            running_loss += loss.item()

    return running_loss / len(dataloader)
