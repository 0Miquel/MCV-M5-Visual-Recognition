from typing import List, Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from week5.src.models import TripletNetIm2Text


def calculate_ap_at_5_im2text(y_true, y_pred) -> float:
    correct_list = []

    for y_pred_i in y_pred:
        pred_correct = 0
        for y_true_j in y_true:
            if y_true_j == y_pred_i:
                pred_correct = 1
                break
        correct_list.append(pred_correct)

    # calculate retrieval average precision
    ap = 0
    for i in range(len(correct_list)):
        ap += correct_list[i] / (i + 1)

    return ap


def calculate_precision_im2text(y_true, y_pred, k_values) -> float:
    correct_list = []

    for y_pred_i in y_pred:
        pred_correct = 0
        for y_true_j in y_true:
            if y_true_j == y_pred_i:
                pred_correct = 1
                break
        correct_list.append(pred_correct)

    # calculate retrieval precision
    precision = 0
    for i in range(k_values):
        precision += correct_list[i] / k_values
    return precision


def evaluate_im2text(
        model: TripletNetIm2Text,
        dataloader: DataLoader,
        db: np.ndarray,
        captions_dict: Dict[int, str],
        visualize: bool = False
) -> Tuple[float, float, float]:
    """
    Evaluate the model on the dataloader and print the predicted captions
    :param model: TripletNetIm2Text model
    :param dataloader: DataLoader containing the images
    :param db: np.ndarray containing the captions embeddings
    :param captions_dict: Dict[int, str] containing the captions
    :param visualize: bool, whether to visualize the image and predicted captions
    :return: mAP@5, mP@1, mP@5
    """
    model.eval()
    captions = np.vstack(db[:, 0]).astype(np.float32)
    ap5_list = []
    p1_list = []
    p5_list = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            anchor_b, y_true_b = batch
            anchor_embedding_b = model.get_embedding_image(anchor_b)
            db_tensor = torch.Tensor(captions)  # .to(model.device)
            for i, anchor_embedded in enumerate(anchor_embedding_b):
                sim_scores = torch.nn.functional.cosine_similarity(anchor_embedded, db_tensor)
                y_true_i = [int(captions_tensor[i].cpu().detach().numpy()) for captions_tensor in y_true_b]
                pred_top_5 = np.argsort(sim_scores.cpu().detach().numpy())[:5]
                pred_top_5_items = [db[j][1] for j in pred_top_5]
                pred_captions = [captions_dict[i] for i in pred_top_5_items]

                ap5_list.append(calculate_ap_at_5_im2text(y_true_i, y_pred=pred_top_5_items))
                p1_list.append(calculate_precision_im2text(y_true_i, y_pred=pred_top_5_items, k_values=1))
                p5_list.append(calculate_precision_im2text(y_true_i, y_pred=pred_top_5_items, k_values=5))

                if visualize:
                    # Visualize image and captions
                    image = anchor_b[i].cpu().detach()
                    std_tensor = torch.tensor([0.485, 0.456, 0.406])
                    mean_tensor = torch.tensor([0.229, 0.224, 0.225])
                    denormalized_image = image * std_tensor[:, None, None] + mean_tensor[:, None, None]
                    denormalized_image = denormalized_image.permute(1, 2, 0).numpy()

                    # Plot the image and remove the axis
                    fig, ax = plt.subplots()
                    ax.imshow(denormalized_image)
                    ax.axis('off')

                    # Add the predicted captions to the bottom of the plot
                    text = "Predicted captions:\n"
                    for j, caption in enumerate(pred_captions[::-1]):
                        text += f"{j + 1}. {caption}\n"
                    ax.text(0.5, 1.05, text, transform=ax.transAxes, ha='center', va='bottom')
                    # Adjust the spacing
                    fig.subplots_adjust(top=0.7)
                    # Show the plot
                    plt.show()

    # Calculate the mean of the metrics
    map5 = float(np.mean(np.array(ap5_list)))
    mp1 = float(np.mean(np.array(p1_list)))
    mp5 = float(np.mean(np.array(p5_list)))

    return map5, mp1, mp5


def evaluate_text2img():
    pass
