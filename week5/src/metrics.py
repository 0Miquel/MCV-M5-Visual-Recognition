from typing import List, Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import image as mpimg
from torch.utils.data import DataLoader
from tqdm import tqdm

from week5.src.models import TripletNetIm2Text, TripletNetText2Im


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
        visualize: bool = False,
        device: str = 'cpu'
) -> Tuple[float, float, float]:
    """
    Evaluate the model on the dataloader and print the predicted captions
    :param model: TripletNetIm2Text model
    :param dataloader: DataLoader containing the images
    :param db: np.ndarray containing the captions embeddings
    :param captions_dict: Dict[int, str] containing the captions
    :param visualize: bool, whether to visualize the image and predicted captions
    :param device: str, device to use
    :return: mAP@5, mP@1, mP@5, mP@10
    """
    model.eval()
    captions = np.vstack(db[:, 0]).astype(np.float32)
    ap5_list = []
    p1_list = []
    p5_list = []
    model.to(device)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            anchor_b, y_true_b = batch
            anchor_embedding_b = model.get_embedding_image(anchor_b.to(device))
            db_tensor = torch.Tensor(captions).to(device)  # .to(model.device)
            for i, anchor_embedded in enumerate(anchor_embedding_b):
                sim_scores = torch.nn.functional.cosine_similarity(anchor_embedded, db_tensor)
                y_true_i = [int(captions_tensor[i].cpu().detach().numpy()) for captions_tensor in y_true_b]
                pred_top_5_items = np.argsort(sim_scores.cpu().detach().numpy())[::-1][:5]
                pred_top_5_captions_ids = [db[j][1] for j in pred_top_5_items]
                pred_captions = [captions_dict[k_id] for k_id in pred_top_5_captions_ids]

                ap5_list.append(calculate_ap_at_5_im2text(y_true_i, y_pred=pred_top_5_captions_ids))
                p1_list.append(calculate_precision_im2text(y_true_i, y_pred=pred_top_5_captions_ids, k_values=1))
                p5_list.append(calculate_precision_im2text(y_true_i, y_pred=pred_top_5_captions_ids, k_values=5))

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

                    # Add the ground truth captions to the bottom of the plot
                    text = "Ground truth captions:\n"
                    for j, caption in enumerate(y_true_i):
                        text += f"{j + 1}. {captions_dict[caption]}\n"

                    # Add the predicted captions to the bottom of the plot
                    text += "\nPredicted captions:\n"
                    for j, caption in enumerate(pred_captions):
                        text += f"{j + 1}. {caption}\n"
                    ax.text(0.5, 1.05, text, transform=ax.transAxes, ha='center', va='bottom')
                    # Adjust the spacing
                    fig.subplots_adjust(top=0.5)
                    # Show the plot
                    plt.show()

    # Calculate the mean of the metrics
    map5 = float(np.mean(np.array(ap5_list)))
    mp1 = float(np.mean(np.array(p1_list)))
    mp5 = float(np.mean(np.array(p5_list)))
    return map5, mp1, mp5


def visualize_retrieved_images(query_caption, gt_image, image_list, num_images=5):
    """
    Display the query caption on top, the ground truth image, and the retrieved images below.

    Args:
        query_caption (str): The query caption to display.
        gt_image (str): The filepath of the ground truth image.
        image_list (list): A list of filepaths or image arrays of the retrieved images.
        num_images (int, optional): The number of images to display. Defaults to 5.
    """
    num_images = min(len(image_list), num_images)

    fig, ax = plt.subplots(1, num_images + 2, figsize=(20, 10))

    # Display the caption text in the first subplot
    ax[0].axis('off')
    ax[0].text(0.5, 0.5, query_caption, fontsize=10, ha='center', va='top', transform=ax[0].transAxes)

    # Display the ground truth image in the second subplot
    gt_img = mpimg.imread(gt_image)
    ax[1].imshow(gt_img)
    ax[1].axis('off')

    # Display the retrieved images in the remaining subplots
    for i in range(num_images):
        img = image_list[i]

        # Check if the input is a filepath or an image array
        if isinstance(img, str):
            img = mpimg.imread(img)

        ax[i + 2].imshow(img)
        ax[i + 2].axis('off')

    plt.tight_layout()
    plt.show()


def evaluate_text2im(
        model: TripletNetText2Im,
        dataloader: DataLoader,
        db: np.ndarray,
        imgs_dict: Dict[int, str],
        visualize: bool = False,
        device: str = 'cpu',
        im_dir: str = None
) -> Tuple[float, float, float]:
    """
    Evaluate the model on the dataloader and print the predicted captions
    :param model: TripletNetIm2Text model
    :param dataloader: DataLoader containing the images
    :param db: np.ndarray containing the captions embeddings
    :param imgs_dict: Dict[int, str] containing the captions
    :param visualize: bool, whether to visualize the image and predicted captions
    :param device: str, device to use
    :param im_dir: str, directory containing the images
    :return: mAP@5, mP@1, mP@5, mP@10
    """
    model.eval()
    captions = np.vstack(db[:, 0]).astype(np.float32)
    ap5_list = []
    p1_list = []
    p5_list = []
    model.to(device)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            anchor_b, y_true_b = batch
            anchor_embedding_b = model.get_embedding_text(anchor_b)
            db_tensor = torch.Tensor(captions).to(device)
            for i, anchor_embedded in enumerate(anchor_embedding_b):
                sim_scores = torch.nn.functional.cosine_similarity(torch.Tensor(anchor_embedded).to(device), db_tensor)
                y_true_i = int(y_true_b[i].cpu().detach().numpy())
                pred_top_5_items = np.argsort(sim_scores.cpu().detach().numpy())[::-1][:5]
                pred_top_5_imgs_ids = [db[j][1] for j in pred_top_5_items]
                pred_imgs_paths = [imgs_dict[k_id] for k_id in pred_top_5_imgs_ids]

                ap5_list.append(calculate_ap_at_5_im2text([y_true_i], y_pred=pred_top_5_imgs_ids))
                p1_list.append(calculate_precision_im2text([y_true_i], y_pred=pred_top_5_imgs_ids, k_values=1))
                p5_list.append(calculate_precision_im2text([y_true_i], y_pred=pred_top_5_imgs_ids, k_values=5))

                if visualize:
                    # Visualize image and captions
                    caption = anchor_b[i]
                    gt_image = f'{im_dir}/{imgs_dict[y_true_i]}'
                    preds_imgs = [Image.open(f'{im_dir}/{img}') for img in pred_imgs_paths]
                    visualize_retrieved_images(query_caption=caption, gt_image=gt_image, image_list=preds_imgs)

    # Calculate the mean of the metrics
    map5 = float(np.mean(np.array(ap5_list)))
    mp1 = float(np.mean(np.array(p1_list)))
    mp5 = float(np.mean(np.array(p5_list)))
    return map5, mp1, mp5


def evaluate_text2img():
    pass
