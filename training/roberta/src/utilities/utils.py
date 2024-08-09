
import os
import inspect
from torch.optim import Adam, AdamW
from transformers.optimization import get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from torchvision.transforms.functional import to_pil_image
from zipfile import ZipFile
import io
import tqdm

def check_zip(*args):
    args = [list(arg) for arg in args]
    length = len(args[0])
    for arg in args:
        assert len(arg) == length
    return zip(*args)

def filter_inputs(func, inputs):
    func_args = inspect.signature(func).parameters.keys()
    filtered = {key:val for key, val in inputs.items() if key in func_args}
    return filtered

def get_optimizer(
    optimizer, optim_groups, base_learning_rate, adam_betas = (0.9, 0.98), adam_epsilon = 1e-6, **kwargs
):
    optimizer = optimizer.lower()
    optim_cls = {
        "adam": AdamW
    }[optimizer]

    args = [optim_groups]
    kwargs = {
        "lr": base_learning_rate,
        "eps": adam_epsilon,
        "betas": adam_betas,
    }
    optimizer = optim_cls(*args, **kwargs)

    return optimizer


def image_binary(img):
    img = to_pil_image(img)
    byte_arr = io.BytesIO()
    img.save(byte_arr, format = 'webp')
    return byte_arr.getvalue()

def write_images(real_imgs, fake_imgs, texts, folder_path, file_name):
    assert real_imgs.shape[0] == fake_imgs.shape[0]
    if texts is not None:
        assert real_imgs.shape[0] == len(texts)

    os.makedirs(folder_path, exist_ok = True)
    with ZipFile(os.path.join(folder_path, file_name), 'w') as zip_file:
        html_str = "<!DOCTYPE html>"
        for img_idx, (real, fake) in tqdm.tqdm(enumerate(zip(real_imgs.unbind(0), fake_imgs.unbind(0)))):
            zip_file.writestr(f'{img_idx:06d}-real.webp', image_binary(real))
            zip_file.writestr(f'{img_idx:06d}-fake.webp', image_binary(fake))
            html_str += f"<img src={img_idx:06d}-real.webp alt=\"null\" width=\"160\" height=\"160\">\n"
            html_str += f"<img src={img_idx:06d}-fake.webp alt=\"null\" width=\"160\" height=\"160\">\n"
            if texts is not None:
                html_str += f"<p>{texts[img_idx]}</p>\n"
            else:
                html_str += f"<p></p>\n"
        zip_file.writestr(f'index.html', html_str)