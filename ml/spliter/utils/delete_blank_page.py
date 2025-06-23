import numpy as np


def _calculate_black_percentage(img):
    grayscale_image = np.array(img.convert("L"))
    return (np.where(grayscale_image < 200, 1, 0).sum()
            / np.ones(shape=grayscale_image.shape).sum()
            * 100
            )


def crop_30(pil_img):
    img_width, img_height = pil_img.size
    return pil_img.crop((60, 60, img_width - 60, img_height - 60))


def del_blank_pages(images):
    not_blank_images = []
    for i, page in enumerate(images, start=1):
        page_in_merged_pdf, image = page

        count_pixel = _calculate_black_percentage(crop_30(image))
        if count_pixel >= 0.05:
            not_blank_images.append((page_in_merged_pdf, image))
    return not_blank_images
