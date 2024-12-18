import os
from PIL import Image
import numpy as np
from numpy import linalg

# 对图片进行统一化处理
def get_thumbnail(image, size=(64, 64), greyscale=False):
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        image = image.convert('L')
    return image

# 计算图片的余弦距离
def calculate_cosine_similarity(image1, image2):
    image1 = get_thumbnail(image1)
    image2 = get_thumbnail(image2)
    vectors = [np.array(image1).flatten(), np.array(image2).flatten()]
    norms = [linalg.norm(vector) for vector in vectors]
    a, b = vectors
    a_norm, b_norm = norms
    similarity = np.dot(a / a_norm, b / b_norm)
    return similarity

def compare_images_in_directory(directory):
    images = []
    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith(('.jpg', '.png')):
            images.append(Image.open(os.path.join(directory, filename)))

    if len(images) < 2:
        print("Not enough images to compare.")
        return []

    similarities = []
    for i in range(len(images) - 1):
        similarity = calculate_cosine_similarity(images[i], images[i + 1])
        similarities.append(similarity)

    return similarities

def find_sharp_changes(similarity_array, k):
    similarity_array = np.array(similarity_array)
    differences = np.abs(np.diff(similarity_array))
    sorted_indices = np.argsort(differences)[::-1]
    top_k_indices = sorted_indices[:k]
    return sorted(top_k_indices.tolist())

# 主程序
if __name__ == "__main__":
    directory = '/home/aogao/code/InstantSplat/data/Tanks/Barn/images'
    similarities = compare_images_in_directory(directory)
    indices = find_sharp_changes(similarities, 2)
    print(indices)
