from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.spatial.distance as dist_funcs


class Faces:
    def __init__(self):
        self.extensions_supported = ['png', 'jpg', 'PNG', 'JPG', 'jpeg', 'JPEG']

    @staticmethod
    def convert_to_gray_and_scale(image_path: str = 'images/me.jpg',
                                  image_output_path: str = 'images/me_scaled.jpg'):
        image = Image.open(image_path)
        # Convert to grayscale
        grayscale_image = image.convert('L')

        # Resize to 256x256
        resized_image = grayscale_image.resize((256, 256))
        resized_image.save('{path}'.format(path=image_output_path))
        return resized_image

    @staticmethod
    def plot_image(image):
        # Show image
        plt.imshow(image, cmap='gray')
        plt.title('Image')
        plt.axis('off')
        plt.show()

    def map_images_folder_and_convert(self, folder: str = 'images/grayscale_faces/') -> list:
        files = []
        [files.extend(glob.glob(folder + '*.' + e)) for e in self.extensions_supported]
        images = [self.convert_to_gray_and_scale(file, file) for file in files]
        return images

    def calculate_average_face(self):
        numpy_faces = map(np.array, self.map_images_folder_and_convert())
        average_face = np.mean(np.array(list(numpy_faces)), axis=0)
        self.plot_image(average_face)
        return average_face

    def calculate_distances_between_images(self, image_one=None, image_two=None, method: str = 'all'):

        def print_distance(method_, distance_):
            print('The {method} distance between the two images is {distance}'.format(method=method_,
                                                                                      distance="{:.2f}".format(
                                                                                          distance_)))

        list_of_methods = ['euclidean', 'minkowski', 'cityblock', 'sqeuclidean', 'cosine',
                           'correlation', 'hamming', 'jaccard', 'chebyshev', 'canberra', 'braycurtis']
        if image_one is None:
            image_one = np.array(self.convert_to_gray_and_scale())
        if image_two is None:
            image_two = self.calculate_average_face()
        if method == 'all':
            for method_i in list_of_methods:
                dist_func = getattr(dist_funcs, method_i)
                dist = dist_func(image_one.flatten(), image_two.flatten())
                print_distance(method_i, dist)
        else:
            if method in list_of_methods:
                dist_func = getattr(dist_funcs, method)
                dist = dist_func(image_one.flatten(), image_two.flatten())
                print_distance(method, dist)

