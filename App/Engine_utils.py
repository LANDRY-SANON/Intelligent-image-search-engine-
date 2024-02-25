import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq
from sklearn import preprocessing
import matplotlib.pyplot as plt


def search_similar_image_by_histogram(query_image_path, database_dir):
    def compute_histogram(image_path):
        img = cv2.imread(image_path)
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def compute_cosine_similarity(query_hist, database_hists):
        similarity_scores = cosine_similarity(query_hist.reshape(1, -1), database_hists)
        return similarity_scores

    def find_most_similar_image(query_hist, database_hists, database_image_paths):
        similarity_scores = compute_cosine_similarity(query_hist, database_hists)
        most_similar_index = np.argmax(similarity_scores)
        return database_image_paths[most_similar_index]

    query_hist = compute_histogram(query_image_path)
    database_hists = []
    database_image_paths = []

    for filename in os.listdir(database_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(database_dir, filename)
            hist = compute_histogram(image_path)
            database_hists.append(hist)
            database_image_paths.append(image_path)

    database_hists = np.array(database_hists)

    most_similar_image_path = find_most_similar_image(query_hist, database_hists, database_image_paths)

    return most_similar_image_path




def search_similar_images(query_image_path, descriptors_list, k=10):
    def compute_descriptors(image_paths):
        sift = cv2.xfeatures2d.SIFT_create()
        des_list = []
        for image_path in image_paths:
            im = cv2.imread(image_path)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            des_list.append((image_path, des))
        return des_list

    def compute_visual_words(descriptors, k):
        descriptors_concat = descriptors[0][1]
        for _, descriptor in descriptors[1:]:
            descriptors_concat = np.vstack((descriptors_concat, descriptor))
        kmeans = KMeans(n_clusters=k, random_state=1).fit(descriptors_concat)
        return kmeans.cluster_centers_

    def compute_image_features(descriptors_list, visual_words):
        k = visual_words.shape[0]
        im_features = np.zeros((len(descriptors_list), k), "float32")
        for i, (_, des) in enumerate(descriptors_list):
            words, _ = vq(des, visual_words)
            for w in words:
                im_features[i][w] += 1
        im_features = preprocessing.normalize(im_features, norm='l2')
        return im_features

    def compute_query_image_features(query_image_path, sift, visual_words):
        query_image = cv2.imread(query_image_path)
        query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        query_kp, query_des = sift.detectAndCompute(query_gray, None)
        query_words, _ = vq(query_des, visual_words)
        query_image_features = np.zeros((1, visual_words.shape[0]), dtype=np.float32)
        for word in query_words:
            query_image_features[0, word] += 1
        query_image_features = preprocessing.StandardScaler().fit_transform(query_image_features)
        return query_image_features

    def find_similar_images(query_image_features, im_features):
        scores = np.dot(query_image_features, im_features.T)
        ranked_indices = np.argsort(-scores)
        return ranked_indices , scores

    def plot_similar_images(image_paths, scores, k=10):
        n = min(k, len(image_paths))
        for i in range(n):
            image_path = image_paths[i]
            score = scores[0][i]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image)
            plt.title(f"Similarity Score: {score}")
            plt.axis('off')
            plt.show()

    # Main logic
    test_path = '/content/drive/MyDrive/IISEngine_test_data/Data/References'
    image_paths = [os.path.join(test_path, f) for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]
    
    descriptors_list = compute_descriptors(image_paths)
    
    k = 100
    visual_words = compute_visual_words(descriptors_list, k)
    
    im_features = compute_image_features(descriptors_list, visual_words)
    
    sift = cv2.xfeatures2d.SIFT_create()
    query_image_features = compute_query_image_features(query_image_path, sift, visual_words)
    
    ranked_indices,scores = find_similar_images(query_image_features, im_features)
    
    most_similar_image_paths = [descriptors_list[i][0] for i in ranked_indices[0][:k]]
    most_similar_scores = scores[0][ranked_indices[0][:k]]
    
    plot_similar_images(most_similar_image_paths, most_similar_scores)