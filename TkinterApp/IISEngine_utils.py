import os
import cv2
import numpy as np
from skimage import io
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq , kmeans
from sklearn import preprocessing
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

def search_similar_image_by_color_histogram(image_entrer,base_image):
    import numpy as np
    import pandas as pd
    import cv2 as cv
    from skimage import io
    from PIL import Image
    import matplotlib.pylab as plt
    import cv2
    import os
    
    Image_List = list()

    def imShow(path):
        import cv2
        import matplotlib.pyplot as plt

        image = cv2.imread(path)
        height, width = image.shape[:2]
        resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

        fig = plt.gcf()
        fig.set_size_inches(18, 10)
        plt.axis("off")
        plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        plt.show()

    def compare_image(image1,image2):
        img1= cv2.imread(image1,1)
        img2= cv2.imread(image2,1)
        histogram1 = cv.calcHist([img1],[0],None,[256],[0,256])
        histogram2 = cv.calcHist([img2],[0],None,[256],[0,256])
        # Normaliser les histogrammes
        cv2.normalize(histogram1, histogram1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(histogram2, histogram2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        # Comparer les histogrammes
        correlation = cv2.compareHist(histogram1, histogram2, cv2.HISTCMP_CORREL)
        if (correlation>0.10):
            Image_List.append(image2)
         

    def image_to_hist_images(image_entrer,base_image):
        print(base_image)
        
        for nom_image in os.listdir(base_image):
            chemin_image = os.path.join(base_image, nom_image)
            if chemin_image.lower().endswith(('.png', '.jpg', '.jpeg')):
                compare_image(image_entrer, chemin_image)


    image_to_hist_images(image_entrer,base_image)
    return Image_List



def search_similar_image_by_SIFT(query_image_path, test_path, k=10):
    import argparse as ap
    import cv2
    import imutils
    import numpy as np
    import os
    import matplotlib.pylab as plt
    from sklearn.svm import LinearSVC
    from sklearn.cluster import KMeans
    from scipy.cluster.vq import kmeans , vp
    from sklearn.preprocessing import StandardScaler

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import neighbors

    from sklearn import preprocessing

    from skimage import io

    def extract_features(image_path):
        sift = cv2.xfeatures2d.SIFT_create()
        im = cv2.imread(image_path)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        return des

    def calculate_similarity(query_features, im_features):
        score = np.dot(query_features, im_features.T)
        rank_ID = np.argsort(-score)
        return rank_ID

    print('Query Image')

    query_descriptors = extract_features(query_image_path)

    allowed_extensions = ['.png', '.jpg', '.jpeg']
    test_image_paths = [os.path.join(test_path, f) for f in os.listdir(test_path) if os.path.splitext(f)[1].lower() in allowed_extensions]
    test_descriptors = [extract_features(image_path) for image_path in test_image_paths]

    descriptors = test_descriptors[0]
    for descriptor in test_descriptors[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    voc, _  = kmeans(descriptors, k, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01))
    im_features = np.zeros((len(test_image_paths), k), "float32")
    for i, descriptor in enumerate(test_descriptors):
        words, _ = vq(descriptor, voc)
        for w in words:
            im_features[i][w] += 1

    nbr_occurrences = np.sum((im_features > 0) * 1, axis=0)
    idf = np.array(np.log((1.0 * len(test_image_paths) + 1) / (1.0 * nbr_occurrences + 1)), 'float32')
    im_features = preprocessing.normalize(im_features, norm='l2')

    query_features = np.zeros((1, k), "float32")
    words, _ = vq(query_descriptors, voc)
    for w in words:
        query_features[0][w] += 1

    query_features = preprocessing.normalize(query_features, norm='l2')

    rank_ID = calculate_similarity(query_features, im_features)
    print('Results :')
    # Display top k similar images
    for i in rank_ID[0][:k]:
        image = cv2.imread(test_image_paths[i])
        plt.imshow(image)
        plt.show()



def search_similar_image_by_Pretrained_Model(query_image_path, database_dir, similarity_threshold=0.7):

    
    def load_and_preprocess_image(image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img
    

    def display_images(query_image_path, similar_image_paths):
        query_image = cv2.imread(query_image_path)
        plt.imshow(query_image)
        plt.show()
        print('Query Image')

        for i, image_path in enumerate(similar_image_paths):
            similar_image = cv2.imread(image_path)
            plt.imshow(similar_image)
            plt.show()
            print(f'Similar Image {i+1}')

    def extract_features(image_path, model):
        img = load_and_preprocess_image(image_path)
        features = model.predict(img)
        return features

    def compute_cosine_similarity(query_features, database_features):
        similarity_scores = cosine_similarity(query_features, database_features)
        return similarity_scores

    base_model = InceptionV3(weights='imagenet', include_top=True)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

    query_features = extract_features(query_image_path, model)
    database_features = []
    database_image_paths = []

    for filename in os.listdir(database_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(database_dir, filename)
            features = extract_features(image_path, model)
            database_features.append(features)
            database_image_paths.append(image_path)

    database_features = np.array(database_features).squeeze()
    similarity_scores = compute_cosine_similarity(query_features, database_features)
    similar_image_paths = [database_image_paths[i] for i in np.where(similarity_scores[0] >= similarity_threshold)[0]]
    #display_images(query_image_path, similar_image_paths)
    return similar_image_paths





def recommend_footwear_of_the_target_brand(img):

    from ultralytics import YOLO
    from skimage import io

    def get_footwear_brand_name(model, img):
        """Return a list which contains the brand names of the footwears which are present on the input image

        model: model used to the detect the brand names
        img: input image
        """
        class_ids = []
        results = model.predict(img)
        result = results[0]

        for box in result.boxes:
            class_ids.append(int(box.cls[0].item()))

        return class_ids
    model = YOLO("./IISEngine_required_file/Models/shoe_brand_detector.pt")
    target_class_ids = get_footwear_brand_name(model, img)

    labels_path = "./IISEngine_required_file/Data/Footwear_dataset/labels"
    images_path = "./IISEngine_required_file/Data/Footwear_dataset/images"
    target_images = []

    for file_path in os.listdir(labels_path):

        file_name = file_path[:-4]
        file_path = os.path.join(labels_path, file_path)
        file = open(file_path, "r")

        for line in file.readlines():
            if int(line[0]) in target_class_ids:
                target_images.append(os.path.join(images_path, file_name + '.jpg'))
                break

        file.close()

    return target_images 


def detect_objects_using_FasterRCNN(image, labels):
    

    net = cv2.dnn.readNetFromTensorflow('./IISEngine_required_file/Models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb', './IISEngine_required_file/Models/faster_rcnn_inception_v2_coco_2018_01_28/faster_rcnn_inception_v2_coco_2018_01_28.pbtxt')

    height, width, _ = image.shape
    net.setInput(cv2.dnn.blobFromImage(image, swapRB=True, crop=False))
    detections = net.forward()

    detected_labels = set()
    detected_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            if class_id < len(labels):
                detected_labels.add(labels[class_id])
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                detected_boxes.append((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
            else:
                print(f"Class ID {class_id} is out of range for the provided labels.")

    #, detected_labels, detected_boxes
    return image ,detected_labels, detected_boxes




def recognize_faces_in_images(query_image_path, images_folder):
    import cv2
    import numpy as np
    import os
    import dlib
    from imutils import face_utils
    from sklearn.metrics.pairwise import cosine_similarity
    Image_List = list()
    Image_name_List = list()

    prototxt_path = "./IISEngine_required_file/Models/face_detection_and_landmark_pred/deploy.prototxt.txt"
    caffemodel_path = "./IISEngine_required_file/Models/face_detection_and_landmark_pred/res10_300x300_ssd_iter_140000.caffemodel"
    face_net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    shape_predictor_path = "./IISEngine_required_file/Models/face_detection_and_landmark_pred/shape_predictor_68_face_landmarks.dat"
    landmark_predictor = dlib.shape_predictor(shape_predictor_path)

    # Function to detect faces and facial landmarks in an image
    def detect_faces_landmarks(image, face_net, landmark_predictor):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()
        faces = []
        landmarks = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.45:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = dlib.rectangle(left=int(startX), top=int(startY), right=int(endX), bottom=int(endY))
                shape = landmark_predictor(image, face)
                shape = face_utils.shape_to_np(shape)
                faces.append((startX, startY, endX, endY))
                landmarks.append(shape)
        return faces, landmarks

    # Function to extract SIFT features from facial landmarks
    def extract_sift_features(image, landmarks):
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is not None:
            descriptors = cv2.resize(descriptors, (80, 80)).flatten()
        else:
            descriptors = np.array([])
        return descriptors

    query_image = cv2.imread(query_image_path)
    query_faces, query_landmarks = detect_faces_landmarks(query_image, face_net, landmark_predictor)

    if len(query_faces) == 0:
        print("No faces found in the query image")
        return


    for query_face, query_landmark in zip(query_faces, query_landmarks):
        (startX, startY, endX, endY) = query_face
        cv2.rectangle(query_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        query_features = extract_sift_features(query_image, query_landmark)
        print("Query Image")
        #plt.imshow(query_image)
        #plt.show()
        if len(query_features) == 0:
            print("No SIFT features found in the query face")
            continue

        for filename in os.listdir(images_folder):
            if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
                image_path = os.path.join(images_folder, filename)
                image = cv2.imread(image_path)
                faces, landmarks = detect_faces_landmarks(image, face_net, landmark_predictor)
                for face, landmark in zip(faces, landmarks):
                    current_features = extract_sift_features(image, landmark)
                    if len(current_features) == 0:
                        continue
                    similarity = cosine_similarity([query_features], [current_features])[0][0]
                    print(f"Similarity with {filename}: {similarity}")
                    if np.round(similarity,2) > 0.62:
                        print(f"Match found in {filename}")
                        (startX, startY, endX, endY) = face
                        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        Image_List.append(image)
                        Image_name_List.append(filename)
                        #plt.imshow(image)
                        #plt.show()
    return Image_List , Image_name_List

def search_images_containing_text(query_text , images_folder ):
    import easyocr
    print("wait a few minute")
    matching_images = []

    def ocr_text(image):
        reader = easyocr.Reader(['fr'])
        result = reader.readtext(image)
        extracted_text = ' '.join([entry[1] for entry in result])
        return extracted_text

    def text_contains(text, substring):
        words = substring.split(" ")
        all_words_found = True
        for word in words:
            if word not in text:
                all_words_found = False
                break
        return all_words_found

    for filename in os.listdir(images_folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(images_folder, filename)
            image = cv2.imread(image_path)
            text = ocr_text(image)
            if text_contains(text.lower() , query_text.lower()) :
                matching_images.append(image_path)
    if matching_images:
        return matching_images
    else:
        print("No Match found")



def search_for_image_fragments(image_utilisateur, dossier_base_de_donnees):

    Image_List = list()
    Image_name_List = list()
    print("Image Query")

    img_utilisateur = cv2.imread(image_utilisateur, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()

    kp_utilisateur, des_utilisateur = sift.detectAndCompute(img_utilisateur, None)

    for nom_image in os.listdir(dossier_base_de_donnees):
        chemin_image = os.path.join(dossier_base_de_donnees, nom_image)
        img_base_with_color = cv2.imread(chemin_image)
        img_base = cv2.imread(chemin_image, cv2.IMREAD_GRAYSCALE)

        if img_base is not None:
            kp_base, des_base = sift.detectAndCompute(img_base, None)
            if des_base is not None:

                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des_utilisateur, des_base, k=2)
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

                #img_base_with_keypoints = cv2.drawKeypoints(img_base, kp_base, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                #print(nom_image,len(good_matches))

                if len(good_matches) > 30:
                    print("L'image", nom_image, "a été trouvée.")

                    src_pts = np.float32([kp_utilisateur[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_base[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    h, w = img_utilisateur.shape
                    corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    warped_corners = cv2.perspectiveTransform(corners, M)
                    cv2.polylines(img_base_with_color, [np.int32(warped_corners)], True, (0, 255, 0), 2)
                    Image_List.append(img_base_with_color)
                    Image_name_List.append(nom_image)
                    #plt.imshow(img_base_with_color)
                    #plt.show()
                else:
                    pass
                    #print("L'image", nom_image, "n'a pas été trouvée.")
            else:
                pass
                #print("Impossible de détecter les keypoints et les descripteurs pour l'image", nom_image)
        else:
            print("Impossible de charger l'image", nom_image)
            
    return Image_List , Image_name_List