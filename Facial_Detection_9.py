import cv2
import mediapipe as mp
import math 
import pandas as pd

prod_df = pd.read_csv('hairproducts.csv')

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Load the input image
img = cv2.imread('diamond.png')

with mp_face_mesh.FaceMesh(max_num_faces=2) as face_mesh:
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        # Draw the facial landmarks on the image
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )
            
            # 10 is top of the head
            # 105 is the forehead left
            # 334 is the forehead right
            # 94 is the lowest nose
            # 152 is the chin
            # 215 is the left cheekbone
            # 716 is the right cheekbon
            # 17 is the lower lip
            # 0 is the upper lip

            # 0 - 0,
            # 10 - 1,
            # 17 - 2,
            # 94 - 3,
            # 105 - 4,
            # 118 - 5,
            # 148 - 6,
            # 152 - 7,
            # 215 - 8,
            # 334 - 9,
            # 347 - 10,
            # 377 - 11,
            # 416 - 12

            required_list_x = []
            required_list_y = []
            # Print the landmark index and position in the terminal
            for id, landmark in enumerate(face_landmarks.landmark):
                # print(f"Landmark {id}: ({landmark.x}, {landmark.y}, {landmark.z})")
                if id in (10, 152, 105, 334,215,94,0,17,416, 118, 347, 377, 148):
                    # required_list_x.append(id)
                    required_list_x.append(landmark.x)
                    # required_list_y.append(id)
                    required_list_y.append(landmark.y)
            
            zipped = list(zip(required_list_x, required_list_y))

            upper_lip = zipped[0]
            top_head = zipped[1]
            lower_lip = zipped[2]
            lower_nose = zipped[3]
            left_forehead = zipped[4]
            left_bone = zipped[5]
            left_chin = zipped[6]
            right_chin = zipped[11]
            right_bone = zipped[10]
            chin = zipped[7]
            left_cheek = zipped[8]
            right_forehead = zipped[9]
            right_cheek = zipped[12]

            # calculate the distances
            ll2c = math.dist(lower_lip, chin)
            lc2rc = math.dist(left_cheek, right_cheek)
            ln2ul = math.dist(lower_nose, upper_lip)
            lf2rf = math.dist(left_forehead, right_forehead)
            uh2c = math.dist(top_head, chin)
            lb2rb = math.dist(left_bone, right_bone)



            # print('the length of leftbone to rightbone  is : ', lb2rb)
            # print('the length of lowerlip to chin  is : ', ll2c)
            # print('the length of leftchin to rightchin is : ', lc2rc)
            # print('the length of lowernose to upperlip is : ', ln2ul)
            # print('the length of leftforehead to rightforehead is : ', lf2rf)
            # print('the length of upperhead to chin is : ', uh2c)

            atr_val = (lc2rc/lb2rb)/lf2rf
            print()
            print('this is the value we have got for the metric')
            print(atr_val)
            print()
            

            # getting the face type from the image using if else condition
            if atr_val < 3.342:
                print('diamond')
                print('Beard Recommendation for your face type: ')
                print('Goatee')
                print('Royal Beard')
                print('Full beard')
            elif atr_val > 3.344 and atr_val < 3.518:
                print('triangle face')
                print('Beard Recommendation for your face type: ')
                print('Goatee')
                print('Royal Beard')
                print('Petite Goatee')
                print('short beard')
            elif atr_val > 3.518 and atr_val < 3.5705:
                print('square')
                print('Beard Recommendation for your face type: ')
                print('short boxed beard')
                print('Horseshoe')
                print('chin strap')
                print('Full beard')
                print('Balbo')
            elif atr_val > 3.5705 and atr_val < 3.62:
                print('oval')
                print('Beard Recommendation for your face type: ')
                print('Petite Goatee')
                print('short boxed beard')
                print('Horseshoe')
            else:
                print('round face')
                print('Beard Recommendation for your face type: ')
                print('Goatee')
                print('short boxed beard')
                print('chin strap')
                print('short beard')
                print('Petite Goatee')
                print('Balbo')

            
            # recommendation system

            # first randomly choose a product
            print()
            print(prod_df['Product'])
            print()
            user_product = input('Please enter a product from our wide range of choices: ')


            from sklearn.feature_extraction.text import TfidfVectorizer

            # removing the unnecessary words from the vectorizer
            tfv = TfidfVectorizer(min_df=3, max_features=None,
                    strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                    ngram_range=(1,3),
                    stop_words = 'english')

            # Filling NaNs with empty string
            prod_df['Use'] = prod_df['Use'].fillna('')

            # creating the sparse matrix
            tfv_matrix = tfv.fit_transform(prod_df['Use'])

            # print(tfv_matrix)
            # print(tfv_matrix.shape)

            # computing the sigmoid kernel
            from sklearn.metrics.pairwise import sigmoid_kernel
            sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

            # print(sig[0])

            # reverse mapping of indices and movie titles
            indices = pd.Series(prod_df.index, index=prod_df['Product']).drop_duplicates()
            # print(indices)


            # giving recommendation
            def give_rec(title, sig=sig):
                # get the index corresponding to 'Product'
                idx = indices[title]

                # get the pairwise similarity scores
                sig_scores = list(enumerate(sig[idx]))

                # sorting the products
                sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)  # sorting it in the ascending order

                # Scores of the 3 most similar products
                sig_scores = sig_scores[1:4]

                # product indices
                prod_indices = [i[0] for i in sig_scores]

                # Top 3 similar products
                return prod_df['Product'].iloc[prod_indices]

            rec = give_rec(user_product)
            print()
            print('More Products Like this: ')
            print()
            print(rec)


    # Resize the image to be bigger
    img = cv2.resize(img, (960, 720))

    # Display the output image
    cv2.imshow("Output", img)
    cv2.waitKey(0)

