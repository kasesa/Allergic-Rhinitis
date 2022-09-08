from imutils import paths
import imutils
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import timeit
from datetime import datetime

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, VGG19, InceptionResNetV2, DenseNet121, ResNet50V2, ResNet101V2, Xception, InceptionV3
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from ImageCorrection import colorCorrect


class ARModel:
    def __init__(self):
        # Initialize data and labels
        # 데이터 및 레이블 초기화
        self.data = []
        self.labels = []
        self.meta = {}

    def cropImage(self, img):
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _,thresholded = cv2.threshold(grayscale, 0, 255, cv2.THRESH_OTSU)
        #cv2.imwrite("otsu.png", thresholded)
        bbox = cv2.boundingRect(thresholded)
        x, y, w, h = bbox
        #print(bbox)
        croppedImg = img[y:y+h, x:x+w]
        return croppedImg

    def addContours(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7,7), 0)

        edged = cv2.Canny(blur, 10, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
    
        #(cnts, _) = imutils.contours.sort_contours(cnts)
        cv2.drawContours(image, cnts, -1, (0,255,0), 1)
        return image

    def loadImages(self, path=r'C:\Users\cvpr\Documents\Bishal\Allergic Rhinitis\Dataset\rotate', plotType="all", crop = False, correctColor=False, 
                    contours=False, printImgDemo=False):
        print("[INFO]: Trying to Read the images from ", path)
        #  Configure the Image Location
        # 이미지 위치 구성하기
        imagePaths = list(paths.list_images(path))
        # Plot type is used only in title of plot image
        # Adding to metadata
        self.meta["dataInfo"] = plotType

        singleImagePrintLabel = []
        
        
        # Formatting data and labels
        for imagePath in imagePaths:
            # Extract the class label from file name and append to labels
            # 파일 이름에서 클래스 레이블을 추출하고 레이블에 추가함
            label = imagePath.split(os.path.sep)[-2]
            self.labels.append(label)
            
            # Load the image, swap color channels, and resize it to be a fixed 224x224 pixels while ignoring the aspect ratio
            # 이미지를 로드하고, 컬러 채널을 스왑하고, 가로 세로 비율을 무시하고 고정 224x224 픽셀로 크기를 조정함
            image = cv2.imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if crop:
                image = self.cropImage(image)
            image = cv2.resize(image, (224,224))

            # If Color Correction is required
            # Perform Color Correction
            if correctColor:
                image = colorCorrect(image)

            # Contours is added in image if required
            if contours:
                image = self.addContours(image)
        
            # Setting code to print image - one from each class
            if label not in singleImagePrintLabel:
                singleImagePrintLabel.append(label)
                if printImgDemo:
                    self.savePlot(image, label)

            # Append to data
            # 데이터에 추가
            self.data.append(image)
            # Adding to metadata
            self.meta["imageCount"] = len(self.data)
        print("Images found :", len(self.data))

    def prepareData(self):

        # Convert the data and labels to NumPy arrays while scaling the pixel intensities to the range [0,1]
        # 픽셀 강도를 [0,1] 범위로 조정하면서 데이터와 레이블을 NumPy 배열로 변환

        print("[INFO]: Preparing Data")

        self.data = np.array(self.data) / 255.0
        self.labels = np.array(self.labels)

        # Perform the one-hot encoding on the labels
        # 레이블에 대해 원핫 인코딩 수행
        self.lb = LabelBinarizer()
        self.labels = self.lb.fit_transform(self.labels)

    def setDataAugmentation(self, rotation=30, zoom=0.15, wShift=0.2, hShift=0.2, shear=0.15, hFlip=True):
        # Initialize the training data augmentation
        # 교육 데이터 억멘테이션 초기화
        self.trainAug = ImageDataGenerator(rotation_range=rotation, zoom_range=zoom, width_shift_range=wShift, height_shift_range=hShift,
		shear_range=shear, fill_mode="nearest", horizontal_flip=hFlip)
         # Adding to metadata
        self.meta["dataAugmentation"] = {}
        self.meta["dataAugmentation"]["rotation"] = rotation
        self.meta["dataAugmentation"]["zoom"] = zoom
        self.meta["dataAugmentation"]["wShift"] = wShift
        self.meta["dataAugmentation"]["hShift"] = hShift
        self.meta["dataAugmentation"]["shear"] = shear
        self.meta["dataAugmentation"]["hFlip"] = hFlip

        print("[INFO]: Augmenting Data with - ")
        print(self.meta["dataAugmentation"])
    
    def setPartition(self, testSize=0.20):
        # Partition the data into training and testing splits using 80% of the training data and the remaining 20% for testing
        # 교육 데이터의 80%, 테스트에 20%를 사용하여 데이터를 교육 및 테스트 분할로 분할
        (self.trainX, self.testX, self.trainY, self.testY) = train_test_split(self.data, self.labels, test_size=testSize, stratify=self.labels, random_state=42)
        # Adding to metadata
        self.meta["partitionRatio"] = str(int((1-testSize)*100))+ ":" +str(int(testSize*100))
        print("[INFO]: Patition Set to - ", self.meta["partitionRatio"])

    def setBaseModel(self, modelType="vgg16"):
        # Load the model network, ensuring the Head-FC layer sets are left off
        # Head-FC 레이어 세트가 포함되지 않도록 VGG16 네트워크를 로드한다
        if modelType == "vgg16":
            self.baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))
        
        elif modelType == "vgg19":
            self.baseModel = VGG19(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))
        
        elif modelType == "inception":
            self.baseModel = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))
        
        elif modelType == "xception":
            self.baseModel = Xception(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))

        elif modelType == "resnet50":
            self.baseModel = ResNet50V2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))

        elif modelType == "resnet101":
            self.baseModel = ResNet101V2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))
        
        elif modelType == "densenet":
            self.baseModel = DenseNet121(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))

        elif modelType == "inceptionResnet":
            self.baseModel = InceptionResNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))
        
        else:
            print("Model not found. Proceding with VGG16.")
            modelType = "vgg16"
            self.baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))
        # Adding to metadata
        self.meta["model"] = modelType
        print("[INFO]: Model Selected - ", modelType)

    def setHeadModel(self, dropoutRate = 0.5):
        # Construct the head model that will be placed on the top of the base model
        # 보디 모델의 맨 위에 배치할 헤드 모델 구성
        self.headModel = self.baseModel.output
        self.headModel = AveragePooling2D(pool_size=(4,4))(self.headModel)
        self.headModel = Flatten(name="flatten")(self.headModel)
        self.headModel = Dense(64, activation="relu")(self.headModel)
        self.headModel = Dropout(dropoutRate)(self.headModel)
        self.headModel = Dense(3, activation="softmax")(self.headModel)
        # Adding to metadata
        self.meta["dropoutRate"] = dropoutRate
    
    def initModel(self):
        # Place the Head-FC model on top of the Base model - This become the actual model that we will train
        # Head-FC 모델을 보디 모델 위에 배치한다. 이것이 우리가 교육할 실제 모델이 될 것이다.
        self.model = Model(inputs=self.baseModel.input, outputs=self.headModel)

        # Make sure that the basemodel layers will not be trained and only head model will be trained.
        # 보디 모델 레이어가 훈련되지 않고 헤드 모델만 훈련되는지 확인한다.
        for layer in self.baseModel.layers:
            layer.trainable = False
        print("[INFO]: Initializing Model")

    def setHyperParameters(self, learningRate = 1e-3, epochs = 100, batchSize = 8):
        # Set the hyper-parameters
        # 하이퍼 파라미터 설정
        # INIT_LR = 1e-3
        self.INIT_LR = learningRate
        self.EPOCHS = epochs
        self.BS = batchSize
        # Adding to metadata
        self.meta["learningRate"] = learningRate
        self.meta["epochs"] = epochs
        self.meta["batchSize"] = batchSize
        
        print("[INFO]: Hyperparameters Set")

    def compileModel(self, loss="binary_crossentropy"):
        # Compile the Model
        # 모델 컴파일
        opt = Adam(learning_rate=self.INIT_LR, decay=self.INIT_LR / self.EPOCHS)
        self.model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])
        print("[INFO]: Compiling Model")

    def startTraining(self):                                                        
        # Train the Network Model
        # 모델 교육
        print("[INFO] Model Training")
        start = timeit.default_timer()
        self.H = self.model.fit(
            self.trainAug.flow(self.trainX, self.trainY, batch_size=self.BS),
            steps_per_epoch=len(self.trainX) // self.BS,
            validation_data=(self.testX, self.testY),
            validation_steps=len(self.testX) // self.BS,
            epochs=self.EPOCHS)

        stop = timeit.default_timer()
        print('Total Training Time: ', stop - start) 
        # Adding to metadata
        self.meta["traingTime"] = stop - start

    def startTesting(self):
        # Make predictions on the testing set
        # 테스트 세트에서 예측한다
        print("Making Predictions on the Test Set")
        self.predIdxs = self.model.predict(self.testX, batch_size=self.BS)

    def evalModel(self):
        print("[INFO]: Model Evaluation")
        self.predIdxs = np.argmax(self.predIdxs, axis=1)
        print("Classification Report")
        print(classification_report(self.testY.argmax(axis=1), self.predIdxs, target_names=self.lb.classes_))

        # Compute Confusion Matrix and derrive raw, accuracy, sensitivity, specificity from it
        # 혼란 매트릭스
        cm= confusion_matrix(self.testY.argmax(axis=1), self.predIdxs)
        total = sum(sum(cm))
        acc = (cm[0,0] + cm[1,1] + cm[2,2]) / total

        sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        # show the confusion matrix, accuracy, sensitivity, and specificity
        # 혼란 매트릭스 보기
        print("Confusion Matrix and its Derrivatives")
        print(cm)
        print("acc: {:.4f}".format(acc))
        print("sensitivity: {:.4f}".format(sensitivity))
        print("specificity: {:.4f}".format(specificity))
        # Adding to metadata
        self.meta["accuracy"] = int(acc*100)

    def generatePlot(self, iter=1):
        # plot the training loss and accuracy
        # 플롯 그래프
        print("[INFO]: Plot Generation")
        N = self.EPOCHS
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, N), self.H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), self.H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), self.H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, N), self.H.history["val_accuracy"], label="val_acc")
        title = "AR-"+self.meta["model"]+"-"+self.meta["dataInfo"]+"-lr_"+str(self.meta["learningRate"])+"-dropout_"+str(self.meta["dropoutRate"])+"-acc_"+str(self.meta["accuracy"])
        #plt.title("Allergic Rhinitis-Xception-aligned-0.5d")
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        figName = "[iter-"+str(iter)+"]plot-" + datetime.now().strftime('%H-%M-%S')
        plt.savefig(figName)

    def savePlot(self, image, text = ""):
        plt.figure()
        plt.imshow(image)
        figName = "[custom]plot-" + text +"-"+datetime.now().strftime('%H-%M-%S')
        plt.savefig(figName)

    def crossValidate(self, path = r'C:\Users\cvpr\Documents\Bishal\Allergic Rhinitis\Dataset', modelType="inception", 
                    dropoutRate=0.5, batchSize=8, epochs=100, learningRate=1e-3, iter=2, dataType="all"):
        '''
        Cross Validation by picking one dataset as test and rest as train at one time.
        Only for Inception and Xception Models
        '''
        dirList = {}
        imagePaths = list(paths.list_images(path))

        for imagePath in imagePaths:
            dataset = imagePath.split(os.path.sep)[-3]
            if dataset not in list(dirList.keys()):
                dirList[dataset] = []
            dirList[dataset].append(imagePath)

        dirListKeys = list(dirList.keys())
        
        for data_i in range(0, len(dirListKeys)):
            testSetData = []
            trainSetData = []
            for data_j in range(0, len(dirListKeys)):
                if data_i == data_j:
                    testSetData = testSetData + dirList[dirListKeys[data_j]]
                else:
                    trainSetData = trainSetData + dirList[dirListKeys[data_j]]
            
            print("Cross Validation 1 - testSet=",dirListKeys[data_i])

            trainData = []
            trainLabels = []
            testData = []
            testLabels = []

            print("[INFO]: Trying to Read the images")
            for imagePath in trainSetData:
                label = imagePath.split(os.path.sep)[-2]
                trainLabels.append(label)
                image = cv2.imread(imagePath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224,224))
                # Append to data
                trainData.append(image)
            print("Train Images found :", len(trainData))

            for imagePath in testSetData:
                label = imagePath.split(os.path.sep)[-2]
                testLabels.append(label)
                image = cv2.imread(imagePath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224,224))
                # Append to data
                testData.append(image)
            print("Test Images found :", len(testData))


            
            print("[INFO]: Preparing Data")
            trainData = np.array(trainData) / 255.0
            trainLabels = np.array(trainLabels)

            lb = LabelBinarizer()
            trainLabels = lb.fit_transform(trainLabels)

            testData = np.array(testData) / 255.0
            testLabels = np.array(testLabels)

            lb = LabelBinarizer()
            testLabels = lb.fit_transform(testLabels)


            print("[INFO]: Augmenting Data - ")
            trainAug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
		                shear_range=0.15, fill_mode="nearest", horizontal_flip=True)

            print("[INFO]: Loading Model")
            if modelType == "inception":
                baseModel = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))
            elif modelType == "xception":
                baseModel = Xception(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))
            else:
                print("Model not supported.")
                return
            
            headModel = baseModel.output
            headModel = AveragePooling2D(pool_size=(4,4))(headModel)
            headModel = Flatten(name="flatten")(headModel)
            headModel = Dense(64, activation="relu")(headModel)
            headModel = Dropout(dropoutRate)(headModel)
            headModel = Dense(3, activation="softmax")(headModel)


            print("[INFO]: Initializing Model")
            model = Model(inputs=baseModel.input, outputs=headModel)
            for layer in baseModel.layers:
                layer.trainable = False

            print("[INFO]: Setting HyperParameters")
            INIT_LR = learningRate
            EPOCHS = epochs
            BS = batchSize
        
            print("[INFO]: Compiling Model")
            opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
            model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

            print("[INFO] Model Training")
            start = timeit.default_timer()
            H = model.fit(
                trainAug.flow(trainData, trainLabels, batch_size=BS),
                steps_per_epoch=len(trainData) // BS,
                validation_data=(testData, testLabels),
                validation_steps=len(testData) // BS,
                epochs=EPOCHS)

            stop = timeit.default_timer()
            print('Total Training Time: ', stop - start) 

            print("Making Predictions on the Test Set")
            predIdxs = model.predict(testData, batch_size=BS)


            print("[INFO]: Model Evaluation")
            predIdxs = np.argmax(predIdxs, axis=1)
            print("Classification Report")
            print(classification_report(testLabels.argmax(axis=1), predIdxs, target_names=lb.classes_))

            cm= confusion_matrix(testLabels.argmax(axis=1), predIdxs)
            total = sum(sum(cm))
            acc = (cm[0,0] + cm[1,1] + cm[2,2]) / total

            sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
            specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

            print("Confusion Matrix and its Derrivatives")
            print(cm)
            print("acc: {:.4f}".format(acc))
            print("sensitivity: {:.4f}".format(sensitivity))
            print("specificity: {:.4f}".format(specificity))


            print("[INFO]: Plot Generation")
            N = EPOCHS
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
            plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
            plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
            plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
            title = "AR-CV-"+modelType+"-"+dataType+"-"+dirListKeys[data_i]+"-lr_"+str(learningRate)+"-dropout_"+str(dropoutRate)+"-acc_"+str(int(acc*100))
            #plt.title("Allergic Rhinitis-Xception-aligned-0.5d")
            plt.title(title)
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend(loc="lower left")
            figName = "[iter-"+str(iter)+"]plot-" + datetime.now().strftime('%H-%M-%S')
            plt.savefig(figName)



            