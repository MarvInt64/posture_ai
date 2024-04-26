from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam
import numpy as np
import cv2
import time
import winsound
from sklearn.model_selection import train_test_split

def load_model_or_create():
    try:
        model = load_model('posture_model.keras')
        print('Model loaded from disk.')

    except Exception as e:
        # here is our neural network and its layers. it is a CNN (google for convolutional neural network)
        # 128x128 (we resize the camera image in the code later on. )for the input x3 for RGB
        print(f"Error loading model: {e}")

        inputs = Input(shape=(128, 128, 3))
        x = Conv2D(32, (3, 3), activation='relu')(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(2, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        print('New model created!')

    return model

model = load_model_or_create()

'''
def create_feature_map_model(model):
    # create a new model, which extracts the outputs of the conv2 layers.
    # we do that to display a feature map (optional)
    layer_outputs = [layer.output for layer in model.layers if isinstance(layer, Conv2D)]
    return Model(inputs=model.input, outputs=layer_outputs)

feature_map_model = create_feature_map_model(model)
''' 

def collect_data():
    cap = cv2.VideoCapture(0)
    data, labels = [], []

    print("Wait for the camera window, then type g (good) or b (bad) to rate your posture in the current frame. Type q to quit learning.")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (128, 128))

        # display the camera image
        cv2.imshow('frame', cv2.resize(frame, (640, 480)))

        # read keyboard input
        key = cv2.waitKey(1) & 0xFF

        # q for quit
        if key == ord('q'):
            break

        # g for good
        elif key == ord('g'):
            print("You said it is a good posture :)")
            # label it as good by adding a zero to labels array
            # and add the image frame to the data array
            labels.append(0)
            data.append(frame)

        # b for bad
        elif key == ord('b'):
            print("You said it is a bad posture :[")
            labels.append(1)
            data.append(frame)

    cap.release()
    cv2.destroyAllWindows()

    return np.array(data), np.array(labels)

def train_model(data, labels):
    # create array one_hot_labels with zeros. each row is a label and each column is a category (we have two... bad/good)
    one_hot_labels = np.zeros((labels.shape[0], 2))

    # np arrange Beispiele
    # np.arange(5)          Gibt ein Array zurück: [0, 1, 2, 3, 4]
    # np.arange(1, 5)       Gibt ein Array zurück: [1, 2, 3, 4]
    # np.arange(1, 10, 2)   Gibt ein Array zurück: [1, 3, 5, 7, 9]

    one_hot_labels[np.arange(labels.shape[0]), labels] = 1

    train_data, val_data, train_labels, val_labels = train_test_split(data, one_hot_labels, test_size=0.2)
    model.trainable = True
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=10, batch_size=8)
    model.save('posture_model.keras')
    print('Model trained and saved.')

def predict_posture():
    cap = cv2.VideoCapture(0)

    # set the camera resolution to 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # create a window with a specific name and size
    cv2.namedWindow('Setz-Dich-Ordentlich-Hin', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Setz-Dich-Ordentlich-Hin', 800, 600)
    cv2.setWindowProperty("Setz-Dich-Ordentlich-Hin", cv2.WND_PROP_TOPMOST, 1)

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        frame_resized = cv2.resize(frame, (128, 128))
        
        # let the model predict if you sit good or bad
        prediction = model.predict(np.array([frame_resized]))
        posture = 'Good' if prediction[0][0] > prediction[0][1] else 'Bad'

        frame = cv2.resize(frame, (640, 480))
        cv2.putText(frame, posture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Setz-Dich-Ordentlich-Hin', frame)

        # beep if good probability is lower than bad propability
        if prediction[0][0] < prediction[0][1]:
            winsound.Beep(int(np.interp(prediction[0][1], (0, 1), (37, 800))), 100)

        # set this to whatever if you dont want to get annoyed
        time.sleep(0.2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    mode = input('Choose mode: train or run\n')

    if mode == 'train':
        data, labels = collect_data()
        train_model(data, labels)

    elif mode == 'run':
        predict_posture()

    else:
        print('Invalid mode')
