from subprocess import call
import tkinter as tk
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageTk
from tkinter import ttk
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump
import tkinter as tk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import tkinter as tk
root = tk.Tk()
root.title("Sickle Cell Anemia Detection")


root.configure(background="black")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))

image = Image.open('s1.jpg')

image = image.resize((w, h))

background_image = ImageTk.PhotoImage(image)

background_image=ImageTk.PhotoImage(image)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=100, y=0) #, relwidth=1, relheight=1)

#img=ImageTk.PhotoImage(Image.open("s1.jpg"))

#img2=ImageTk.PhotoImage(Image.open("s2.jpg"))



logo_label=tk.Label()
logo_label.place(x=0,y=0)

x = 1




  # , relwidth=1, relheight=1)
lbl = tk.Label(root, text="Sickle Cell Anemia Detection", font=('times', 35,' bold '), height=1, width=62,bg="purple",fg="white")
lbl.place(x=0, y=0)
# _+++++++++++++++++++++++++++++++++++++++++++++++++++++++


    


def Model_Training():
    
    data = pd.read_csv("C:/Code/Code/train.csv")

    
    data = data.dropna()

    
    print("Column Names:", data.columns)
    
    data.columns = data.columns.str.strip()

   
    print("Column Names:", data.columns)

    # Feature Selection => Manual
    if 'label' in data.columns:
        x = data.drop(['label'], axis=1)  # Fix axis=1 to drop along columns
        y = data['label']

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=140)

        # Create and train the SVM classifier
        svcclassifier = SVC(kernel='linear')
        svcclassifier.fit(x_train, y_train)

        # Make predictions
        y_pred = svcclassifier.predict(x_test)

        # results
        print("=" * 40)
        print("Classification Report:", classification_report(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred) * 100)

        # Display results in tkinter GUI (assuming 'root' is defined somewhere in your code)
        ACC = accuracy_score(y_test, y_pred) * 100
        repo = classification_report(y_test, y_pred)

        label4 = tk.Label(root, text=str(repo), width=45, height=10, bg='khaki', fg='black', font=("Tempus Sanc ITC", 14))
        label4.place(x=305, y=200)

        label5 = tk.Label(root, text="Accuracy: {:.2f}%\nModel saved as svm.joblib".format(ACC),
                          width=45, height=3, bg='khaki', fg='black', font=("Tempus Sanc ITC", 14))
        label5.place(x=305, y=420)

        # Save the trained model
        dump(svcclassifier, "svm.joblib")
        print("Model saved as svm.joblib")
        
        # # Plot the accuracy graph
        # C_values = [0.1, 1, 10, 100]
        # accuracies = []

        # for C in C_values:
        #     svm_model = SVC(kernel='linear', C=C)
        #     svm_model.fit(x_train, y_train)
        #     y_pred = svm_model.predict(x_test)
        #     accuracy = accuracy_score(y_test, y_pred)
        #     accuracies.append(accuracy)

        # # Plotting
        # fig = Figure(figsize=(8, 6))
        # ax = fig.add_subplot(111)
        # ax.plot(C_values, accuracies, marker='o')
        # ax.set_title('Accuracy')
        # ax.set_xlabel('label')
        # ax.set_ylabel('Accuracy')
        # ax.grid(True)

        # # Display the plot in Tkinter window
        # canvas = FigureCanvasTkAgg(fig, master=root)
        # canvas_widget = canvas.get_tk_widget()
        # canvas_widget.place(x=900, y=200)
        # canvas.draw()

    else:
        print("Error: 'label' column not found in the DataFrame.")



def Model_Training1():
    
    data = pd.read_csv("C:/Code/Code/train.csv")

    
    data = data.dropna()

    
    data.columns = data.columns.str.strip()

    
    print("Column Names:", data.columns)

    # Feature Selection => Manual
    if 'label' in data.columns:
        x = data.drop(['label'], axis=1)  # Fix axis=1 to drop along columns
        y = data['label']

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

        # Create and train the Random Forest classifier
        rf_classifier = RandomForestClassifier(n_estimators=10, random_state=140)
        rf_classifier.fit(x_train, y_train)

        # Make predictions
        y_pred = rf_classifier.predict(x_test)

        # Display results
        print("=" * 40)
        print("Classification Report:", classification_report(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred) * 100)

        # Display results 
        ACC = accuracy_score(y_test, y_pred) * 100
        repo = classification_report(y_test, y_pred)

        label4 = tk.Label(root, text=str(repo), width=45, height=10, bg='khaki', fg='black', font=("Tempus Sanc ITC", 14))
        label4.place(x=405, y=200)

        label5 = tk.Label(root, text="Accuracy: {:.2f}%\nModel saved as rf.joblib".format(ACC),
                          width=45, height=3, bg='khaki', fg='black', font=("Tempus Sanc ITC", 14))
        label5.place(x=405, y=420)

        # Save the trained model
        dump(rf_classifier, "rf.joblib")
        print("Model saved as rf.joblib")
    else:
        print("Error: 'label' column not found in the DataFrame.")


    

    
    


def window():
    root.destroy()



button3 = tk.Button(root, foreground="white", background="#152238", font=("Tempus Sans ITC", 14, "bold"),
                    text="Model_SVM", command=Model_Training, width=15, height=2)
button3.place(x=5, y=200)

button5 = tk.Button(root, foreground="white", background="#152238", font=("Tempus Sans ITC", 14, "bold"),
                    text="Model_RF", command=Model_Training1, width=15, height=2)
button5.place(x=5, y=300)



exit = tk.Button(root, text="Exit", command=window, width=15, height=2, font=('times', 15, ' bold '),bg="red",fg="white")
exit.place(x=5, y=600)

root.mainloop()

'''+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''