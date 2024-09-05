import tkinter as tk
from tkinter import ttk
import sqlite3
from functools import partial
import customtkinter
import CTkMessagebox
import datetime
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from collections import Counter
from datetime import date
from PIL import ImageTk, Image
from io import *
import cv2
import os
from multiprocessing import Process
import atexit
import random

processarray = []

#function to connect to db and submit query
def qdb(query, db):
    
    connection = sqlite3.connect(db)
    cursor = connection.cursor()
    
    cursor.execute(query)
    output = cursor.fetchall()
    
    connection.commit()
    connection.close()
    return output

#load the model for the class that is wanted
def load_model(classlabel):
    
    #classlabel is the label of the class that is currently selected
    querystr = f"SELECT name FROM {classlabel}"
    blob = qdb(querystr, 'application_database.db') 
    class_names = []
    for i in range(len(blob)):
        cur_name = blob[i][0]
        ''.join(cur_name.split()).lower()
        class_names.append(cur_name)
        
    class_names = sorted(class_names)
    print(class_names)
    #load the .h5 file
    model = keras.models.load_model(f"{classlabel}.h5")

    return model, class_names


#function that clusters the images together by (x,y) coordinates
def get_face(conn):

    cur = conn.cursor()
    #get the first face from the database
    cur.execute("SELECT id,x,y FROM students_detected LIMIT 1")
    blob = cur.fetchall()
    x = blob[0][1]
    y = blob[0][2]
    
    
    print(blob)
    #query all photos within 130 pixels of the first image taken
    string = f"SELECT frame, id FROM students_detected WHERE x BETWEEN {x-130} AND {x+130} AND y BETWEEN {y-130} AND {y+130}"
    cur.execute(string)
    blob = cur.fetchall()
    
    #delete all of the images that are from this one face
    for i in range(len(blob)):
        string = f"DELETE from students_detected WHERE id={blob[i][1]}"
        cur.execute(string)
        
    conn.commit()
    conn.close()
    return blob

#function that handles the start button
def play(classlabel):
    
    #get the length of the class by looking at the name of the database
    timeSTR = classlabel
    timeSTR = timeSTR.split("_")
    #for demo purposes cut by 3 and be in seconds rather than minutes.
    duration = int(timeSTR[2])
    duration = duration/3
    
    conn = sqlite3.connect('application_database.db')
    cursor = conn.cursor()
    model, classnames = load_model(classlabel)

    #drop table and create to clean out the students_detected table so that no previous classes images are held
    cursor.execute('''DROP TABLE IF EXISTS students_detected''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS students_detected (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, x INTEGER, y INTEGER, frame BLOB)''')
    conn.commit()

    # Intialize the video capture
    cap = cv2.VideoCapture(0)
    # Pull in the haarcascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    first_start = datetime.datetime.now()
    # Have the video capture last the length of the class
    while (datetime.datetime.now() - first_start).seconds < duration:
        # Capture images for x amount of time in seconds
        capture_duration = 20
        interval_seconds = 1/2
        start_time = datetime.datetime.now()

        while (datetime.datetime.now() - start_time).seconds < capture_duration:
            ret, frame = cap.read()

            # Check if the frame is not empty
            if not ret or frame is None:
                break
            
            #haar-cascade needs grayscaled image for detection only
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
            
            for (x, y, w, h) in faces:
                        
                if w < 150 or h < 150:
                # Resize the face if less than 150x150 pixels
                    face_resized = cv2.resize(frame[y:y+h, x:x+w], (512, 512))
                    _, img_encoded = cv2.imencode('.png', face_resized)

                else:
                    frame_to_crop = cv2.resize(frame[y:y+h, x:x+w], (512, 512))
                    _, img_encoded = cv2.imencode('.png', frame_to_crop)
                    
                # Upload data from the detection into database including blob image, timestamp, and (x, y) coords    
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cursor.execute('INSERT INTO students_detected (timestamp, x, y, frame) VALUES (?, ?, ?, ?)', (timestamp, int(x), int(y), img_encoded.tobytes()))
                conn.commit()

            cv2.imshow('Frame', frame)

            time.sleep(interval_seconds)

            # Break the loop if 'q' is pressed
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            
        
        sizeTable = int((qdb("SELECT COUNT(*) FROM students_detected", "application_database.db"))[0][0])
        if sizeTable > 0:
            print(sizeTable)
        
        #do facial recognition while images are still in the students_detected table
        while sizeTable > 0:
            
            facepredict(model, classnames, classlabel)
            if int((qdb("SELECT COUNT(*) FROM students_detected", "application_database.db"))[0][0]) == 0:
                break
        print("making switch enabled")
        

#facial prediction function            
def facepredict(model, names, classlabel):
    
    #make connection to database
    connection = sqlite3.connect('application_database.db')
    #get cluster of images
    blob = get_face(connection)
    predNames = []
    predProbs = []
    
    #ignore the cluster if less than 15 images
    if len(blob) < 15:
        
        return
    
    
    blobindices = []
    
    for i in range(len(blob)):
        
        #get each image one by one
        file_like = BytesIO(blob[i][0])
        img = Image.open(file_like)
        img = img.resize((512,512))
    
        #predict the classification of the face in frame
        pred_probability = model.predict(np.expand_dims(img, axis=0))
        pred_int = np.argmax(pred_probability, axis=-1)
        prediction = names[list(np.argmax(pred_probability, axis=-1).astype("int"))[0]]
        high_prob = pred_probability[0][pred_int.astype("int")]
    
        print(pred_probability)
        print(prediction)
        
        #confidence threshold, ignore image if less than 0.82
        if high_prob[0] < 0.82:
            continue
        
        predNames.append(prediction)
        predProbs.append(high_prob[0])
        
        #get the number of predictions by class
        occurence = Counter(predNames)
        #most common occuring name is the predicted individual
        weighted_pred = occurence.most_common(1)[0][0]
        blobindices.append((i, weighted_pred))
    
    #ignore cluster if most common prediction occurs less than 12 times
    try: 
        if occurence.most_common(1)[0][1] <  12:
            return
    except:
        return
    flag = False
    
    #get one image from the cluster of the predicted individual to save to database to display in UI
    while flag == False:
        choice = random.choice(blobindices)
        if choice[1] == occurence.most_common(1)[0][0]:
            indice = choice[0]
            insertFace = blob[indice][0]
        flag = True
    #set the student to attending
    querystr = f"UPDATE {classlabel} SET attending=1 WHERE name='{weighted_pred}'"
    qdb(query=querystr, db='application_database.db')
    connection = sqlite3.connect('application_database.db')
    #add image to table
    connection.execute((f"UPDATE {classlabel} SET img=? WHERE name=?"), (insertFace, weighted_pred))
    connection.commit()
    connection.close()
    return

#create thread for facial prediction in order for the recognition to work in the background so user can use the GUI
def multiThread(classlabel):
    
    classlabel = str(classlabel)
    #make process of play function
    p = Process(target=play, args=(classlabel,), daemon=True)
    processarray.append(p)
    p.start()
    return

class App:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Classroom Attendance")
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("dark-blue")
        
        #Top nav frame
        self.top_nav_frame = customtkinter.CTkFrame(root)
        self.top_nav_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 15))

        #query to find all the tables in the db
        rows = self.qdb("SELECT name FROM sqlite_master WHERE type='table';", 'application_database.db')

        #place all the rows into buttons in the top nav
        def populateTopNav(self):
            #first remove all widgets in the frame
            for child in self.top_nav_frame.winfo_children():
                child.destroy()
            #place the settings button before looping
            top_button1 = customtkinter.CTkButton(self.top_nav_frame, 
                                              text="Settings",
                                              font= ("Helvetica", 15), 
                                              command=self.openSettingsWindow)
            top_button1.pack(side=tk.LEFT , padx=10, pady=10)

            #now we place all the buttons given they arent some system tables 
            for i in range(len(rows)):
                if ((rows[i][0] != "sqlite_sequence") and (rows[i][0] != "students_detected")):
                    customtkinter.CTkButton(self.top_nav_frame, text=rows[i], command=partial(self.display_tables, rows[i][0])).pack(side=tk.LEFT , padx=10, pady=10)


        populateTopNav(self)


        #scrollbar
        self.container = customtkinter.CTkFrame(root)
        canvas = customtkinter.CTkCanvas(self.container)
        scrollbar = customtkinter.CTkScrollbar(self.container, command=canvas.yview)
        scrollable_frame = customtkinter.CTkFrame(self.container)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: self.container.configure(
                scrollregion=self.container.bbox("all")
            )
        )

        
        #vreating the start switch
        self.switch_var  = customtkinter.StringVar(value = "off")
        self.start_switch = customtkinter.CTkSwitch(self.top_nav_frame, 
                                            text="Start",
                                            variable=self.switch_var,
                                            onvalue="on",
                                            offvalue="off",
                                            font= ("Helvetica", 15),
                                            #function here
                                            command= self.switch)
        self.start_switch.pack(side=tk.RIGHT , padx=10, pady=10)
        
        #no class selected is default and will be changed when a button is clicked
        self.classLabel = customtkinter.CTkLabel(self.container, text="No Class Selected", font= ("Helvetica", 25, 'bold'))
        self.classLabel.pack()

        #static attendance label that wont change
        atendanceLabel = customtkinter.CTkLabel(self.container, text="Attendance", font= ("Helvetica", 18))
        atendanceLabel.pack()

        #set the styling for the treeview
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview.Heading", 
                        background = "white smoke",  
                        font=("Helvetica", 20))
        
        #treeview attributes
        columns = ('ID', 'Name', 'Attending')  # Adjust based on your table structure
        self.tree = ttk.Treeview(self.container, columns=columns, show='headings')

        #bind the double click on the treeview to our function handler
        self.tree.bind("<Double-1>", self.onDoubleClick)

        #populate the attributes
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=200) # Adjust column width as needed


        #place the treeview
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)

        self.container.pack(fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def onDoubleClick(self, event):
        #create a top level
        picwindow = customtkinter.CTkToplevel(root)
        picwindow.title('Student picture')
        picwindow.geometry("550x550")
        item = self.tree.selection()[0]

        #query for the image that was clicked on
        query = "SELECT img FROM " + self.classLabel.cget("text") + " WHERE name = '" + self.tree.item(item, "values")[1] + "';"
        image = self.qdb(query, "application_database.db")

        something = BytesIO(image[0][0])

        picture = Image.open(something)
        picture = picture.resize((1000,1000), Image.Resampling.LANCZOS)

        #place the image in the label
        imgtoplace = customtkinter.CTkImage(dark_image=picture, size=(512,512))
        label = customtkinter.CTkLabel(picwindow, width=550, height=550, image=imgtoplace, text="")
        label.pack()
    
        
    def outputToLog(self):
        #get the class code
        classCode = self.classLabel.cget("text")
        #so long as the label isnt no class selected
        if classCode != "No Class Selected":
            #query the db for the attending students and then write them all to a folder named logs
            names = self.qdb("SELECT * FROM " + classCode + " WHERE attending=1;", "application_database.db")
            print(names)
            path = "./logs"
            fullpath = os.path.abspath(path)
            file_name = classCode + "_" + str(date.today()) + ".txt"
            file = os.path.join(fullpath, file_name)
            if not os.path.exists(path):
                os.makedirs(path)
            f = open(file, 'a')
            f.write(classCode + "_" + str(date.today()) + "\n")
            for name in names:
                f.write(name[1] + "\n")
            f.close()
        else:
            #generate error message if no class is selected
            CTkMessagebox.CTkMessagebox(title="Error", message="Please select a class", icon="warning")

    def outputToLogatExit(self):
        #same function just has the second if clause to check if both the button is activated and we are closing the program
        classCode = self.classLabel.cget("text")
        if classCode != "No Class Selected":
            if self.switch_var.get() == 'on':
                names = self.qdb("SELECT * FROM " + classCode + " WHERE attending=1;", "application_database.db")
                print(names)
                path = "./logs"
                fullpath = os.path.abspath(path)
                file_name = classCode + "_" + str(date.today()) + ".txt"
                file = os.path.join(fullpath, file_name)
                print(file)
                if not os.path.exists(path):
                    os.makedirs(path)
                f = open(file, 'a')
                f.write(classCode + "_" + str(date.today()) + "\n")
                for name in names:
                    f.write(name[1] + "\n")
                f.close()

        
    def qdb(self, query, db):
        #queries the db with the passed query
        connection = sqlite3.connect(db)
        cursor = connection.cursor()
        cursor.execute(query)
        output = cursor.fetchall()
        connection.commit()
        connection.close()
        return output

    def cleanup(self):
        #this is our general cleanup to reset the attribute values for attending and img to 0 and null resepctively 
        label = self.classLabel.cget("text")
        if label != "No Class Selected":
            query = f"UPDATE " + label + " SET attending =0;"
            print(query)
            qdb(query, "application_database.db")
            query2 = f"UPDATE " + label + " SET img = NULL;"
            qdb(query2, "application_database.db")            
        else:
            return
    
    def addRemoveStudent(self):
        #remove the widgets in the right frame of settings
        for child in self.tempFrame.winfo_children():
            child.destroy()

        #query for the table names
        rows = self.qdb("SELECT name FROM sqlite_master WHERE type='table';", 'application_database.db')
        vals = []

        #query returns tuples and we have to extract values
        for i in range(len(rows)):
            vals.append(rows[i][0])
            
            
        #redefined inside of the function due to tkinter issues
        def qdb(query, db):
            connection = sqlite3.connect(db)
            cursor = connection.cursor()
            cursor.execute(query) 
            output = cursor.fetchall()
            connection.commit()
            connection.close()
            return output
        
        
        #function to populate the right frame based on the previous chosen values
        def popframe2(self):
            for child in frame2.winfo_children():
                child.destroy()
            if addRemove.get() == "Add": # if we want to add a student
                #student name label and entry
                customtkinter.CTkLabel(master=frame2, text="Student name:").pack(pady=10)
                stdName = customtkinter.CTkEntry(master=frame2)
                stdName.pack()
                #define the function to handle submitting the add or remove
                def submitBtn():
                    if addRemove.get() == "Add": #if we want to add we execute this query
                        query = "INSERT INTO " + classes.get() + " (name) VALUES " + "('" + stdName.get() + "')" + ";"
                        print(query)
                        qdb(query, "application_database.db")
                    else: #else we will execute this query
                        test = student.get()
                        query = "DELETE FROM " + classes.get() + " WHERE name = '" + test + "';"
                        qdb(query, "application_database.db")

                submit = customtkinter.CTkButton(master=frame2, text="Submit", command=submitBtn)
                submit.pack(pady=10)
            else: #else we want to remove and we need to handle it differently
                #find the names in the class chosen
                qclass = classes.get()
                query = "SELECT NAME FROM " + qclass + ";"
                names = qdb(query, "application_database.db")
                names2 =[]
                for i in range(len(names)):
                    names2.append(names[i][0])
                #values for the optionmenu for students to remove
                customtkinter.CTkLabel(master=frame2, text="Student to remove:").pack(pady=10)
                student = customtkinter.CTkOptionMenu(master=frame2, values=names2)
                student.pack()

    
                #redefine the function for the submit button because of tkinter issues
                def submitBtn():
                    if addRemove.get() == "Add":
                        query = "INSERT INTO " + classes.get() + " (name) VALUES " + self.stdName.get() + ";"
                        qdb(query, "application_database.db")
                    else:
                        test = student.get()
                        query = "DELETE FROM " + classes.get() + " WHERE name = '" + test + "';"
                        qdb(query, "application_database.db")

                submit = customtkinter.CTkButton(master=frame2, text="Submit", command=submitBtn)
                submit.pack(pady=10)

        
        #add or remove option menu
        addRemove = customtkinter.CTkOptionMenu(master=self.tempFrame, values=["Add", "Remove"], command=popframe2)
        addRemove.pack(pady=10)
        
        #frame to populate
        mainframe = customtkinter.CTkFrame(master=self.tempFrame)
        mainframe.pack()

        #label 1
        info1 = customtkinter.CTkLabel(master=mainframe, text="student to/from class:")
        info1.pack()

        #classes dropdown menu
        classes = customtkinter.CTkOptionMenu(master=mainframe, values=vals, command=popframe2)
        classes.pack(pady=10)

        #temp frame that we use to place widgets into 
        frame2 = customtkinter.CTkFrame(master=mainframe)
        frame2.pack()



    #pretty self explanatory
    mode = "dark"
    def themeMode(self):
        global mode
        if self.mode == "dark":
            customtkinter.set_appearance_mode("light")
            self.mode = "light"
        else:
            customtkinter.set_appearance_mode("dark")
            self.mode = "dark"
        self.newWindow.after(5, self.newWindow.lift)



    def addRemoveClass(self):
        #once again remove all widgets in the right temp frame
        for child in self.tempFrame.winfo_children():
            child.destroy()
        customtkinter.CTkLabel(self.tempFrame, text="Class Customization").pack() 

        #add or remove dropdown menu
        addRemove = customtkinter.CTkOptionMenu(master=self.tempFrame, values=["Add", "Remove"])
        addRemove.pack()

        #class name entry field
        classNameEntry = customtkinter.CTkEntry(master=self.tempFrame, placeholder_text="Enter Class Name...")
        classNameEntry.pack()

        #class time entry field
        timeEntry = customtkinter.CTkEntry(master=self.tempFrame, placeholder_text="Enter Class Time...")
        timeEntry.pack()
        
        #duration entry field
        durationEntry = customtkinter.CTkEntry(master=self.tempFrame, placeholder_text="Enter Class Duration in minutes")
        durationEntry.pack()

        #class day dropdown menu
        comboBoxDay = customtkinter.CTkComboBox(master=self.tempFrame, 
                                                values = ["MW", "MWF", "TTH", "M", "T", "W", "TH", "F"])
        comboBoxDay.pack()

        #submit button function
        def submit_class():
            class_name = classNameEntry.get()
            class_time = timeEntry.get()
            class_day = comboBoxDay.get()
            class_duration = durationEntry.get()            
            addorRemove = addRemove.get()

            #handle add or remove differently
            if addorRemove == "Add":
                query = "CREATE TABLE IF NOT EXISTS " + class_name + "_" + class_time + "_" + class_duration + "_" + class_day + " (id INTEGER PRIMARY KEY, name TEXT NOT NULL, attending INTEGER, img BLOB);"
                print(query)
                self.qdb(query, "application_database.db")
            else:
                query = "DROP TABLE IF EXISTS " + class_name + "_" + class_time + "_" + class_duration + "_" + class_day + ";"
                self.qdb(query, "application_database.db")
            rows = self.qdb("SELECT name FROM sqlite_master WHERE type='table';", 'application_database.db')

            
            #redefine populate top nav so that we can repopulate the top nav after creating a class and make it look more seamless
            def populateTopNav(self):
                for child in self.top_nav_frame.winfo_children():
                    child.destroy()
                top_button1 = customtkinter.CTkButton(self.top_nav_frame, 
                                              text="Settings",
                                              font= ("Helvetica", 15), 
                                              command=self.openSettingsWindow)
                top_button1.pack(side=tk.LEFT , padx=10, pady=10)
                for i in range(len(rows)):
                    if ((rows[i][0] != "sqlite_sequence") and (rows[i][0] != "students_detected")):
                        customtkinter.CTkButton(self.top_nav_frame, text=rows[i], command=partial(self.display_tables, rows[i][0])).pack(side=tk.LEFT , padx=10, pady=10)

                self.start_switch = customtkinter.CTkSwitch(self.top_nav_frame, 
                                                    text="Start",
                                                    variable=self.switch_var,
                                                    onvalue="on",
                                                    offvalue="off",
                                                    font= ("Helvetica", 15),
                                                    #function here
                                                    command= self.switch)
                self.start_switch.pack(side=tk.RIGHT , padx=10, pady=10)
                
            populateTopNav(self)
            

        # Submit button
        submitButton = customtkinter.CTkButton(master=self.tempFrame, text="Submit", command=submit_class)
        submitButton.pack()

    def cams(self):
        #delete all widgets on the right frame
        for child in self.tempFrame.winfo_children():
            child.destroy()
        #create the label to put images into
        camvideo = customtkinter.CTkLabel(self.tempFrame, text="")
        camvideo.pack()

        view = cv2.VideoCapture(0)
        #connect to the camera

        if (view.isOpened() == False):
            #unable to open camera
            camvideo.configure(text="Unable to open camera")
            return

        #define a function to then populate the label with frames pulled from the camera
        def video_stream():
            _, frame = view.read()
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            camvideo.imgtk = imgtk
            camvideo.configure(image=imgtk)
            camvideo.after(1, video_stream) 

        video_stream()
        return


    def openSettingsWindow(self):
        #if the settings menu is already open then we will only lift to the top instead of making another top level
        if hasattr(self, 'newWindow') and self.newWindow.winfo_exists():
            self.newWindow.lift()
            return

        self.newWindow = customtkinter.CTkToplevel(root)
        self.newWindow.title("Settings")
        self.newWindow.geometry("800x600")
        
        #left side navigation frame
        side_nav = customtkinter.CTkFrame(self.newWindow)
        side_nav.pack(anchor="nw", side="left")

        #all of our buttons
        themeBtn = customtkinter.CTkButton(side_nav, text="Dark/Light Theme", command=self.themeMode)
        themeBtn.pack(pady=20, padx=20)

        addRemoveClassBtn = customtkinter.CTkButton(side_nav, text="Add/Remove Class", command=self.addRemoveClass)
        addRemoveClassBtn.pack(pady=20, padx=20)
       
        addRemoveStudentBtn = customtkinter.CTkButton(side_nav, text="Add/Remove Student", command=self.addRemoveStudent)
        addRemoveStudentBtn.pack(pady=20, padx=20)

        outputTologBtn = customtkinter.CTkButton(side_nav, text="Output current class to log", command=self.outputToLog)
        outputTologBtn.pack(pady=20, padx=20)

        cameraView = customtkinter.CTkButton(side_nav, text="Camera View", command=self.cams)
        cameraView.pack(pady=20, padx=20)

        #right frame to put stuff in
        self.right_frame = customtkinter.CTkFrame(self.newWindow)
        self.right_frame.pack(anchor="center")

        #tempframe inside the right frame to put widgets into based on the buttons pressed
        self.tempFrame= customtkinter.CTkFrame(self.right_frame)
        self.tempFrame.pack()

        #after exiting the settings window we lift the original window
        self.newWindow.protocol("DELETE_WINDOW", self.closeSettingsWindow)
        self.newWindow.after(5, self.newWindow.lift)


    def closeSettingsWindow(self):
        # Calling when the settings window is closed
        self.newWindow.destroy()
        del self.newWindow

    def switch(self):
        #function that handles the start switch
        if self.switch_var.get() == 'on':
            #if we have just turned on the start switch
            label = self.classLabel.cget("text")
            if label != "No Class Selected":
                #and the selected class is not no class selected
                try: #make sure to try and except bc we dont know if they have the model needed or not
                    # print(label + ".h5")
                    keras.models.load_model(f"{label}.h5")
                except:
                    #if there is no model to be loaded we will throw an error
                    CTkMessagebox.CTkMessagebox(title="Error", message="Error with model contact admin", icon="warning")
                    self.switch_var.set('off')
                    return
                label = str(label)
                # print(label)
                #start the multithreading
                multiThread(label)
                #once complete call cleanup
                self.cleanup()
            else: #if it was no class selected we will just reset the switch button and throw an error
                self.switch_var.set('off')
                CTkMessagebox.CTkMessagebox(title="Error", message="Please select a class", icon="warning")
        else:
            #else we want to kill all child processes
            for process in processarray:
                process.kill()
            self.outputToLog()
            
            
    #populate the treeview with the queried data
    def display_tables(self, classid):
        self.classLabel.configure(text=classid)
    
        query = 'SELECT * FROM ' + classid

        rows = self.qdb(query, 'application_database.db')

        # Clear any previous data in the treeview
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Insert data into the treeview
        for row in rows:
            self.tree.insert('', 'end', values=row)
            
 

if __name__ == "__main__":
    root = customtkinter.CTk()
    app = App(root)
    
    #setup our exit handlers
    atexit.register(partial(app.cleanup))
    atexit.register(partial(app.outputToLogatExit))
    
    root.geometry("800x600")
    root.mainloop()
