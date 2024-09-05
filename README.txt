AttendEase.exe- Run program, make sure model and application_database.db in working directory.

Helper Files: Folder containing .py files that aren't necessary to run program but used. Also contains copy of image directories and training/validiation/testing directories used.
	openCv_facial_detection_RT.py- Takes frames of videos and sends it to faces_database, this file is used for making the dataset
	move_photos.py- Moves photos from faces_database to image directory of choosing, is hard-coded, see comments
	facial_recognition.py-Code to make the model/dataset using image directories
	directory_mover.py- Code to make training/validation/testing set from the image directories
	CSVS- Folder containing file to hold names of image directories
		lfw_allnames, includes all names of the LFW dataset along with the four group members with the number of images in each directory. LFW dataset not included as not used in final product except for this one csv.

MainCode: Folder includes the code for the main file
	AttendEase.py-code for the main program

logs: Folder that holds the logs of the program, including an example log file

Application_database.db: Database used for main program

faces_database.db: Database used for making image directories

haarcascade_frontalface_default.xml- Configuration file for haarcascade classifier 
	Source: https://github.com/kipr/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

csc450_1115_90_MW.h5- Model used for class that is preloaded in application_database.db

AttendEase.exe- .exe for main program, must have haarcascade_frontalface_default.xml,csc450_1115_90_MW.h5, and application_database in the same directory
