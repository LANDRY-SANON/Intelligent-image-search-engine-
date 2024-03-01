# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 18:16:10 2024

@author: landry
"""

import tkinter as tk
from tkinter import filedialog
import cv2
import os
from PIL import Image, ImageTk
from tkinter import ttk
from IISEngine_utils import  search_similar_image_by_color_histogram,search_for_image_fragments ,search_images_containing_text, search_similar_image_by_Pretrained_Model ,recommend_footwear_of_the_target_brand,detect_objects_using_FasterRCNN , recognize_faces_in_images
from shutil import copyfile
from tkinter import messagebox
from Tooltip import Tooltip

def select_image(entry_widget, file_icon_label):
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg; *.jpeg; *.png")])
    if file_path:
        Return_Done_Icon(file_icon_label)
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, file_path)
        
    else : 
        Return_Not_Done_Icon(file_icon_label)
        
def select_folder(entry_widget,folder_icon_label):
    folder_path = filedialog.askdirectory()
    print(folder_path)
    if folder_path:
        Return_Done_Icon(folder_icon_label)
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, folder_path)
        
    else : 
        Return_Not_Done_Icon(folder_icon_label)
    

def show_selected_form(event):
    selected_feature = feature_combobox.get()
    if selected_feature:
        for feature, form in feature_forms.items():
            if feature == selected_feature:
                form.pack(fill="both", expand=True)
                #images = []  
                #display_images(images, image_frame)
            else:
                form.pack_forget()
                
def Return_Done_Icon(icon_label):

    folder_icon = Image.open("./static/done.png") 
    folder_icon = folder_icon.resize((20, 20), Image.LANCZOS)  
    folder_icon = ImageTk.PhotoImage(folder_icon)
    icon_label.config(image=folder_icon)
    icon_label.image = folder_icon
    
    
def Return_Not_Done_Icon(icon_label):

    folder_icon = Image.open("./static/notdone.png")  
    folder_icon = folder_icon.resize((20, 20), Image.LANCZOS)  
    folder_icon = ImageTk.PhotoImage(folder_icon)
    icon_label.config(image=folder_icon)
    icon_label.image = folder_icon
    
def display_images(images, frame , read_yet = False):
    print(len(images))
    for widget in frame.winfo_children():
        widget.destroy()

    canvas = tk.Canvas(frame, width=500)  
    canvas.pack(side="top", fill="both", expand=True)

    scrollbar = tk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
    scrollbar.pack(side="bottom", fill="x")

    canvas.configure(xscrollcommand=scrollbar.set)
    image_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=image_frame, anchor="nw")
    
    image_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    tk.Frame(image_frame).grid(row=0, column=0, padx=5, pady=40)

    for i, image_path in enumerate(images):
        
        if not read_yet :
            image = Image.open(image_path)
            image = image.resize((200, 200), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            label = tk.Label(image_frame, image=photo)
            label.photo = photo
            label.grid(row=3, column=i, padx=5)
        else:
            image_resized = cv2.resize(image_path, (200, 200), interpolation=cv2.INTER_LANCZOS4)
            image = Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
            #image = image_path.resize((100, 100), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            label = tk.Label(image_frame, image=photo)
            label.photo = photo
            label.grid(row=3, column=i, padx=5)

    # Update the scroll region when the canvas size changes
    canvas.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))
    
    
def save_images(Folder , Image_path_list_or_matrix , read_yet = False , names = False):
    print("ok")
    
    if not read_yet :
        for img_path in Image_path_list_or_matrix:
            img_filename = os.path.basename(img_path)
            img_extension = os.path.splitext(img_path)[1]
            destination_path = os.path.join(Folder, img_filename)
            copyfile(img_path, destination_path)
            print(img_filename)
            print(img_extension)
    if read_yet and names :
        for i,img in enumerate(Image_path_list_or_matrix):
            img_path = os.path.join(Folder, f"{names[i]}") 
            cv2.imwrite(img_path, img)
        


def on_button_click_detect_objects_using_FasterRCNN(database_dir, label):
    images_list = [filename for filename in os.listdir(database_dir) if filename.lower().endswith(".jpg") or filename.lower().endswith(".png")]
    detect_images_list = list()
    detect_images_name_list = list()
    for filename in images_list:
        image_path = os.path.join(database_dir, filename)
        image = cv2.imread(image_path)
        image = cv2.resize(image, None, fx=0.4, fy=0.4)
        detected_image, detected_labels, detected_boxes = detect_objects_using_FasterRCNN(image, classes)

        if label in detected_labels:
            for box in detected_boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(detected_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            detect_images_list.append(detected_image)
            detect_images_name_list.append(filename)
    return detect_images_list , detect_images_name_list

def on_download_clicked(database_dow_widget,Image_list  , read_yet = False , names = False):
    if database_dow_widget.get():
        print("ok")
        save_images(database_dow_widget.get() , Image_list , read_yet , names)
    else :
        messagebox.showerror("Error","Select folder where you want to save files." )
def on_find_similar_images_histogram_clicked():

    query_image_path_colhist = query_image_path_widget_colhist.get()
    database_dir_colhist = database_dir_widget_colhist.get()
    print(os.path.isfile(query_image_path_colhist),database_dir_colhist)
    if os.path.isfile(query_image_path_colhist) and os.path.isdir(database_dir_colhist) :
        messagebox.showinfo("Success","wait, this may take several minutes.")
        
        Image_list = search_similar_image_by_color_histogram(query_image_path_colhist, database_dir_colhist)
        if not Image_list :
            messagebox.showinfo("Null","No images found.")
        else :
            display_images(Image_list, image_frame_hist)
            database_dow_widget = tk.Entry(form_frame_hist)
            folder_icon_label = tk.Label(form_frame_hist)
            database_dow_button = tk.Button(form_frame_hist, text="Download-Select Folder to save ", command=lambda: [select_folder(database_dow_widget,folder_icon_label) , on_download_clicked(database_dow_widget,Image_list)])
            Tooltip(database_dow_button, "we've assigned the same names to the files with the original ones, in case you want to retrieve the images from your previous folder")
            database_dow_button.grid(row=5, column=1, padx=5, pady=5)
            folder_icon_label.grid(row=5, column=2, padx=5, pady=5)
    else :
        messagebox.showerror("Error","Error! You forgot to fill in some fields." )
    
def on_find_similar_images_PTM_clicked():
    query_image_path_PTM = query_image_path_widget_PTM.get()
    database_dir_PTM = database_dir_widget_PTM.get()
    if os.path.isfile(query_image_path_PTM) and os.path.isdir(database_dir_PTM) :
        messagebox.showinfo("Success","wait, this may take several minutes.")
        Image_list = search_similar_image_by_Pretrained_Model(query_image_path_PTM, database_dir_PTM)
        if not Image_list :
            messagebox.showinfo("Null","No images found.")
        else :
            display_images(Image_list, image_frame_PTM)
            database_dow_widget = tk.Entry(form_frame_PTM)
            folder_icon_label = tk.Label(form_frame_PTM)
            database_dow_button = tk.Button(form_frame_PTM, text="Download-Select Folder to save ", command=lambda: [select_folder(database_dow_widget,folder_icon_label) , on_download_clicked(database_dow_widget,Image_list)])
            Tooltip(database_dow_button, "we've assigned the same names to the files with the original ones, in case you want to retrieve the images from your previous folder")
            database_dow_button.grid(row=5, column=1, padx=5, pady=5)
            folder_icon_label.grid(row=5, column=2, padx=5, pady=5)
    else :
        messagebox.showerror("Error","Error! You forgot to fill in some fields." )

def on_detect_objects_using_FasterRCNN_clicked():
    database_dir_detfastrcnn = database_dir_widget_detfastrcnn.get()
    label_detfastrcnn = label_widget_detfastrcnn.get()
    if os.path.isdir(database_dir_detfastrcnn) :
        messagebox.showinfo("Success","wait, this may take several minutes.")
        Image_list , Image_name_list = on_button_click_detect_objects_using_FasterRCNN(database_dir_detfastrcnn, label_detfastrcnn)
        if not Image_list :
            messagebox.showinfo("Null","No images found.")
        else :
            display_images(Image_list, image_frame_fastrcnn,read_yet=True)
            database_dow_widget = tk.Entry(form_frame_fastrcnn)
            folder_icon_label = tk.Label(form_frame_fastrcnn)
            database_dow_button = tk.Button(form_frame_fastrcnn, text="Download-Select Folder to save ", command=lambda: [select_folder(database_dow_widget,folder_icon_label) , on_download_clicked(database_dow_widget,Image_list ,read_yet = True , names =Image_name_list )])
            Tooltip(database_dow_button, "we've assigned the same names to the files with the original ones, in case you want to retrieve the images from your previous folder")            
            database_dow_button.grid(row=5, column=1, padx=5, pady=5)
            folder_icon_label.grid(row=5, column=2, padx=5, pady=5)
    else :
        messagebox.showerror("Error","Error! You forgot to fill in some fields." )
        
def on_recognize_faces_in_images_clicked():
    query_image_path_facereg = query_image_path_widget_facereg.get()
    database_dir_facereg = database_dir_widget_facereg.get()
    if os.path.isfile(query_image_path_facereg) and os.path.isdir(database_dir_facereg) :
        messagebox.showinfo("Success","wait, this may take several minutes.")
        Image_list   , Image_name_list = recognize_faces_in_images(query_image_path_facereg, database_dir_facereg)
        if (not Image_list) and (not Image_name_list) :
            messagebox.showinfo("Null","No images found.")
        else :
            display_images(Image_list, image_frame_facereg,read_yet=True)
            database_dow_widget = tk.Entry(form_frame_facereg)
            folder_icon_label = tk.Label(form_frame_facereg)
            database_dow_button = tk.Button(form_frame_facereg, text="Download-Select Folder to save ", command=lambda: [select_folder(database_dow_widget,folder_icon_label) , on_download_clicked(database_dow_widget,Image_list ,read_yet = True , names =Image_name_list )])
            Tooltip(database_dow_button, "we've assigned the same names to the files with the original ones, in case you want to retrieve the images from your previous folder")            
            database_dow_button.grid(row=5, column=1, padx=5, pady=5)
            folder_icon_label.grid(row=5, column=2, padx=5, pady=5)
    else :
        messagebox.showerror("Error","Error! You forgot to fill in some fields." )
        
def on_detect_images_using_fragment_clicked():
    query_image_path_frag = query_image_path_widget_frag.get()
    database_dir_frag = database_dir_widget_frag.get()
    if os.path.isfile(query_image_path_frag) and os.path.isdir(database_dir_frag) : 
        messagebox.showinfo("Success","wait, this may take several minutes.")
        Image_list  , Image_name_list = search_for_image_fragments(query_image_path_frag, database_dir_frag)
        if (not Image_list) and (not Image_name_list) :
            messagebox.showinfo("Null","No images found.")
        else :
            display_images(Image_list, image_frame_frag,read_yet=True)
            database_dow_widget = tk.Entry(form_frame_frag)
            folder_icon_label = tk.Label(form_frame_frag)
            database_dow_button = tk.Button(form_frame_frag, text="Download-Select Folder to save ", command=lambda: [select_folder(database_dow_widget,folder_icon_label) , on_download_clicked(database_dow_widget,Image_list ,read_yet = True , names =Image_name_list )])
            Tooltip(database_dow_button, "we've assigned the same names to the files with the original ones, in case you want to retrieve the images from your previous folder")            
            database_dow_button.grid(row=5, column=1, padx=5, pady=5)
            folder_icon_label.grid(row=5, column=2, padx=5, pady=5)
    else :
        messagebox.showerror("Error","Error! You forgot to fill in some fields." )
        
def on_search_images_containing_text_clicked():
    query_text_ocr = query_text_widget_ocr.get()
    database_dir_ocr = database_dir_widget_ocr.get()
    if os.path.isfile(query_text_ocr) and os.path.isdir(database_dir_ocr) : 
        messagebox.showinfo("Success","wait, this may take several minutes.")
        Image_list = search_images_containing_text(query_text_ocr, database_dir_ocr)
        if not Image_list :
            messagebox.showinfo("Null","No images found.")
        else :
            display_images(Image_list, image_frame_ocr )
            database_dow_widget = tk.Entry(form_frame_ocr)
            folder_icon_label = tk.Label(form_frame_ocr)
            database_dow_button = tk.Button(form_frame_ocr, text="Download-Select Folder to save ", command=lambda: [select_folder(database_dow_widget,folder_icon_label) , on_download_clicked(database_dow_widget,Image_list)])
            Tooltip(database_dow_button, "we've assigned the same names to the files with the original ones, in case you want to retrieve the images from your previous folder")
            database_dow_button.grid(row=5, column=1, padx=5, pady=5)
            folder_icon_label.grid(row=5, column=2, padx=5, pady=5)
    else :
        messagebox.showerror("Error","Error! You forgot to fill in some fields." )
        
def on_recommand_footwear_clicked():
    query_image_rec = query_image_path_widget_rec.get()
    if os.path.isfile(query_image_rec) : 
        messagebox.showinfo("Success","wait, this may take several minutes.")
        Image_list = recommend_footwear_of_the_target_brand(query_image_rec)
        if not Image_list :
            messagebox.showinfo("Null","No images found.")
        else :
            display_images(Image_list, image_frame_rec )
    else :
        messagebox.showerror("Error","Error! You forgot to fill in some fields." )
        
    
with open("./IISEngine_required_file/Models/faster_rcnn_inception_v2_coco_2018_01_28/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
current_directory = os.path.dirname(os.path.realpath(__file__))
image_path = os.path.join(current_directory, "static", "curved6-1500h.jpg")

image = Image.open(image_path)
# Define the main Tkinter window
root = tk.Tk()
root.title("Intelligent Search Image Engine")
root.resizable(False, False)


image = ImageTk.PhotoImage(image)
image_label = tk.Label(root, image=image)
image_label.pack(fill="x")



canvas = tk.Canvas(root)
canvas.pack(side="left", fill="both", expand=True)

scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollbar.pack(side="right", fill="y")

canvas.configure(yscrollcommand=scrollbar.set)


scrollable_frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

# Define feature forms
feature_forms = {
    "Find Similar Images using histogram": tk.Frame(scrollable_frame),  # Replace with your form for Find Similar Images using histogram
    "Find Similar Images using PT model": tk.Frame(scrollable_frame),  # Replace with your form for Feature 2
    "Detect images using fragment": tk.Frame(scrollable_frame),
    "Recognize faces in images" : tk.Frame(scrollable_frame),
    "Detect objects using FasterRCNN" : tk.Frame(scrollable_frame),
    "Find Images that contain this word" : tk.Frame(scrollable_frame),
    "Recommand footwear" :  tk.Frame(scrollable_frame),

}

# Create a Combobox to select the feature
feature_combobox_label = ttk.Label(scrollable_frame, text="Select Feature to test :")
feature_combobox_label.pack(padx=30, pady=5)
feature_options = list(feature_forms.keys())
feature_combobox = ttk.Combobox(scrollable_frame, values=feature_options, state="readonly", width=37)
feature_combobox.pack(padx=30, pady=5)
feature_combobox.bind("<<ComboboxSelected>>", show_selected_form)

# Add widgets to the forms (replace with your widgets)

# Example widgets for Find Similar Images using histogram form


def AddFormPattern_QueryImage_Folder(name, text, command):
    # Create a frame for the widget inputs and buttons
    form_frame = tk.Frame(feature_forms[name])
    form_frame.grid(row=0, column=0, padx=5, pady=80)

    # Add labels and entry widgets for query image path and database directory
    query_image_path_widget_label = tk.Label(form_frame, text="Query Image Path:")
    query_image_path_widget_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
    query_image_path_widget = tk.Entry(form_frame)  # Set state to 'readonly'
    #query_image_path_widget.grid(row=0, column=1, padx=5, pady=5)
    
    file_icon_label = tk.Label(form_frame)
    select_image_button = tk.Button(form_frame, text="Select Image", command=lambda: select_image(query_image_path_widget, file_icon_label))
    select_image_button.grid(row=0, column=1, padx=5, pady=5)
    file_icon_label.grid(row=0, column=2, padx=5, pady=5)
    database_dir_widget_label = tk.Label(form_frame, text="Database Directory:")
    database_dir_widget_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
    database_dir_widget = tk.Entry(form_frame)
    #database_dir_widget.grid(row=1, column=1, padx=5, pady=5)
    folder_icon_label = tk.Label(form_frame)
    select_folder_button = tk.Button(form_frame, text="Select Folder", command=lambda: select_folder(database_dir_widget,folder_icon_label))
    select_folder_button.grid(row=1, column=1, padx=5, pady=5)
    folder_icon_label.grid(row=1, column=2, padx=5, pady=5)
    # Add button for finding similar images
    find_similar_images_histogram_button = tk.Button(form_frame, text=text, command=command)
    find_similar_images_histogram_button.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

    # Create a separate frame for displaying images
    image_frame = tk.Frame(feature_forms[name])
    image_frame.grid(row=0, column=1, padx=5, pady=5, sticky="e")
    
    

    return query_image_path_widget, database_dir_widget, image_frame  , form_frame



def AddFormPattern_Label_Folder(name,text,command) :
    form_frame = tk.Frame(feature_forms[name])
    form_frame.grid(row=0, column=0, padx=5, pady=80)
    
    label_widget_2_label = tk.Label(form_frame, text="Label:")
    label_widget_2_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
    label_options = classes  
    label_widget_2 = ttk.Combobox(form_frame, values=label_options, width=47, state="readonly")
    label_widget_2.grid(row=1, column=1, padx=5, pady=5)
    label_widget_2.current(0)
    
    database_dir_widget_2_label = tk.Label(form_frame, text="Database Directory:")
    database_dir_widget_2_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")
    database_dir_widget_2 = tk.Entry(form_frame)
    folder_icon_label = tk.Label(form_frame)
    database_dir_widget_2_button = tk.Button(form_frame, text="Select Folder", command=lambda: select_folder(database_dir_widget_2,folder_icon_label))
    database_dir_widget_2_button.grid(row=2, column=1, padx=5, pady=5)
    folder_icon_label.grid(row=2, column=2, padx=5, pady=5)
    detect_objects_using_FasterRCNN_button_2 = tk.Button(form_frame, text=text, command=command)
    detect_objects_using_FasterRCNN_button_2.grid(row=3, columnspan=2, padx=5, pady=5, sticky="ew")
    
    # Create a separate frame for displaying images
    image_frame = tk.Frame(feature_forms[name])
    image_frame.grid(row=0, column=1, padx=5, pady=5, sticky="e")
    
    
    return database_dir_widget_2 , label_widget_2 , image_frame , form_frame

def AddFormPattern_Text_Folder(name,text,command) :
    
    form_frame = tk.Frame(feature_forms[name])
    form_frame.grid(row=0, column=0, padx=5, pady=80)
    

    query_text_widget_3_label = tk.Label(form_frame, text="Query Text:")
    query_text_widget_3_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
    query_text_widget_3 = tk.Entry(form_frame, width=50)
    query_text_widget_3.grid(row=1, column=1, padx=5, pady=5)
    
    database_dir_widget_3_label = tk.Label(form_frame, text="Database Directory:")
    database_dir_widget_3_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")
    database_dir_widget_3 = tk.Entry(form_frame)
    folder_icon_label = tk.Label(form_frame)
    database_dir_widget_3_button = tk.Button(form_frame, text="Select Folder", command=lambda: select_folder(database_dir_widget_3,folder_icon_label))
    database_dir_widget_3_button.grid(row=2, column=1, padx=5, pady=5)
    folder_icon_label.grid(row=2, column=2, padx=5, pady=5)
    search_images_contaning_text_button_3 = tk.Button(form_frame, text=text, command=command)
    search_images_contaning_text_button_3.grid(row=3, columnspan=2, padx=5, pady=5, sticky="ew")
    
    image_frame = tk.Frame(feature_forms[name])
    image_frame.grid(row=0, column=1, padx=5, pady=5, sticky="e")
    return  query_text_widget_3 ,database_dir_widget_3 , image_frame, form_frame

def AddFormPattern_QueryImage(name,text,command) :
    form_frame = tk.Frame(feature_forms[name])
    form_frame.grid(row=0, column=0, padx=5, pady=80)
    
    query_image_path_widget_label_4 = tk.Label(form_frame, text="Query Image Path:")
    query_image_path_widget_label_4.grid(row=1, column=0, padx=5, pady=5, sticky="e")
    query_image_path_widget_4 = tk.Entry(form_frame)  
    file_icon_label = tk.Label(form_frame)
    select_image_button_4 = tk.Button(form_frame, text="Select Image", command=lambda: select_image(query_image_path_widget_4, file_icon_label))
    select_image_button_4.grid(row=1, column=1, columnspan=2, pady=5)
    file_icon_label.grid(row=1, column=3, padx=5, pady=5)
    
    search_images_contaning_text_button_4 = tk.Button(form_frame, text=text, command=command)
    search_images_contaning_text_button_4.grid(row=2,column=0, columnspan=2, padx=5, pady=5, sticky="ew")
    image_frame = tk.Frame(feature_forms[name])
    image_frame.grid(row=0, column=1, padx=5, pady=5, sticky="e")
    
    return  query_image_path_widget_4  , image_frame

 


query_image_path_widget_colhist, database_dir_widget_colhist , image_frame_hist , form_frame_hist = AddFormPattern_QueryImage_Folder("Find Similar Images using histogram","Find Similar Images using histogram",on_find_similar_images_histogram_clicked)    

query_image_path_widget_PTM, database_dir_widget_PTM ,image_frame_PTM , form_frame_PTM = AddFormPattern_QueryImage_Folder("Find Similar Images using PT model","Find Similar Images using PT model",on_find_similar_images_PTM_clicked)
query_image_path_widget_frag, database_dir_widget_frag ,image_frame_frag , form_frame_frag= AddFormPattern_QueryImage_Folder("Detect images using fragment","Detect images using fragment",on_detect_images_using_fragment_clicked)
query_image_path_widget_facereg, database_dir_widget_facereg ,image_frame_facereg , form_frame_facereg = AddFormPattern_QueryImage_Folder("Recognize faces in images","Recognize faces in images",on_recognize_faces_in_images_clicked)
database_dir_widget_detfastrcnn , label_widget_detfastrcnn , image_frame_fastrcnn , form_frame_fastrcnn = AddFormPattern_Label_Folder("Detect objects using FasterRCNN","Detect objects using FasterRCNN",on_detect_objects_using_FasterRCNN_clicked)
query_text_widget_ocr , database_dir_widget_ocr , image_frame_ocr , form_frame_ocr = AddFormPattern_Text_Folder("Find Images that contain this word","Find Images that contain this word",on_search_images_containing_text_clicked) 
query_image_path_widget_rec ,image_frame_rec = AddFormPattern_QueryImage("Recommand footwear","Recommand footwear",on_recommand_footwear_clicked) 




def open_url(url):
    import webbrowser
    webbrowser.open_new(url)




# Team member data (photo, description, LinkedIn link, GitHub link)
team_members = [
    {
        "name": "Landry SANON",
        "photo": os.path.join(current_directory, "static", "Landry.png"),
        "description": "I am interested in Machine Learning R&D, ML Ops and mathematics applied to AI (mathematical optimization and Operations research).",
        "linkedin": "https://www.linkedin.com/in/landry-sanon/",
        "github": "https://github.com/LANDRY-SANON"
    },
    {
        "name": "Ausni Kafando",
        "photo": os.path.join(current_directory, "static", "Ausni.jpg"),
        "description": "I have a passion for data visualization, analysis and interpretation, as well as for AI technologies.",
        "linkedin": "https://www.linkedin.com/in/ausni-kafando/",
        "github": "https://github.com/Ausni"
    },
    {
        "name": "Adrien TCHUEM",
        "photo": os.path.join(current_directory, "static", "Adrien.jpg"), 
        "description": "Passionate about AI, particularly in the medical sector. I regularly publish content on Linkedin about this subject so don't hesitate to follow me.",
        "linkedin": "https://www.linkedin.com/in/adrien-junior-tchuem-tchuente",
        "github": "https://github.com/AdrienJ0/"
    }
]




footer_frame = tk.Frame(scrollable_frame)
footer_frame.pack(side="bottom", fill="x", padx=10, pady=10, anchor="center")


for i, member in enumerate(team_members):
    if i == 1 :
        separator = ttk.Separator(scrollable_frame, orient="horizontal")
        separator.pack(side="bottom", fill="x" ,pady=50)
    photo_path = member["photo"]
    if os.path.exists(photo_path):
        photo = Image.open(photo_path)
        photo = photo.resize((100, 100), Image.LANCZOS)
        photo = ImageTk.PhotoImage(photo)
        photo_label = tk.Label(footer_frame, image=photo)
        photo_label.photo = photo  
        photo_label.grid(row=0, column=i, padx=50)

    name_label = tk.Label(footer_frame, text=member["name"], font=("Arial", 12, "bold"))
    name_label.grid(row=1, column=i, sticky="w", padx=85)
    description_label = tk.Label(footer_frame, text=member["description"], wraplength=180)
    description_label.grid(row=2, column=i, sticky="w", padx=60)

    linkedin_link = tk.Label(footer_frame, text="LinkedIn", fg="blue", cursor="hand2")
    linkedin_link.grid(row=3, column=i, sticky="w", padx=50)
    github_link = tk.Label(footer_frame, text="GitHub", fg="blue", cursor="hand2")
    github_link.grid(row=3, column=i, sticky="e", padx=50)

    linkedin_link.bind("<Button-1>", lambda e, link=member["linkedin"]: open_url(link))
    github_link.bind("<Button-1>", lambda e, link=member["github"]: open_url(link))


def update_scroll_region(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

# Bind the update_scroll_region function to the <Configure> event of the scrollable frame
scrollable_frame.bind("<Configure>", update_scroll_region)

# Run the Tkinter main loop
root.mainloop()


