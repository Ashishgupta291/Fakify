# version 8.0.0.0
# to do- check mail format make verification, log in with google, token system and premium
# to do- add real time processing
# to do- DONT SAVE BOTH VDO AND IMG FILES IN UPLOAD, 302 redirection disconnect manually while redirection

from flask import Flask, render_template, request, jsonify,redirect,url_for, send_from_directory ,session
import os
from werkzeug.utils import secure_filename

import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
from moviepy.editor import VideoFileClip, AudioFileClip

from flask_socketio import SocketIO
from datetime import datetime, timedelta
import base64

import face_recognition
import mysql.connector

from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired

print("<<<<<<<<<<<<<<<<<<<<<<<<starting>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
app = Flask(__name__)


app.config.from_pyfile(r"config.cfg")
mail = Mail(app)
s = URLSafeTimedSerializer('Thisisasecret!')

socketio = SocketIO(app)

appp = FaceAnalysis(name='buffalo_l')
appp.prepare(ctx_id=0, det_size=(640, 640))

swapper = insightface.model_zoo.get_model(r"inswapper_128.onnx", download=False, download_zip=False)

email_verf_time=3600 # 1 hour
watermark_text = 'Fakify'

class TemporaryStorage:
    def __init__(self):
        self.data = {}

    def store(self, socket_id, data):
        self.data[socket_id] = data
    
    def update(self, socket_id, target, val):
        self.data[socket_id][target] = val

    def retrieve(self, socket_id):
        return self.data.get(socket_id)

    def flush(self, socket_id):
        if socket_id in self.data:
            del self.data[socket_id]

storage = TemporaryStorage()

# Read the logo image
logo = cv2.imread(r"watermark_logo.png", cv2.IMREAD_UNCHANGED)
def add_logo_watermark(image):
    
    global logo, watermark_text

    # Get dimensions of the main image
    height, width = image.shape[:2]

    tok = height if height<width else width
    # Resize logo if necessary
    logo_width = int(tok * 0.08) 
    logo_height = int(logo.shape[0] * (logo_width / logo.shape[1]))
    logo = cv2.resize(logo, (logo_width, logo_height))

    # Get dimensions of the logo
    logo_height, logo_width = logo.shape[:2]
    
    x_offset = int(tok *0.04) 
    y_offset = height - logo_height - x_offset

    # put text 
    # Watermark text properties
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = int(tok *0.005) 
    font_color = (255,255,255)  
    font_thickness = int(tok *0.008) 
    
    image = cv2.putText(image, watermark_text, (x_offset+logo_width+int(tok *0.015),y_offset+int(logo_height*0.75)), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    # Check if the logo has an alpha channel (transparency)
    if logo.shape[2] == 4:
        # Split the logo into BGR and alpha channels
        b, g, r, a = cv2.split(logo)
        overlay_color = cv2.merge((b, g, r))
        mask = a
    else:
        overlay_color = logo
        mask = 255 * np.ones(logo.shape, logo.dtype)

    # Define the region of interest (ROI) on the main image
    roi = image[y_offset:y_offset + logo_height, x_offset:x_offset + logo_width]

    # Create an inverse mask of the logo
    mask_inv = cv2.bitwise_not(mask)

    # Black-out the area of the logo in the ROI
    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only the region of the logo from the logo image
    logo_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img_bg, logo_fg)
    image[y_offset:y_offset + logo_height, x_offset:x_offset + logo_width] = dst

    return image

def convert_images_to_base64(images):
    images_base64 = []
    # Convert each image to base64
    for image in images:
        # Convert the image to JPEG format
        _, buffer = cv2.imencode('.jpg', image)
        # Convert the image buffer to base64
        image_base64 = base64.b64encode(buffer)
        images_base64.append(image_base64.decode('utf-8'))
    return images_base64

# IT WILL DELETE 1 HOUR OLD PROCESSED FILES FROM STATIC FOLDER
# static folder will be cleaned, it stores processed files
def delete_old_files(directory_path, age_threshold_hours=1):
    try:
        # Get the current time
        current_time = datetime.now()

        # Iterate through files in the directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            # Get the file modification time
            file_modification_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            # Calculate the age of the file
            file_age = current_time - file_modification_time
            
            # Check if the file is older than the threshold
            if file_age.total_seconds() >= age_threshold_hours * 3600:
                # Delete the file
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

        print("Deletion process completed.")
    except Exception as e:
        print(f"Error deleting old files: {e}")


def extract_audio(video_path,socket_id):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    if audio_clip:
       storage.update(socket_id,'flaG', 1)
       audio_clip.write_audiofile("audio.mp3", codec='mp3')
       audio_clip.close()
    video_clip.close()

def merge_audio_with_video(video_path, processed_video_path,socket_id):
    video_clip = VideoFileClip(video_path)
    retrieved_data = storage.retrieve(socket_id)
    if retrieved_data['flaG']:
       audio_clip = AudioFileClip("audio.mp3")
       video_clip = video_clip.set_audio(audio_clip)    
       
    video_clip.write_videofile(processed_video_path, codec='libx264', audio_codec='aac')
    video_clip.close()
    if retrieved_data['flaG']:
        audio_clip.close()
        os.remove(audio_clip.filename)  # delete audio file
    print("=========Audio Done==========Audio Done==========Audio Done============")
    os.remove("temp.mp4")           # delete temporarily created audioless file
    

# Define the folder to store uploads #### REMOVE
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mkv', 'jpg', 'jpeg', 'png'}
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.secret_key= 'my_secret_key'
# MySQL connection configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Ashish@291',
    'database': 'mydb'
}


@app.route('/',methods=['POST','GET'])
def home():
  global email_verf_time
  try:
    user_id = session.get('user_id') # only email
    user_name = session.get('user_name')
    msg = request.args.get('arg1')
    color=''
    if msg == None:
       msg=""
    if user_id:
       session['user_id'] = user_id
       return render_template('deepfun.html', msg=msg ,userName = user_name) 
    if request.method == 'POST':
       # for login request 
       log_rag = request.form['log_rag']
       if log_rag == "login":
          user_id = request.form['id']
          password = request.form['password']

          #  create database mydb; use mydb;
          #  create table account(user_id varchar(255) primary key,password varchar(255),userName varchar(255)); # to do --------make more columns as acc_typ ENUM(alpha,beta,gama, 10),
          #  ALTER TABLE account ADD COLUMN creation_time DATETIME DEFAULT CURRENT_TIMESTAMP;
         
          connection = mysql.connector.connect(**db_config)
          cursor = connection.cursor(dictionary=True)
          cursor.execute('SELECT * FROM account WHERE user_id = %s AND password = %s', (user_id, password))
          account = cursor.fetchall()
          # Close the cursor and connection
          cursor.close()
          connection.close()
          if len(account)==0:
              print(f"Din't get any account:{account}")
              msg="Username or Password wrong !!"
              color = 'rgba(255, 0, 0, 0.589)'
              return render_template('login.html',msg=msg,color=color)

          elif (len(account)==1) and (account[0]['user_id'] == user_id): 
               
              print(f"got something in account:{account}")
              userName = account[0]['userName']
              session['user_id']= account[0]['user_id']
              session['user_name'] = userName
              return render_template('deepfun.html', userName = userName)
          
       elif log_rag == "signup":
          # for signup request 
          user_id = request.form['id']
          Create_password = request.form['Create']
          Confirm_password =  request.form['Confirm']
          #### to do---------------check mail format here-------------------------------
          
          if Create_password != Confirm_password:
             msg = "Password does not match !!" # dont change this string
             color = 'rgba(255, 0, 0, 0.589)'
             return render_template('login.html',msg=msg,color=color)
          
          Name = request.form['Name']
          data = {'user_id': user_id, 'password': Create_password, 'Name':Name}

          connection = mysql.connector.connect(**db_config)
          cursor = connection.cursor(dictionary=True)
          cursor.execute('SELECT * FROM account WHERE user_id = %s', (user_id,))
          account = cursor.fetchall()
          cursor.close()
          connection.close()

          if len(account)==0:

             token = s.dumps(data, salt='email-confirm')
             link = url_for('confirm_email', token=token, _external=True)

             # HTML content for the email body with a button
             html_body = """
             <html>
              <body>
                <p>Click the button below to verify your email:</p>
                <p><a href="{}"><button style="background-color: #4CAF50; /* Green */
                   border: none;
                   border-radius: 6.4px;
                   color: white;
                   padding: 15px 32px;
                   text-align: center;
                   text-decoration: none;
                   display: inline-block;
                   font-size: 16px;
                   margin: 4px 2px;
                   cursor: pointer;">Verify Email</button></a></p>
                </body>
             </html>
             """.format(link)
             mesg = Message('Email confirmation request from Fakify', sender='noreply.fakify@gmail.com', recipients=[user_id])
             
             # mesg.body = 'click on {} to verify.'.format(link)
             mesg.html = html_body
             mail.send(mesg)

             msg = "Check your email for verification" 
             color = '#008CBA'
          elif (len(account)==1) and (account[0]['user_id'] == user_id):
             msg = "User already exists !!" # dont change this string
             color = 'rgba(255, 0, 0, 0.589)'
             
          
          return render_template('login.html',msg=msg,color=color)           
             
    return render_template('login.html',msg=msg,color=color)
  except mysql.connector.Error as e:
    connection.rollback()
    print(f"Sorry !! Failed to login : {str(e)}")
    return render_template('login.html',msg=f"Sorry !! Failed to login" )


@app.route('/confirm_email/<token>')
def confirm_email(token):
    global email_verf_time
    try:
        data = s.loads(token, salt='email-confirm', max_age=email_verf_time) # 1 hour,# email will contain the mail id which you have bound with the token 
        
        print(f">>>>>>>>>>>>>>>>>>>>>>{data}<<<<<<<<<<") 
        
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)
        ## check if already a row with user_id
        cursor.execute('SELECT * FROM account WHERE user_id = %s', (data['user_id'],))
        account = cursor.fetchall()
        if len(account)==0:
           cursor.execute('INSERT INTO account (user_id, password, userName) VALUES (%s, %s, %s)', (data['user_id'], data['password'], data['Name']))
           connection.commit()
           cursor.close()
           connection.close()
        else:
           cursor.close()
           connection.close()
           return '<h1>The Token is expired!! you are already registered</h1>'
        cursor.close()
        connection.close()

        confirmation_html = """
        <html>
          <body>
            <h1>Thanks for verifying.</h1>
            <h1>You may now <a style="background-color: rgba(129, 29, 129, 0.804); padding:6px 8px; border-radius: 6.4px; text-decoration: none; color: white;" href="{}">login</a>.</h1>
          </body>
        </html>
        """.format(url_for('home'))
        return confirmation_html
              
    except SignatureExpired:
        return '<h1>The Token is expired! Register again</h1>'
    

@app.route('/forgot',methods=['POST','GET'])
def forgot():
    try:
        msg=''
        color = '#008CBA'
        if request.method == 'POST':
          user_id = request.form['id']
          Create_new_password = request.form['Create_new']
          if Create_new_password != request.form['Confirm_new']:
             msg = "Password does not match !!" 
             color = 'rgba(255, 0, 0, 0.589)'
             return render_template('forgot.html',msg=msg,color=color)
          
          connection = mysql.connector.connect(**db_config)
          cursor = connection.cursor(dictionary=True)
          cursor.execute('SELECT * FROM account WHERE user_id = %s', (user_id,))
          account = cursor.fetchall()
          cursor.close()
          connection.close()

          if len(account)==0:
             msg = "No account registered with this email" 
             color = 'rgba(255, 0, 0, 0.589)'
             return render_template('forgot.html',msg=msg,color=color)
             
          data= {'user_id': user_id, 'password': Create_new_password}
          token = s.dumps(data, salt='email-confirm')
          link = url_for('confirm_email_forgot', token=token, _external=True)

          # HTML content for the email body with a button
          html_body = """
             <html>
              <body>
                <p>Dear fakify user, we have received forget password request, click the button below to verify your email if it was you:</p>
                <p><a href="{}"><button style="background-color: #4CAF50; /* Green */
                   border: none;
                   border-radius: 6.4px;
                   color: white;
                   padding: 15px 32px;
                   text-align: center;
                   text-decoration: none;
                   display: inline-block;
                   font-size: 16px;
                   margin: 4px 2px;
                   cursor: pointer;">Verify Email</button></a></p>
                </body>
             </html>
          """.format(link)
          mesg = Message('Email confirmation request from Fakify for forget password', sender='noreply.fakify@gmail.com', recipients=[user_id])
             
          mesg.html = html_body
          mail.send(mesg)

          msg = "Check your email for verification" 
          color = '#008CBA'
          
        return render_template('forgot.html', msg=msg, color=color)  
    except mysql.connector.Error as e:
        connection.rollback()
        print(f"Sorry !! Failed to login : {str(e)}")
        return '***An error occured in forget route***'
    
@app.route('/confirm_email_toForget/<token>')
def confirm_email_forgot(token):
    global email_verf_time
    try:
        data = s.loads(token, salt='email-confirm', max_age=email_verf_time) # 1 hour,# email will contain the mail id which you have bound with the token 
        
        print(f">>>>>>>>>>>>>>>>>>>>>>{data}<<<<<<<<<<") 
        
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)
        ## check if already a row with user_id
        cursor.execute('UPDATE account SET password=%s WHERE user_id = %s', (data['password'],data['user_id'],))
        connection.commit()
        cursor.close()
        connection.close()

        confirmation_html = """
        <html>
          <body>
            <h1>Thanks for verifying. Password changed successfully.</h1>
            <h1>You may now <a style="background-color: rgba(129, 29, 129, 0.804); padding:6px 8px; border-radius: 6.4px; text-decoration: none; color: white;" href="{}">login</a>.</h1>
          </body>
        </html>
        """.format(url_for('home'))
        return confirmation_html
              
    except SignatureExpired:
        return '<h1>The Token is expired! Register again</h1>'

@app.route('/logout')
def logout():
    
    session.pop('user_id', None)
    session.pop('user_name', None)
    #msg="logged out successfuly!!"
    return redirect(url_for('home')) 

# when files upload clicked
@app.route('/processing', methods=['GET', 'POST'])
def process_files():
  
  user_id=session.get('user_id')
  user_name=session.get('user_name') # NOT IN USE BECAUSE SPA
  if user_id:  
    directory_to_clean = r"static"
    delete_old_files(directory_to_clean) # ATLEAST 1 HOUR OLD
    if request.method == 'POST':
        global swapper,appp
        
        # Check if the post request has the required files
        if 'videoFile' not in request.files or 'imageFile' not in request.files:
            return 'Error: Video and Image files are required.'
        
        socket_id = request.form.get('socket_id')
        print(f"client socket:{socket_id}")
        video_file = request.files['videoFile']
        image_file = request.files['imageFile']
        checkbox_value = request.form.get('Checkbox')
        print(f"checkbox:{checkbox_value}")

        # Check if the files have allowed extensions
        if not (video_file and allowed_file(video_file.filename)) or \
           not (image_file and allowed_file(image_file.filename)):
            return 'Error: Invalid file format. Supported formats are mp4, avi, mkv, jpg, jpeg, png.'
        
        # get processing mode image or vdo
        P_mode = 1 if video_file.filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mkv'} else 0
        print(f"mode:{P_mode}")
        ###### to do----------check acc_typ here-------------------------------------------------------------

        # Save the uploaded files to the 'uploads' folder WITH ORIGINAL NAME

        video_filename = secure_filename(video_file.filename)
        orig_vdo_filename = video_filename
        video_filename = socket_id +'_'+ video_filename
        image_filename = secure_filename(image_file.filename)
        orig_img_filename = image_filename
        image_filename = socket_id +'_'+ image_filename

        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)

        video_file.save(video_path) 
        image_file.save(image_path)
        
        socketio.emit('update_status', {'percentage': "Analizing Source file..."}, room=socket_id)
    #######___realtime deepfake with a image is very slow and computation costly 
        assert float('.'.join(insightface.__version__.split('.')[:2]))>=float('0.7')
        if appp is None:           #### dont load if already loaded
           appp = FaceAnalysis(name='buffalo_l')
           appp.prepare(ctx_id=0, det_size=(640, 640))
        
        img = cv2.imread(image_path) # source image

        faces = appp.get(img)
        faces = sorted(faces, key = lambda x : x.bbox[0]) # sorting left to right
        
        options=[]
        for item in faces:
             temp = item['bbox']  # Extracting the bounding box
             x_min, y_min, x_max, y_max = [int(b) for b in temp]
             # Slicing the image to get the face region
             face_region = img[y_min:y_max, x_min:x_max]
             options.append(face_region)
        source_options=convert_images_to_base64(options)
        if len(faces)<1:
            os.remove(video_path)
            os.remove(image_path)
            msg="No Face Detected in Source Image!!"
            #disconnect(socket_id)
            #socketio.close_room(room=socket_id)
            #socketio.emit('disconnect', room=socket_id)

            return redirect(url_for('home', arg1=msg)) # working properly because of ajax # CHECK WETHER NEW SOCKET IS CREATING
        ### RETURN ALL FACES TO GET SOURCE FACE FROM USER ### 
        
        user_data = {
            'video_filename': video_filename, # with socket_id
            'orig_vdo_filename': orig_vdo_filename,
            'image_filename': image_filename, # with socket_id
            'orig_img_filename' : orig_img_filename,
            'video_path': video_path, # with socket_id in name
            'image_path': image_path, # with socket_id in name
            'faces': faces,
            'checkbox_value': checkbox_value,
            'P_mode': P_mode,
            'index_source_face': None,
            'vd_face_list':[],
            'vd_f_encoded':[],
            'flaG':0  # it has audio or not (default no audio)

                }
        # store details with key of socket_id
        storage.store(socket_id, user_data)
        socketio.emit('update_status', {'percentage': "Source analysis done !"}, room=socket_id)
        return jsonify({"source_options": source_options, "nexturl":'/source' if checkbox_value is None else '/target'})
        #return render_template('deepfun.html',userName = user_name ,source_options = source_options, nexturl='/source' if checkbox_value is None else '/target')
        # assert len(faces)>=1   # one face atleast
    
    return redirect(url_for('home'))
  else:
    return redirect(url_for('home'))
  

# vd_face_list=[] # to be send to user  ....
# vd_f_encoded=[] # for matching purpose ....

# for analysing target files
@app.route('/source', methods=['POST'])
def vdo_faces():
  
  user_id=session.get('user_id')
  if user_id:  
    global appp
    
    if appp is None:           #### dont load if already loaded
        appp = FaceAnalysis(name='buffalo_l')
        appp.prepare(ctx_id=0, det_size=(640, 640))
    
    vd_face_list=[] 
    vd_f_encoded=[] 
    
    if request.method == 'POST':
        socket_id = request.form.get('socket_id')
        storage.update(socket_id,'index_source_face', int(request.form['index'])-1)
        retrieved_data = storage.retrieve(socket_id)
        print(f"client socket:{socket_id}")
        print(f"index of source is:{retrieved_data['index_source_face']}")
        socketio.emit('update_status', {'percentage': "Analizing Target file..."}, room=socket_id)

        # mode for vdo
        if retrieved_data['P_mode']:

          temp_cap = cv2.VideoCapture(retrieved_data['video_path'])
          total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
          
          ## take premium membership
          #if total_frames >= 200:
          #   msg="Buy Premium"
          #   os.remove(retrieved_data['video_path'])
          #   os.remove(retrieved_data['image_path'])
          #   storage.flush(socket_id)
          #   return redirect(url_for('home', arg1=msg)) # working properly because of ajax
          
          boost_fac=3 # 2 frames will be ignored alternatively
          frame_count = 0   # tracking
          count = 0 # missed faces due to recognition
          while temp_cap.isOpened() and frame_count != total_frames:
              success, image = temp_cap.read()
              
              # if image is None or not success: # missed
              #   print("FRAMES MISSED")  
              #   continue
              if image is None and not success: # missed
                print("FRAMES MISSED")  
                break
              
              if frame_count % boost_fac:
                frame_count+=1
                percentage_analized = (frame_count / total_frames) * 100
                print(f"Percentage analized: {percentage_analized:.2f}%")
                percentage_analized = str(int(percentage_analized)) + '%'+ " Analized"
                socketio.emit('update_status', {'percentage': percentage_analized}, room=socket_id)
                continue
              
              det = appp.get(image)
              #print("running")
              for item1 in det:
                  x_min, y_min, x_max, y_max = [int(b) for b in item1['bbox']]
                  # print(f"h,v:{x_max-x_min},{y_max-y_min}")
                  Padding_x = int((x_max-x_min)*0.2)
                  Padding_y = int((y_max-y_min)*0.2)
                  temp_image=image[(y_min-Padding_y) if (y_min-Padding_y)>0 else 0:(y_max + Padding_y) if (y_max + Padding_y)<int(temp_cap.get(4)) else int(temp_cap.get(4)) , (x_min - Padding_x) if (x_min - Padding_x)>0 else 0:(x_max + Padding_x) if (x_max + Padding_x)<int(temp_cap.get(3)) else int(temp_cap.get(3))]  #ready for rendering

                  face_cut = np.array(temp_image)  # ready for encoding ....
                  face_cut_encoded_list = face_recognition.face_encodings(face_cut)  # ....
                  if len(face_cut_encoded_list)==0:  # always 1 or 0 ....(no face or one face)
                      #cv2.imshow('face', temp_image)
                      count+=1 # ....
                      print(f"count is:{count}") # ....
                      continue # ....
                  face_cut_encoded = face_cut_encoded_list[0] # ONLY ONE FACE IN IMAGE # ....
                  

                #   least_tol_val=1
                #   for item2 in vd_f_encoded: # ....
                #       tol = face_recognition.face_distance([face_cut_encoded], item2)
                #       if tol< least_tol_val:
                #         least_tol_val = tol
                #   if least_tol_val <= 0.6: 
                #      break
                #   vd_face_list.append(temp_image) 
                #   vd_f_encoded.append(face_cut_encoded)

                  # or 
                  flag1=1 
                  for item2 in vd_f_encoded: # ....
                      if face_recognition.compare_faces([face_cut_encoded], item2)[0]: # match with faces # ....
                          flag1=0 # ....
                          break # ....
                  if(flag1): 
                      # encode and put into list
                      vd_face_list.append(temp_image) 
                      vd_f_encoded.append(face_cut_encoded) # ....
            
              # send msg vdo processing
              frame_count+=1
              percentage_analized = (frame_count / total_frames) * 100
              print(f"Percentage analized: {percentage_analized:.2f}%")
              percentage_analized = str(int(percentage_analized)) + '%'+ " Analized"
              socketio.emit('update_status', {'percentage': percentage_analized}, room=socket_id)

              # Break the loop if 'q' key is pressed
              # if cv2.waitKey(1) & 0xFF == ord('q'):
              #   break
          temp_cap.release()
          cv2.destroyAllWindows()
          # return vd_face_list #only if not empty otherwise msg
          print(f"number of faces detected in vdo:{len(vd_face_list)}")
          print(f"miss count is:{count}") # ....
          target_options = convert_images_to_base64(vd_face_list)
          socketio.emit('update_status', {'percentage': "Analysis completed"}, room=socket_id)
          if len(vd_face_list) <= 0:
            os.remove(retrieved_data['video_path'])
            os.remove(retrieved_data['image_path'])
            # delete associeted object
            storage.flush(socket_id)
            msg="No Face Detected in video!!"
          else:
            storage.update(socket_id,'vd_face_list', vd_face_list)
            storage.update(socket_id,'vd_f_encoded', vd_f_encoded)
          return jsonify({"target_options": target_options}) if len(vd_face_list) > 0 else redirect(url_for('home',arg1=msg))  # working properly because of ajax
        
        else:      

          img = cv2.imread(retrieved_data['video_path']) # target image
          det = appp.get(img)
          det = sorted(det, key = lambda x : x.bbox[0])
          options=[]
          for item in det:
             temp = item['bbox']  # Extracting the bounding box
             x_min, y_min, x_max, y_max = [int(b) for b in temp]
             # Slicing the image to get the face region
             face_region = img[y_min:y_max, x_min:x_max]
             options.append(face_region)
          target_options=convert_images_to_base64(options)
          if len(det)<1:
             os.remove(retrieved_data['video_path'])
             os.remove(retrieved_data['image_path'])
             # delete associeted object
             storage.flush(socket_id)
             msg="No Face Detected in Target Image!!"
             return redirect(url_for('home', arg1=msg)) # working properly because of ajax
          ### RETURN ALL FACES TO GET SOURCE FACE FROM USER ### 
          socketio.emit('update_status', {'percentage': "Analysis completed"}, room=socket_id)
          return jsonify({"target_options": target_options})
  else:
    return redirect(url_for('home'))        
        

@app.route('/target', methods=['POST'])
def process_vdo():
  
  user_id=session.get('user_id')
  if user_id:
    global  swapper
    
    if swapper is None:        #### dont load if already loaded
        swapper = insightface.model_zoo.get_model(r"inswapper_128.onnx", download=False, download_zip=False)
    
    if request.method == 'POST':
        socket_id = request.form.get('socket_id') if request.form.get('socket_id') else request.json.get('socket_id')
        retrieved_data = storage.retrieve(socket_id)
        if retrieved_data['checkbox_value'] is None:
             index_target_face = request.json.get('index') # a list of indices
             print(f"client socket:{socket_id}")
             print(f"indices of targets is:{index_target_face}")
             
        else:
             storage.update(socket_id,'index_source_face', int(request.form['index'])-1)
             print(f"client socket:{socket_id}")
             print(f"index of source is:{retrieved_data['index_source_face']}")

        retrieved_data = storage.retrieve(socket_id)
        source_face = retrieved_data['faces'][retrieved_data['index_source_face']] # face selected by user
        
        

        # vdo mode
        if retrieved_data['P_mode']:
          
          cap = cv2.VideoCapture(retrieved_data['video_path'])
          extract_audio(retrieved_data['video_path'],socket_id)
        
          fps = cap.get(cv2.CAP_PROP_FPS)
          total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

          ## take premium membership
          #if total_frames >= 200:
          #   msg="Buy Premium"
          #   os.remove(retrieved_data['video_path'])
          #   os.remove(retrieved_data['image_path'])
          #   storage.flush(socket_id)
          #   return redirect(url_for('home',arg1=msg)) # working properly because of ajax

          width = int(cap.get(3))
          height = int(cap.get(4))
          output_path = 'temp.mp4'
          frame_count = 0   #tracking

          fourcc = cv2.VideoWriter_fourcc(*'mp4v')
          out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
          

          while cap.isOpened() and frame_count != total_frames:
              success, image = cap.read()
              if not success:
                #continue
                break
              dfaces = appp.get(image)
              dfaces = sorted(dfaces, key = lambda x : x.bbox[0])

              #bbox = source_face['bbox']
              #bbox = [int(b) for b in bbox]
              cimg = image.copy()
              
              if retrieved_data['checkbox_value'] is None:
                for df in dfaces:
                  x_min, y_min, x_max, y_max = [int(b) for b in df['bbox']]
                  Padding_x = int((x_max-x_min)*0.2)
                  Padding_y = int((y_max-y_min)*0.2)
                  temp_image=image[(y_min-Padding_y) if (y_min-Padding_y)>0 else 0:(y_max+Padding_y) if (y_max+Padding_y)<height else height , (x_min-Padding_x) if (x_min-Padding_x)>0 else 0:(x_max+Padding_x) if (x_max+Padding_x)<width else width]  #ready for rendering
                  face_cut = np.array(temp_image)  # ready for encoding
                  face_cut_encoded_list = face_recognition.face_encodings(face_cut)
                  if len(face_cut_encoded_list)==0:
                      continue
                  face_cut_encoded = face_cut_encoded_list[0]
                  for target_face in index_target_face:
                      tf = retrieved_data['vd_f_encoded'][target_face]                 
                      match= face_recognition.compare_faces([face_cut_encoded],tf )
                      if match[0]:
                         cimg = swapper.get(cimg, df, source_face, paste_back=True)
                         # break
              else:
                for df in dfaces:
                  cimg = swapper.get(cimg, df, source_face, paste_back=True)
              
              # Add watermark to the frame
              cimg = add_logo_watermark(cimg)

              out.write(cimg)
              # key = cv2.waitKey(1) & 0xFF
              # if key == ord('q') or key == ord('Q'):
              #     break
    
              frame_count+=1
              percentage_processed = (frame_count / total_frames) * 100
              print(f"Percentage processed: {percentage_processed:.2f}%")
              percentage_processed = str(int(percentage_processed)) + '%' + " Processed"
              socketio.emit('update_status', {'percentage': percentage_processed}, room=socket_id)

          print("======Done==========Done==========Done============")
        
          socketio.emit('update_status', {'percentage': 'Audio Setup..'}, room=socket_id)

          cap.release()
          out.release()
          
          processed_video_filename = 'fakify_' +socket_id + '_' +os.path.splitext( retrieved_data['orig_img_filename'] )[0] + '_' + retrieved_data['orig_vdo_filename']
          
          processed_video_path = os.path.abspath(os.path.join(app.root_path, 'static', processed_video_filename))
          merge_audio_with_video("temp.mp4", processed_video_path, socket_id)
          socketio.emit('update_status', {'percentage': 'File Ready !!'}, room=socket_id)
          cv2.destroyAllWindows()
        
          os.remove(retrieved_data['video_path'])
          os.remove(retrieved_data['image_path'])
          # delete object
          storage.flush(socket_id)
          return render_template('download.html', path = "../static/"+ processed_video_filename)
        else:
          socketio.emit('update_status', {'percentage': 'Processing Target Image..'},room=socket_id)
          image = cv2.imread(retrieved_data['video_path'])

          dfaces = appp.get(image)
          dfaces = sorted(dfaces, key = lambda x : x.bbox[0])
          cimg = image.copy()
          if retrieved_data['checkbox_value'] is None:
            for i in index_target_face:
                cimg = swapper.get(cimg, dfaces[i], source_face, paste_back=True)
          else:
            for df in dfaces:
                cimg = swapper.get(cimg, df, source_face, paste_back=True)

          processed_video_filename = 'fakify_' + socket_id + '_' + os.path.splitext( retrieved_data['orig_img_filename'] )[0] + '_' + retrieved_data['orig_vdo_filename']
          
          processed_video_path = os.path.abspath(os.path.join(app.root_path, 'static', processed_video_filename))

          # Add watermark to the frame(logo+text)
          cimg = add_logo_watermark(cimg)

          cv2.imwrite(processed_video_path , cimg)
          print("Image saved successfully.")
          socketio.emit('update_status', {'percentage': 'File Ready !!'}, room=socket_id)
          os.remove(retrieved_data['video_path'])
          os.remove(retrieved_data['image_path'])
          # delete object
          storage.flush(socket_id)
          return render_template('download.html', path2 = "../static/"+ processed_video_filename)
  else:
    return redirect(url_for('home'))          


@socketio.on('connect')
def handle_connect():
    print(f'_________________________Client {request.sid} connected')

@socketio.on('disconnect')
def handle_disconnect():
    print(f'_________________________Client {request.sid} disconnected')

if __name__ == '__main__':
    # app.run(debug=True)
    socketio.run(app, debug=False, host='0.0.0.0')
