
import cv2
import numpy as np
import smtplib
import playsound
import threading 

Alarm_Status = False
Email_Status = False
Fire_Reported = 0

def play_alarm_sound_function():
	while True:
		playsound.playsound('alarm-sound.mp3',True)

# Tenemos que cambiar video por el video que se vaya grabando con la SP32
video = cv2.VideoCapture(0) # If you want to use webcam use Index like 0,1.

# Mientra vaya cogiendo imágenes del video (si existen, si no existen para):
while True:
    (grabbed, frame) = video.read()
    if not grabbed:
        break

    # Guarda el frame (la imagen actual) en una variable
    frame = cv2.resize(frame, (960, 540))

    # Pone el vide un poco borroso para sacar mejor los colores
    blur = cv2.GaussianBlur(frame, (21, 21), 0)
    # Convierte la imagen a un archivo de color
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Establece los colores para las temperaturas bajas y altas
    lower = [18, 50, 50]
    upper = [35, 255, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # Creamos el detector de temperaturas altas
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(frame, hsv, mask=mask)

    # Comprobamos la cantidad de píxeles de temperaturas altas se han encontrado
    no_red = cv2.countNonZero(mask)
    if int(no_red) > 15000:
        # Si supera cierta cantidad aumentamos la cantidad de fuegos detectados
        Fire_Reported = Fire_Reported + 1

    cv2.imshow("output", output)

    # Si se ha detectado un fuego suena una alarma
    if Fire_Reported >= 1:
    	if Alarm_Status == False:
            # Si detectamos un fuego deberíamos sacar las coordenadas de dónde
            # De eso se encargará el gps, ¡¡ Pero en este punto tendremos que avisarle !!
    		threading.Thread(target=play_alarm_sound_function).start()
    		Alarm_Status = True

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
video.release()
