import cv2
import numpy as np
import threading
import playsound
import time

# Iniciar la captura de la webcam
cap = cv2.VideoCapture(1)

# Comprobar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

# Definir los rangos de color mejorados para detectar la llama
# lower_fire = np.array([18, 50, 50])   # Amarillo claro a naranja
# upper_fire = np.array([35, 255, 255]) # Amarillo-blanco brillante

lower_fire = np.array([10,5,230])
upper_fire = np.array([55,50,255])

# Leer el primer frame para la detección de movimiento
ret, prev_frame = cap.read()
prev_frame = cv2.resize(prev_frame, (960, 540))
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)

fire_detected = False  # Estado del fuego
last_fire_time = 0  # Último instante en que se detectó fuego
alarm_playing = False  # Estado de la alarma

# Función para reproducir el sonido de la alarma en un hilo separado
def play_alarm():
    global alarm_playing
    playsound.playsound("alarm-sound.mp3", True)  # Asegúrate de que el archivo existe
    alarm_playing = False  # Cuando termine, permite reproducir de nuevo

last_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el frame")
        break

    # Redimensionar para que coincida con prev_frame
    frame = cv2.resize(frame, (960, 540))

    # Aplicar filtro Gaussiano para reducir ruido
    blur = cv2.GaussianBlur(frame, (21, 21), 0)

    # Convertir a HSV para detección de fuego
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Crear una máscara para detectar fuego según el rango HSV corregido
    fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)

    # Contar los píxeles detectados como fuego
    fire_pixels = cv2.countNonZero(fire_mask)

    # Convertir a escala de grises y suavizar para detectar movimiento
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Comparar con el frame anterior para detectar movimiento
    diff = cv2.absdiff(prev_frame, gray)
    _, motion_mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    
    # Buscar los contornos del fuego detectado y mostrarlo
    contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 700:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)
    
    # Combinar detección de fuego con movimiento
    combined_mask = cv2.bitwise_and(fire_mask, motion_mask)
    
    # Determinar si hay fuego con base en el número de píxeles detectados
    if fire_pixels > 2700:  # Ajusta este umbral según pruebas
        print(time.time() - last_time)
        if time.time() - last_time >= 1.0:
            fire_detected = True
            last_fire_time = time.time()  # Actualizamos el último momento en que hubo fuego
            
            # Si no hay una alarma en curso, iniciar una nueva
            if not alarm_playing:
                alarm_playing = True
                threading.Thread(target=play_alarm, daemon=True).start()  # Ejecutar en un hilo separado     
    else:
        last_time = time.time()
        fire_detected = False

    # Mostrar alerta en pantalla durante al menos 3 segundos después de que el fuego desaparezca
    if fire_detected or (time.time() - last_fire_time <= 2):  # 🔥 Se mantiene el mensaje 3s extra
            cv2.putText(frame, "🔥 FUEGO DETECTADO 🔥", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    # Mostrar las imágenes
    cv2.imshow("Webcam", frame)
    cv2.imshow("Mascara de Fuego", fire_mask)
    cv2.imshow("Detección de Movimiento", motion_mask)
    cv2.imshow("Fuego en Movimiento", combined_mask)

    # Actualizar el frame anterior
    prev_frame = gray.copy()

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
