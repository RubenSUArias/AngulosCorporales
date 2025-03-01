import math
import cv2
import mediapipe as mp

def calculate_angle(landmark1, landmark2, landmark3):
  """Calcula el ángulo entre tres puntos de referencia."""

  # Obtén las coordenadas de los puntos de referencia.
  x1, y1 = landmark1.x, landmark1.y
  x2, y2 = landmark2.x, landmark2.y
  x3, y3 = landmark3.x, landmark3.y

  # Calcula los vectores entre los puntos de referencia.
  vector1 = [x1 - x2, y1 - y2]
  vector2 = [x3 - x2, y3 - y2]

  # Calcula el producto punto de los vectores.
  dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

  # Calcula la magnitud de los vectores.
  magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
  magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

  # Calcula el ángulo en radianes.
  angle_radians = math.acos(dot_product / (magnitude1 * magnitude2))

  # Convierte el ángulo a grados.
  angle_degrees = math.degrees(angle_radians)

  return angle_degrees



def calculate_elipse(landmark1, landmark2, landmark3):
  """Calcula el ángulo entre tres puntos de referencia."""

  # Obtén las coordenadas de los puntos de referencia.
  x1, y1 = landmark1[0], landmark1[1]
  x2, y2 = landmark2.x, landmark2.y
  x3, y3 = landmark3.x, landmark3.y

  # Calcula los vectores entre los puntos de referencia.
  vector1 = [x1 - x2, y1 - y2]
  vector2 = [x3 - x2, y3 - y2]

  # Calcula el producto punto de los vectores.
  dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

  # Calcula la magnitud de los vectores.
  magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
  magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

  # Calcula el ángulo en radianes.
  angle_radians = math.acos(dot_product / (magnitude1 * magnitude2))

  # Convierte el ángulo a grados.
  angle_degrees = math.degrees(angle_radians)

  return angle_degrees


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Inicializa MediaPipe Pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Captura el video desde la cámara web (0) o un archivo de video
img = cv2.imread("https://github.com/RubenSUArias/AngulosCorporales/blob/main/imagenes/Sentado2.png")  # Reemplaza 0 con la ruta a tu archivo de video si es necesario
# Obtiene las dimensiones del video de entrada
height, width, _ = img.shape
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Procesa la imagen con MediaPipe Pose
results = pose.process(image_rgb)
if results.pose_landmarks:
        # Crea una lista de conexiones a dibujar, excluyendo las de la cara
        connections_to_draw = [conn for conn in mp_pose.POSE_CONNECTIONS
                              if conn[0] not in range(0, 11) and conn[1] not in range(0, 11)]
        for connection in connections_to_draw:
            landmark1 = results.pose_landmarks.landmark[connection[0]]
            landmark2 = results.pose_landmarks.landmark[connection[1]]
            if landmark1.visibility > 0.5 and landmark2.visibility > 0.5:  # Ajusta el umbral de visibilidad según sea necesario
                x1 = int(landmark1.x * width)
                y1 = int(landmark1.y * height)
                x2 = int(landmark2.x * width)
                y2 = int(landmark2.y * height)
                cv2.circle(img, (x1, y1), 4, (0,0, 255), -2)
                cv2.circle(img, (x2, y2), 4, (0, 0, 255), 2)
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                #mp_drawing.draw_landmarks(frame, results.pose_landmarks, connections_to_draw)

        # Calcula el ángulo del codo izquierdo (si está disponible)
        if results.pose_landmarks:

          right_ankle = results.pose_landmarks.landmark[28]
          right_elbow = results.pose_landmarks.landmark[14]

          left_ankle = results.pose_landmarks.landmark[27]


          left_shoulder = results.pose_landmarks.landmark[11]
          left_hip = results.pose_landmarks.landmark[23]
          left_knee = results.pose_landmarks.landmark[25]
          right_shoulder = results.pose_landmarks.landmark[12]
          right_hip = results.pose_landmarks.landmark[24]
          right_knee = results.pose_landmarks.landmark[26]


          cadera_rodilla_tobillo = calculate_angle(right_hip, right_knee, right_ankle)
          codo_hombro_cadera = calculate_angle(right_hip, right_shoulder, right_elbow)
          left_abb_angle = calculate_angle(right_shoulder, right_hip, right_knee)
          left_pierna = calculate_angle(left_hip, left_knee, left_ankle)

# Muestra el ángulo en el frame
if right_hip.visibility > 0.5 and right_knee.visibility > 0.5 and right_ankle.visibility > 0.5:
  cv2.putText(img, f": {int(cadera_rodilla_tobillo)}", (int(right_knee.x * width), int(right_knee.y * height)),
              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
if right_shoulder.visibility > 0.5 and right_hip.visibility > 0.5 and right_knee.visibility > 0.5:
  cv2.putText(img, f": {int(left_abb_angle)}", (int(right_hip.x * width),int(right_hip.y*height)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

if right_shoulder.visibility > 0.5 and right_hip.visibility > 0.5 and right_elbow.visibility > 0.5:
  cv2.putText(img, f": {int(codo_hombro_cadera)}", (int(right_shoulder.x * width),int(right_shoulder.y*height)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Ángulo inicial y final del arco (en grados)
angle =0
#start_angle =0
#end_angle = 90  # Por ejemplo, un cuarto de círculo

# Color del arco
color = (0, 255, 0)  # Verde
axes_length = (50, 50)
# Grosor del arco
thickness = 2
#############Hombro##########################
center_coordinates = (int(right_shoulder.x * width), int(right_shoulder.y*height))
puntocero=[right_shoulder.x+0.1, right_shoulder.y]
coord_puntocero=(int((right_shoulder.x+0.1)*width), int(right_shoulder.y*height))
start_angle=calculate_elipse(puntocero, right_shoulder, right_elbow)
end_angle=calculate_elipse(puntocero, right_shoulder, right_hip)
cv2.ellipse(img, center_coordinates, axes_length,angle,start_angle, end_angle, color, thickness)
cv2.circle(img, coord_puntocero, 4, (0, 0, 250), -1)
print(end_angle,start_angle,end_angle-start_angle)
#############Rodilla##########################
center_coordinates = (int(right_knee.x * width), int(right_knee.y*height))
puntocero=[right_knee.x+0.1, right_knee.y]
coord_puntocero=(int((right_knee.x+0.1)*width), int(right_knee.y*height))
start_angle=calculate_elipse(puntocero, right_knee, right_ankle)
end_angle=180+180-calculate_elipse(puntocero, right_knee, right_hip)
cv2.ellipse(img, center_coordinates, axes_length,angle,start_angle,end_angle, color, thickness)
cv2.circle(img, coord_puntocero, 4, (0, 0, 250), -1)
print(end_angle,start_angle,end_angle-start_angle)
#############Cadera##########################
center_coordinates = (int(right_hip.x * width), int(right_hip.y*height))
puntocero=[right_hip.x+0.1, right_hip.y]
coord_puntocero=(int((right_hip.x+0.1)*width), int(right_hip.y*height))
start_angle=calculate_elipse(puntocero, right_hip, right_knee)
end_angle=calculate_elipse(puntocero, right_hip, right_shoulder)
cv2.ellipse(img, center_coordinates, axes_length,angle,start_angle, -end_angle, color, thickness)
cv2.circle(img, coord_puntocero, 4, (0, 0, 250), -1)
print(end_angle,start_angle,end_angle+start_angle)
cv2_imshow(img)
