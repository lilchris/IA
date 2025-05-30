import pygame
import random
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.tree import DecisionTreeClassifier 
import numpy as np
import csv
import os

# consts
W, H = 800, 400
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
FPS = 30

# pos max
DISTANCIA_MAXIMA_DESDE_INICIO = 35  # Máximo que puede alejarse desde x=50 (prueba: 25, 35, 45, 60)
POSICION_INICIAL_X = 50  # Posición inicial del jugador
POSICION_MAXIMA_X = POSICION_INICIAL_X + DISTANCIA_MAXIMA_DESDE_INICIO  # Posición máxima permitida

# init
pygame.init()
pantalla = pygame.display.set_mode((W, H))
pygame.display.set_caption("Juego: Disparo de Bala, Salto, Nave y Menú")
fuente = pygame.font.SysFont('Arial', 24)
reloj = pygame.time.Clock()

# variables de juego
jugador = pygame.Rect(50, H - 100, 32, 48)
bala = pygame.Rect(W - 50, H - 90, 16, 16)
bala_vertical = pygame.Rect(50, 0, 16, 16)
nave = pygame.Rect(W - 100, H - 100, 64, 64)

salto = False
salto_altura = 15
gravedad = 1
en_suelo = True

velocidad_bala = -30
bala_disparada = False
bala_vertical_disparada = False

mover_derecha_temporal = False
contador_mov_derecha = 0
MAX_MOV_DERECHA = 20
distancia_objetivo = 20  

pausa = False
menu_activo = True
modo_auto = False

datos_modelo = []
datos_guardados_count = 0  
modelo_nn = None
scaler_nn = None

modelo_knn = None
scaler_knn = None
modelo_dt = None
scaler_dt = None
tipo_modelo_activo = None  # nn=red neuronal, knn=k-neighbors, dt=decision tree

# carga recursos
jugador_frames = [pygame.image.load(f'assets/sprites/mono_frame_{i}.png') for i in range(1, 5)]
bala_img = pygame.image.load('assets/sprites/purple_ball.png')
fondo_img = pygame.transform.scale(pygame.image.load('assets/game/fondo2.png'), (W, H))
nave_img = pygame.image.load('assets/game/ufo.png')

# animación del jugador
current_frame = 0
frame_speed = 10
frame_count = 0

# Fondo en movimiento
fondo_x1, fondo_x2 = 0, W

# funcs

def disparar_bala():
    global bala_disparada, velocidad_bala, bala_vertical_disparada, bala_vertical
    if not bala_disparada:
        velocidad_bala = random.randint(-10, -5)
        bala_disparada = True
    if not bala_vertical_disparada:
        bala_vertical.topleft = (50, 0)
        bala_vertical_disparada = True

def reset_bala():
    global bala, bala_disparada
    bala.x = W - 50
    bala_disparada = False

def manejar_salto():
    global jugador, salto, salto_altura, en_suelo
    if salto:
        jugador.y -= salto_altura
        salto_altura -= gravedad
        if jugador.y >= H - 100:
            jugador.y = H - 100
            salto = False
            salto_altura = 15
            en_suelo = True

def actualizar_fondo():
    global fondo_x1, fondo_x2
    fondo_x1 -= 1
    fondo_x2 -= 1
    if fondo_x1 <= -W:
        fondo_x1 = W
    if fondo_x2 <= -W:
        fondo_x2 = W
    pantalla.blit(fondo_img, (fondo_x1, 0))
    pantalla.blit(fondo_img, (fondo_x2, 0))

def actualizar_animacion_jugador():
    global current_frame, frame_count
    frame_count += 1
    if frame_count >= frame_speed:
        current_frame = (current_frame + 1) % len(jugador_frames)
        frame_count = 0
    pantalla.blit(jugador_frames[current_frame], jugador.topleft)

def actualizar_balas():
    global bala, velocidad_bala, bala_disparada, bala_vertical, bala_vertical_disparada

    # Bala horizontal
    if bala_disparada:
        bala.x += velocidad_bala
        if bala.x < 0:
            reset_bala()
    pantalla.blit(bala_img, bala.topleft)

    # Bala vertical
    if bala_vertical_disparada:
        bala_vertical.y += 5
        if bala_vertical.y > H:
            bala_vertical.topleft = (50, 0)
    pantalla.blit(bala_img, bala_vertical.topleft)

# lim de pos
def manejar_movimiento_jugador():
    global mover_derecha_temporal, contador_mov_derecha, jugador, distancia_objetivo
    global POSICION_MAXIMA_X, POSICION_INICIAL_X
    
    if mover_derecha_temporal:
        if contador_mov_derecha < distancia_objetivo:
            # Verificar que no exceda la pos mmax permitida
            nueva_posicion = jugador.x + 2
            if nueva_posicion <= POSICION_MAXIMA_X:
                jugador.x = nueva_posicion
                contador_mov_derecha += 2
            else:
                # Si alcanzó el lim, detener el movimiento
                jugador.x = POSICION_MAXIMA_X
                mover_derecha_temporal = False
                contador_mov_derecha = 0
        else:
            mover_derecha_temporal = False
            contador_mov_derecha = 0
    else:
        # Regresar gradualmente a la pos inicial
        if jugador.x > POSICION_INICIAL_X:
            jugador.x = max(POSICION_INICIAL_X, jugador.x - 2)

def detectar_colisiones():
    if jugador.colliderect(bala) or jugador.colliderect(bala_vertical):
        print("Colisión detectada!")
        guardar_datos_en_csv()
        reiniciar_juego()

def guardar_datos():
    distancia_horizontal = abs(jugador.x - bala.x)
    distancia_vertical = abs(jugador.y - bala_vertical.y)
    accion = 2 if salto else 1 if mover_derecha_temporal else 0
    datos_modelo.append((velocidad_bala, distancia_horizontal, distancia_vertical, accion))

def pausa_juego():
    global pausa
    pausa = not pausa
    if pausa:
        print("Juego pausado. Datos registrados:", len(datos_modelo))
        guardar_datos_en_csv()
    else:
        print("Juego reanudado.")

def mostrar_menu():
    global menu_activo, modo_auto
    pantalla.fill(NEGRO)
    texto1 = fuente.render("'A' Red Neuronal, 'S' K-Neighbors, 'D' Decision Tree", True, BLANCO)
    texto2 = fuente.render("'M' para Manual, 'Q' para Salir", True, BLANCO)
    texto3 = fuente.render("Christian Rojas Anaya", True, BLANCO)
    
    pantalla.blit(texto1, (W // 8, H // 2 - 40))
    pantalla.blit(texto2, (W // 4, H // 2 - 10))
    pantalla.blit(texto3, (W // 4+5, H // 2 + 20))
    pygame.display.flip()

    while menu_activo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_a:
                    cargar_datos_desde_csv()
                    if entrenar_red_neuronal():
                        modo_auto = True
                        return
                elif evento.key == pygame.K_s:
                    cargar_datos_desde_csv()
                    if entrenar_k_neighbors():
                        modo_auto = True
                        return
                elif evento.key == pygame.K_d:
                    cargar_datos_desde_csv()
                    if entrenar_decision_tree():
                        modo_auto = True
                        return
                elif evento.key == pygame.K_m:
                    modo_auto = False
                    return
                elif evento.key == pygame.K_q:
                    print("Juego terminado. Datos recopilados:", len(datos_modelo))
                    pygame.quit()
                    exit()

def reiniciar_juego():
    global menu_activo, jugador, bala, nave, bala_disparada, salto, en_suelo
    global bala_vertical, bala_vertical_disparada, mover_derecha_temporal, contador_mov_derecha
    global distancia_objetivo

    menu_activo = True
    jugador.topleft = (POSICION_INICIAL_X, H - 100)  # pos del jugador inicial
    bala.x = W - 50
    nave.topleft = (W - 100, H - 100)
    bala_disparada = False
    salto = False
    en_suelo = True
    mover_derecha_temporal = False
    contador_mov_derecha = 0
    distancia_objetivo = MAX_MOV_DERECHA  # Valor de mov a la derecha
    bala_vertical.topleft = (50, 0)
    bala_vertical_disparada = False

    print("Datos recopilados para el modelo:", len(datos_modelo))
    mostrar_menu()

def entrenar_red_neuronal():
    global modelo_nn, scaler_nn, datos_modelo, pausa, modo_auto, tipo_modelo_activo
    if len(datos_modelo) < 10:
        print("No hay suficientes datos para entrenar la red neuronal.")
        return False
    datos = np.array(datos_modelo)
    X = datos[:, :3]
    y = datos[:, 3]
    scaler_nn = StandardScaler()
    X_scaled = scaler_nn.fit_transform(X)
    modelo_nn = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=42)
    modelo_nn.fit(X_scaled, y)
    tipo_modelo_activo = 'nn'
    print("Red neuronal entrenada con éxito.")
    pausa = False
    modo_auto = True
    return True

def entrenar_k_neighbors():
    global modelo_knn, scaler_knn, datos_modelo, pausa, modo_auto, tipo_modelo_activo
    if len(datos_modelo) < 10:
        print("No hay suficientes datos para entrenar K-neighbors.")
        return False
    datos = np.array(datos_modelo)
    X = datos[:, :3]
    y = datos[:, 3]
    scaler_knn = StandardScaler()
    X_scaled = scaler_knn.fit_transform(X)
    k_value = min(5, len(set(y)))
    modelo_knn = KNeighborsClassifier(n_neighbors=k_value)
    modelo_knn.fit(X_scaled, y)
    tipo_modelo_activo = 'knn'
    print(f"K-neighbors entrenado con éxito (k={k_value}).")
    pausa = False
    modo_auto = True
    return True

def entrenar_decision_tree():
    global modelo_dt, scaler_dt, datos_modelo, pausa, modo_auto, tipo_modelo_activo
    if len(datos_modelo) < 10:
        print("No hay suficientes datos para entrenar Decision Tree.")
        return False
    datos = np.array(datos_modelo)
    X = datos[:, :3]
    y = datos[:, 3]
    scaler_dt = StandardScaler()
    X_scaled = scaler_dt.fit_transform(X)
    modelo_dt = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    modelo_dt.fit(X_scaled, y)
    tipo_modelo_activo = 'dt'
    print("Decision Tree entrenado con éxito.")
    pausa = False
    modo_auto = True
    return True

# Calcular distancia antes de moverse
def decision_salto_automatico():
    global modelo_nn, scaler_nn, modelo_knn, scaler_knn, modelo_dt, scaler_dt, tipo_modelo_activo
    global velocidad_bala, jugador, bala, bala_vertical, mover_derecha_temporal, contador_mov_derecha
    global distancia_objetivo
    
    # modelo usar
    if tipo_modelo_activo == 'nn' and modelo_nn and scaler_nn:
        modelo_actual = modelo_nn
        scaler_actual = scaler_nn
        nombre_modelo = "Red Neuronal"
    elif tipo_modelo_activo == 'knn' and modelo_knn and scaler_knn:
        modelo_actual = modelo_knn
        scaler_actual = scaler_knn
        nombre_modelo = "K-neighbors"
    elif tipo_modelo_activo == 'dt' and modelo_dt and scaler_dt:
        modelo_actual = modelo_dt
        scaler_actual = scaler_dt
        nombre_modelo = "Decision Tree"
    else:
        print("Ningún modelo entrenado disponible.")
        return
    
    distancia_horizontal = abs(jugador.x - bala.x)
    distancia_vertical = abs(jugador.y - bala_vertical.y)
    entrada = np.array([[velocidad_bala, distancia_horizontal, distancia_vertical]])
    entrada_norm = scaler_actual.transform(entrada)
    accion = int(modelo_actual.predict(entrada_norm)[0])
    
    if accion == 1:
        # movimiento estándar, el límite se aplica en manejar_movimiento_jugador()
        distancia_objetivo = MAX_MOV_DERECHA
        mover_derecha_temporal = True
        contador_mov_derecha = 0
    elif accion == 2:
        realizar_salto()

def realizar_salto():
    global salto, en_suelo
    if en_suelo:
        salto = True
        en_suelo = False

def guardar_datos_en_csv(nombre_archivo="datos_entrenamiento.csv"):
    global datos_modelo, datos_guardados_count
    
    # verificar si hay datos nuevos para guardar
    datos_nuevos = datos_modelo[datos_guardados_count:]
    if not datos_nuevos:
        print("No hay datos nuevos para guardar.")
        return
    
    file_exists = os.path.exists(nombre_archivo)
    
    with open(nombre_archivo, mode="w", newline="") as archivo:
        writer = csv.writer(archivo)
        
        if not file_exists:
            writer.writerow(["velocidad_bala", "distancia_horizontal", "distancia_vertical", "accion"])
            print(f"Archivo {nombre_archivo} creado con encabezado.")
        
        # escribir solo los datos nuevos
        writer.writerows(datos_nuevos)
        
        datos_guardados_count = len(datos_modelo)
        
        print(f"Agregados {len(datos_nuevos)} nuevos registros a {nombre_archivo}")
        print(f"Total de datos en sesión actual: {len(datos_modelo)}")

def cargar_datos_desde_csv(nombre_archivo="datos_entrenamiento.csv"):
    global datos_modelo, datos_guardados_count
    if not os.path.exists(nombre_archivo):
        print(f"No se encontró {nombre_archivo}, comenzando desde cero.")
        datos_modelo = []
        datos_guardados_count = 0
        return
    with open(nombre_archivo, mode="r") as archivo:
        reader = csv.reader(archivo)
        next(reader)
        datos_modelo = [tuple(map(float, fila)) for fila in reader]
    
    datos_guardados_count = len(datos_modelo)
    print(f"Datos cargados desde {nombre_archivo}: {len(datos_modelo)} entradas")

def main():
    global salto, en_suelo, bala_disparada, modo_auto, menu_activo, mover_derecha_temporal, contador_mov_derecha
    global distancia_objetivo

    cargar_datos_desde_csv()
    mostrar_menu()

    correr = True
    while correr:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                correr = False
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE and en_suelo and not pausa:
                    realizar_salto()
                elif evento.key == pygame.K_p:
                    pausa_juego()
                elif evento.key == pygame.K_a and pausa:
                    if entrenar_red_neuronal():
                        menu_activo = False
                elif evento.key == pygame.K_s and pausa:
                    if entrenar_k_neighbors():
                        menu_activo = False
                elif evento.key == pygame.K_d and pausa:
                    if entrenar_decision_tree():
                        menu_activo = False
                elif evento.key == pygame.K_q:
                    print("Juego terminado. Datos recopilados:", len(datos_modelo))
                    correr = False

        if not pausa:
            if modo_auto and (modelo_nn is not None or modelo_knn is not None or modelo_dt is not None):
                decision_salto_automatico()
            keys = pygame.key.get_pressed()
            if not modo_auto and keys[pygame.K_RIGHT]:
                distancia_objetivo = MAX_MOV_DERECHA
                mover_derecha_temporal = True
                contador_mov_derecha = 0

            manejar_salto()
            guardar_datos()
            if not bala_disparada:
                disparar_bala()
            actualizar_fondo()
            actualizar_animacion_jugador()
            pantalla.blit(nave_img, nave.topleft)
            actualizar_balas()
            manejar_movimiento_jugador()
            detectar_colisiones()

        pygame.display.flip()
        reloj.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()