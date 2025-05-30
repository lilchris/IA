import pygame
from queue import PriorityQueue

pygame.init()
# Configuraciones iniciales
ANCHO_VENTANA = 800
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Visualización de Nodos")

# Colores (RGB)
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)
VERDE = (0, 255, 0)
ROJO = (255, 0, 0)
NARANJA = (255, 165, 0)
PURPURA = (128, 0, 128)
PESO = 0

class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = fila * ancho
        self.y = col * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas
        # self.peso = float("inf")
        self.g = float("inf")  # Costo desde el inicio
        self.h = float("inf")  # Estimación hasta el final
        self.f = float("inf")  # Costo total

    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA

    def restablecer(self):
        self.color = BLANCO
        # self.peso = float("inf")
        self.g = float("inf")
        self.h = float("inf")
        self.f = float("inf")

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA

    def actualizar_vecinos(self, grid):
        self.vecinos = []
        #movimientos verticales y horizontales
        if self.fila < self.total_filas - 1 and not grid[self.fila + 1][self.col].es_pared():  # Abajo
            self.vecinos.append(grid[self.fila + 1][self.col])

        if self.fila > 0 and not grid[self.fila - 1][self.col].es_pared():  # Arriba
            self.vecinos.append(grid[self.fila - 1][self.col])

        if self.col < self.total_filas - 1 and not grid[self.fila][self.col + 1].es_pared():  # Derecha
            self.vecinos.append(grid[self.fila][self.col + 1])

        if self.col > 0 and not grid[self.fila][self.col - 1].es_pared():  # Izquierda
            self.vecinos.append(grid[self.fila][self.col - 1])
        
        #movimientos en diagonal
        dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in dirs:
            new_fila = self.fila + dx
            new_col = self.col + dy
            if 0 <= new_fila < self.total_filas and 0 <= new_col < self.total_filas:
                if not grid[new_fila][new_col].es_pared():
                    self.vecinos.append(grid[new_fila][new_col])
    
    def hacer_cerrado(self):
        self.color = ROJO

    def hacer_abierto(self):
        self.color = VERDE

    def hacer_camino(self):
        self.color = (64, 224, 208)

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))
        fuente = pygame.font.Font(None, 18)  
        if self.h != float("inf"):
            texto_h = fuente.render(f"H:{int(self.h)}", True, (255, 255, 255))
            ventana.blit(texto_h, (self.x + 2, self.y + 2))
        
        if self.g != float("inf"):
            texto_g = fuente.render(f"G:{int(self.g)}", True, (0, 0, 255))
            ventana.blit(texto_g, (self.x + 2, self.y + self.ancho // 2 - 8))

        if self.f != float("inf"):
            texto_f = fuente.render(f"F:{int(self.f)}", True, (0, 0, 0))
            ventana.blit(texto_f, (self.x + 2, self.y + self.ancho - 18))

def crear_grid(filas, ancho):
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    return grid

def dibujar_grid(ventana, filas, ancho):
    ancho_nodo = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, GRIS, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, GRIS, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))

def dibujar(ventana, grid, filas, ancho):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)

    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()

def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    y, x = pos
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col

def heuristica(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return (abs(x1 - x2) + abs(y1 - y2))*10

def reconstruir_camino(came_from, actual, dibujar):
    while actual in came_from:
        actual = came_from[actual]
        actual.hacer_camino()
        dibujar()

def algoritmo_a_estrella(dibujar, grid, inicio, fin):
    contador = 0
    open_set = PriorityQueue()
    open_set.put((0, contador, inicio))
    came_from = {}

    g_score = {nodo: float("inf") for fila in grid for nodo in fila}
    g_score[inicio] = 0

    f_score = {nodo: float("inf") for fila in grid for nodo in fila}
    f_score[inicio] = heuristica(inicio.get_pos(), fin.get_pos())

    open_set_hash = {inicio}

    while not open_set.empty():
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()

        actual = open_set.get()[2]
        open_set_hash.remove(actual)

        if actual == fin:
            reconstruir_camino(came_from, fin, dibujar)
            fin.hacer_fin()
            inicio.hacer_inicio()
            return True

        for vecino in actual.vecinos:
            dx = abs(vecino.fila - actual.fila)
            dy = abs(vecino.col - actual.col)
            
            # Movimiento recto (horizontal o vertical)
            if dx + dy == 1:
                costo = 10
            # Movimiento diagonal
            elif dx == 1 and dy == 1:
                costo = 14
            else:
                continue  # Esto no debería pasar, pero es por seguridad

            temp_g_score = g_score[actual] + costo

            if temp_g_score < g_score[vecino]:
                came_from[vecino] = actual
                g_score[vecino] = temp_g_score
                h = heuristica(vecino.get_pos(), fin.get_pos())
                f_score[vecino] = temp_g_score + h

                vecino.g = temp_g_score     # Mostrar G
                vecino.h = h                # Mostrar H
                vecino.f = f_score[vecino]  # Mostrar F
                
                if vecino not in open_set_hash:
                    contador += 1
                    open_set.put((f_score[vecino], contador, vecino))
                    open_set_hash.add(vecino)
                    vecino.hacer_abierto()

        dibujar()

        if actual != inicio:
            actual.hacer_cerrado()

    return False

def main(ventana, ancho):
    FILAS = 7
    grid = crear_grid(FILAS, ancho)

    inicio = None
    fin = None

    corriendo = True

    while corriendo:
        dibujar(ventana, grid, FILAS, ancho)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and inicio and fin:
                    for fila in grid:
                        for nodo in fila:
                            nodo.actualizar_vecinos(grid)

                    algoritmo_a_estrella(lambda: dibujar(ventana, grid, FILAS, ancho), grid, inicio, fin)
            
            if event.type == pygame.QUIT:
                corriendo = False

            if pygame.mouse.get_pressed()[0]:  # Click izquierdo
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                if not inicio and nodo != fin:
                    inicio = nodo
                    inicio.hacer_inicio()

                elif not fin and nodo != inicio:
                    fin = nodo
                    fin.hacer_fin()

                elif nodo != fin and nodo != inicio:
                    nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]:  # Click derecho
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                nodo.restablecer()
                if nodo == inicio:
                    inicio = None
                elif nodo == fin:
                    fin = None

    pygame.quit()

main(VENTANA, ANCHO_VENTANA)
