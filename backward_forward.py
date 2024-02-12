import random
from tabulate import tabulate

estados = ('Lluvioso', 'Soleado', 'Nublado', 'Nevado')
observaciones = ('caminar', 'comprar', 'limpiar')

probabilidad_inicial = {'Lluvioso': 0.25, 'Soleado': 0.25, 'Nublado': 0.25, 'Nevado': 0.25}

probabilidad_transicion = {
    'Lluvioso': {'Lluvioso': 0.3, 'Soleado': 0.2, 'Nublado': 0.3, 'Nevado': 0.2},
    'Soleado': {'Lluvioso': 0.2, 'Soleado': 0.4, 'Nublado': 0.2, 'Nevado': 0.2},
    'Nublado': {'Lluvioso': 0.25, 'Soleado': 0.25, 'Nublado': 0.3, 'Nevado': 0.2},
    'Nevado': {'Lluvioso': 0.3, 'Soleado': 0.2, 'Nublado': 0.2, 'Nevado': 0.3}
}

def backward_forward(obs):
    forward = [{}]
    backward = [{}]
    estados_mas_probables = []

    for estado in estados:
        forward[0][estado] = probabilidad_inicial[estado] * probabilidad_emision[estado][obs[0]]

    for t in range(1, len(obs)):
        forward.append({})
        for estado_destino in estados:
            forward[t][estado_destino] = sum(forward[t-1][estado_origen] * probabilidad_transicion[estado_origen][estado_destino] * probabilidad_emision[estado_destino][obs[t]] for estado_origen in estados)

    for estado in estados:
        backward[0][estado] = 1

    observaciones_pasadas = [obs[0]]
    for t in range(len(obs) - 2, -1, -1):
        backward.insert(0, {})
        estado_origen = max(probabilidad_transicion.keys(), key=lambda k: probabilidad_transicion[k][estado_destino] * probabilidad_emision[estado_destino][obs[t + 1]] * backward[1][estado_destino])
        observaciones_pasadas.insert(0, obs[t + 1])
        for estado_destino in estados:
            backward[0][estado_origen] = probabilidad_transicion[estado_origen][estado_destino] * probabilidad_emision[estado_destino][obs[t + 1]] * backward[1][estado_destino]

    marginales_forward = []
    marginales_backward = []

    for t in range(len(obs)):
        estado_mas_probable_forward = max(forward[t], key=forward[t].get)
        estado_mas_probable_backward = max(backward[t], key=backward[t].get)
        estados_mas_probables.append((f'Tiempo {t+1}', estado_mas_probable_forward, estado_mas_probable_backward, observaciones_pasadas[t]))

        marginales_forward.append([forward[t][estado] * backward[t][estado] for estado in estados])
        marginales_backward.append([backward[t][estado] for estado in estados])

    return estados_mas_probables, marginales_forward, marginales_backward

for observacion in observaciones:
    print(f"En la observación {observacion}, el estado más probable es: {random.choice(estados)}")

while True:
    probabilidad_emision = {
        estado: {observacion: random.uniform(0, 1) for observacion in observaciones}
        for estado in estados
    }

    tabla_probabilidades = []
    for estado in estados:
        fila = [probabilidad_emision[estado][observacion] * probabilidad_inicial[estado] for observacion in observaciones]
        tabla_probabilidades.append([estado.capitalize()] + fila)

    print("\nMatriz de Probabilidades:")
    print(tabulate(tabla_probabilidades, headers=[''] + [f'Observacion {i+1}' for i in range(len(observaciones))], tablefmt='grid'))

    observacion_actual = random.choice(observaciones)
    print(f"\nObservación actual: {observacion_actual}\n")

    estados_mas_probables, marginales_forward, marginales_backward = backward_forward([observacion_actual])

    print("Matriz Backward:")
    print(tabulate(marginales_backward, headers=[''] + list(estados), showindex=['Tiempo ' + str(i+1) for i in range(len(marginales_backward))], tablefmt='grid'))

    print("\nObservaciones con Estados Más Probables:")
    print(tabulate(estados_mas_probables, headers=['Tiempo', 'Forward', 'Backward', 'Observacion Pasada'], tablefmt='grid'))

    print("\nMatriz Forward:")
    print(tabulate(marginales_forward, headers=estados, showindex=['Tiempo ' + str(i+1) for i in range(len(marginales_forward))], tablefmt='grid'))

    continuar = input("¿Desea generar otra predicción? (Sí/No): ").strip().lower()
    if continuar != 'si' and continuar != 'sí':
        break