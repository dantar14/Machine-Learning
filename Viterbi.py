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

while True:
    probabilidad_emision = {
        estado: {observacion: random.uniform(0, 1) for observacion in observaciones}
        for estado in estados
    }

    tabla_probabilidades = []
    for estado in estados:
        fila = [probabilidad_emision[estado][observacion] * probabilidad_inicial[estado] for observacion in observaciones]
        tabla_probabilidades.append([estado.capitalize()] + fila)

    print("Matriz de Probabilidades:")
    print(tabulate(tabla_probabilidades, headers=[''] + [f'Observacion {i+1}' for i in range(len(observaciones))], tablefmt='grid'))

    for i, observacion in enumerate(observaciones):
        estado_mas_probable = max(estados, key=lambda estado: probabilidad_emision[estado][observacion] * probabilidad_inicial[estado])
        print(f'\nEn la observación {observacion}, el estado más probable es: {estado_mas_probable.capitalize()}')

    continuar = input("¿Desea generar otra predicción? (Sí/No): ").strip().lower()
    if continuar != 'si' and continuar != 'sí':
        break
