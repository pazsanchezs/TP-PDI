# -*- coding: utf-8 -*-
"""Procesamiento y comparacion de metodos de ecualizacion de histograma en el dataset Exclusively Dark Image Dataset (exDARK)""" 

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy
import time #para medir tiempos de ejecucion

def proposed_hist_equalization(image):
    """Implementación del método propuesto"""
    # Convertir a escala de grises si es una imagen en color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     # Calcular histograma de la imagen 
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    # Calcular punto de separación (SP) usando la mediana (más robusta que la media)
    sp = int(np.median(image))
    
    # Segmentar histograma en dos partes
    # - his_l: valores menores o iguales al punto de separación
    # - his_h: valores mayores al punto de separación
    his_l = hist[:sp + 1]
    his_h = hist[sp + 1:]
    
    # Calcular límites de meseta (Plateau Limits) adaptativos:
    # 1. Encontrar picos de cada sub-histograma
    pk_l, pk_h = np.max(his_l), np.max(his_h)
    # 2. Calcular ratios de niveles de gris (GR)
    #gr_l2: ratio para el histograma inferior
    #gr_h2: ratio para el histograma superior
    gr_l2 = (sp - np.mean(np.where(hist[:sp] > 0)[0])) / sp if sp > 0 else 0.5
    gr_h2 = (255 - np.mean(np.where(hist[sp:] > 0)[0]) - sp) / (255 - sp) if (255 - sp) > 0 else 0.5
    # 3. Calcular límites de meseta usando fórmula adaptativa
    pl_l = (0.15 + 0.35 * gr_l2) * pk_l
    pl_h = (0.15 + 0.35 * gr_h2) * pk_h
    
    # Aplicar clipping mejorado usando raíz cuadrada para preservar mejor los detalles
    his_l_clipped = np.where(his_l > pl_l, pl_l + np.sqrt(his_l - pl_l), his_l)
    his_h_clipped = np.where(his_h > pl_h, pl_h + np.sqrt(his_h - pl_h), his_h)
    
    # Función interna para ecualizar sub-histogramas
    def equalize_sub_hist(sub_hist, sub_min, sub_max):
        cdf = sub_hist.cumsum()  # Función de distribución acumulativa
        cdf_normalized = (cdf - cdf.min()) * (sub_max - sub_min) / (cdf.max() - cdf.min() + 1e-6)
        return cdf_normalized.astype('uint8')
    # Ecualizar ambos sub-histogramas por separado
    cdf_l = equalize_sub_hist(his_l_clipped, 0, sp)
    cdf_h = equalize_sub_hist(his_h_clipped, sp + 1, 255)
    
    #Aplicar la transformación a la imagen original
    equalized_image = np.zeros_like(image)
    for i in range(256):
        if i <= sp and i < len(cdf_l):
            equalized_image[image == i] = cdf_l[i]
        elif i > sp and (i - sp - 1) < len(cdf_h):
            equalized_image[image == i] = cdf_h[i - sp - 1]
    
    return equalized_image

def process_image(image_path, output_dir, img_name):
    """ Procesa una imagen individual aplicando los tres métodos de ecualización
   y calculando las métricas correspondientes."""
    try:
        # Leer imagen manteniendo el color si es necesario
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: No se pudo leer {img_name}")
            return None
        
        # Convertir a escala de grises para procesamiento
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()
        
       # Aplicar los tres métodos de ecualización:
        # 1. HE (Histogram Equalization) estándar
        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # 3. Nuestro método propuesto
        methods = {
            'HE': cv2.equalizeHist(gray_image),
            'CLAHE': cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray_image),
            'Proposed': proposed_hist_equalization(gray_image)
        }
        
       # Calcular métricas para la imagen original y cada método aplicado
        metrics = {'Original': {
            'Brightness': np.mean(gray_image),
            'Contrast': np.std(gray_image),
            'Entropy': shannon_entropy(gray_image)
        }}
        # Agregar métricas para cada método
        for name, result in methods.items():
            metrics[name] = {
                'Brightness': np.mean(result),
                'Contrast': np.std(result),
                'Entropy': shannon_entropy(result)
            }
        
        # Guardar resultados
        base_name = os.path.splitext(img_name)[0]
        output_paths = {
            'original': os.path.join(output_dir, 'images', f'{base_name}_original.png'),
            'HE': os.path.join(output_dir, 'images', f'{base_name}_HE.png'),
            'CLAHE': os.path.join(output_dir, 'images', f'{base_name}_CLAHE.png'),
            'Proposed': os.path.join(output_dir, 'images', f'{base_name}_Proposed.png'),
            'plot': os.path.join(output_dir, 'plots', f'{base_name}_comparison.png')
        }
        
        # Guardar imágenes
        cv2.imwrite(output_paths['original'], gray_image)
        for name in methods:
            cv2.imwrite(output_paths[name], methods[name])
        
        # Generar gráfico comparativo
        plt.figure(figsize=(18, 9))
        titles = ['Original', 'HE Estándar', 'CLAHE', 'Algoritmo Propuesto']
        
        for i, (title, img) in enumerate(zip(titles, [gray_image] + list(methods.values())), 1):
            plt.subplot(2, 4, i)
            plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.axis('off')
            
            plt.subplot(2, 4, i+4)
            plt.hist(img.ravel(), 256, [0, 256], color=['r', 'g', 'b', 'm'][i-1])
            plt.xlim([0, 256])
            plt.title(f'Histograma {title}')
        
        plt.tight_layout()
        plt.savefig(output_paths['plot'], dpi=100, bbox_inches='tight')
        plt.close()
        
        return metrics
        
    except Exception as e:
        print(f"Error procesando {img_name}: {str(e)}")
        return None

def main():
    """
    Función principal que:
    1. Busca imágenes en el dataset
    2. Procesa cada imagen
    3. Calcula métricas globales
    4. Guarda resultados
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(script_dir, "Exclusively-Dark-Image-Dataset-master")
    
    # Buscar imágenes en todas las subcarpetas relevantes
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_paths = []
    
    for root, dirs, files in os.walk(dataset_root):
        if "Dataset" in root or "Low" in root:
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        print("No se encontraron imágenes en el dataset. Verifica la estructura:")
        print(f"Directorio raíz del dataset: {dataset_root}")
        print("Las imágenes deben estar en carpetas llamadas 'Dataset' o 'Low'")
        return
    
    # Configurar directorio de salida
    output_dir = os.path.join(script_dir, "exdark_results")
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'metrics'), exist_ok=True)
    
    # Procesar imágenes
    all_metrics = []
    print(f"\nIniciando procesamiento de {len(image_paths)} imágenes...")
    
    for i, img_path in enumerate(image_paths, 1):
        img_name = os.path.basename(img_path)
        metrics = process_image(img_path, output_dir, img_name)
        
        if metrics:
            all_metrics.append(metrics)
            
            # Guardar métricas individuales
            with open(os.path.join(output_dir, 'metrics', f'metrics_{i}.txt'), 'w') as f:
                for method, values in metrics.items():
                    f.write(f"{method}:\n")
                    for k, v in values.items():
                        f.write(f"  {k}: {v:.2f}\n")
                    f.write("\n")
            
            if i % 10 == 0 or i == len(image_paths):
                print(f"Procesadas {i}/{len(image_paths)} imágenes...")
    
    # Calcular y mostrar métricas promedio
    if all_metrics:
        methods = ['HE', 'CLAHE', 'Proposed']
        avg_metrics = {method: {
            'Brightness': np.mean([m[method]['Brightness'] for m in all_metrics]),
            'Contrast': np.mean([m[method]['Contrast'] for m in all_metrics]),
            'Entropy': np.mean([m[method]['Entropy'] for m in all_metrics])
        } for method in methods}
        
        # Guardar métricas promedio
        with open(os.path.join(output_dir, 'average_metrics.txt'), 'w') as f:
            f.write("Métricas Promedio:\n\n")
            f.write(f"Número de imágenes procesadas: {len(all_metrics)}\n\n")
            
            f.write("Original:\n")
            for k, v in all_metrics[0]['Original'].items():
                f.write(f"  {k}: {np.mean([m['Original'][k] for m in all_metrics]):.2f}\n")
            f.write("\n")
            
            for method in methods:
                f.write(f"{method}:\n")
                for k, v in avg_metrics[method].items():
                    f.write(f"  {k}: {v:.2f}\n")
                f.write("\n")
        
        # Mostrar resumen
        print("\n" + "="*50)
        print("RESUMEN DE MÉTRICAS PROMEDIO")
        print("="*50)
        
        print("\nOriginal:")
        for k, v in all_metrics[0]['Original'].items():
            print(f"  {k}: {np.mean([m['Original'][k] for m in all_metrics]):.2f}")
        
        for method in methods:
            print(f"\n{method}:")
            for k, v in avg_metrics[method].items():
                print(f"  {k}: {v:.2f}")
    
    print(f"\nProcesamiento completado. Resultados guardados en: {output_dir}")

if __name__ == "__main__":
    main()