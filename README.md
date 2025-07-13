# 🚀 Advanced Product Recommendation System

Un sistema avanzado de recomendación de productos que implementa múltiples algoritmos de machine learning para proporcionar recomendaciones precisas y diversas. El sistema incluye búsqueda por descripción, filtrado colaborativo y recomendaciones basadas en perfiles de cliente.

## 🎯 Características Principales

### 🔍 **Búsqueda por Descripción** (3 Métodos)
- **KNN + Embeddings**: Búsqueda usando vecinos más cercanos con embeddings de SentenceTransformers
- **TF-IDF + Cosine**: Búsqueda basada en similitud de texto usando TF-IDF y similitud coseno
- **Clustering + Embeddings**: Búsqueda optimizada usando clustering K-means y embeddings

### 🤝 **Recomendación Colaborativa** (3 Métodos)
- **SVD**: Descomposición en valores singulares para encontrar productos similares
- **NMF**: Factorización matricial no-negativa para recomendaciones
- **Direct Similarity**: Similitud directa item-to-item usando matriz de interacciones

### 👤 **Recomendación Basada en Cliente** (3 Métodos)
- **Profile Similarity**: Perfil de usuario basado en embeddings promedio
- **Weighted Categories**: Recomendaciones ponderadas por categorías preferidas
- **Hybrid Approach**: Combinación de embeddings, categorías y ratings

### 📊 **Sistema de Métricas**
- Evaluación de diversidad, novedad y cobertura
- Métricas de precision@k
- Comparación de rendimiento entre métodos
- Análisis de tiempo de ejecución

## 🛠️ Instalación y Configuración

### Requisitos del Sistema
```bash
# Versión de Python recomendada
Python 3.8+
```

### Dependencias
```bash
pip install pandas numpy scikit-learn sentence-transformers gradio
pip install scipy matplotlib seaborn
```

### Estructura de Archivos Requerida
```
project/
├── main.py                          # Código principal
├── productos.csv                    # Dataset de productos
├── embedding_index_mapping.csv      # Mapeo de índices de embeddings
├── embeddings.npy                   # Embeddings precalculados
├── descriptions.npy                 # Descripciones correspondientes
├── ratings_aggregated.csv           # Ratings agregados (V2)
├── ratings_detailed.csv             # Ratings detallados (V2)
└── ratings.csv                      # Ratings básicos (V1 - fallback)
```

## 📋 Formato de Datos

### productos.csv
```csv
parent_asin,title,description,main_category,image_urls_best
B07NTK7T5P,Product Title,Product description text,Electronics,[url1,url2]
```

### embedding_index_mapping.csv
```csv
parent_asin,index
B07NTK7T5P,0
B0751M85FV,1
```

### ratings_aggregated.csv (V2)
```csv
parent_asin,average_rating,rating_count
B07NTK7T5P,4.5,150
```

### ratings_detailed.csv (V2)
```csv
user_id,parent_asin,rating,timestamp
user123,B07NTK7T5P,5,2023-01-01
```

## 🚀 Uso del Sistema

### Ejecución Básica
```bash
python main.py
```

### Interfaz Web
El sistema lanza una interfaz web en `http://localhost:7860` con las siguientes pestañas:

1. **🔍 Búsqueda por Descripción**
   - Introduce una descripción en inglés
   - Selecciona el método de búsqueda
   - Ajusta el número de resultados

2. **🤝 Recomendaciones Colaborativas**
   - Selecciona un producto base
   - Elige el método colaborativo
   - Obtén recomendaciones similares

3. **👤 Recomendaciones Basadas en Cliente**
   - Ingresa ASINs de productos seleccionados
   - Selecciona el método de recomendación
   - Genera recomendaciones personalizadas

4. **📊 Métricas y Comparación**
   - Visualiza métricas de rendimiento
   - Compara diferentes métodos
   - Analiza estadísticas del sistema

## 🔧 Configuración Avanzada

### Parámetros del Sistema
```python
# Configuración de filtrado colaborativo
min_ratings_per_user = 5
min_ratings_per_item = 5

# Configuración de clustering
n_clusters = min(100, len(df_products) // 10)

# Configuración de embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
```

### Personalización de Métricas
```python
# Pesos para el método híbrido
hybrid_weights = {
    'embedding_similarity': 0.5,
    'category_score': 0.3,
    'rating_score': 0.2
}
```

## 📊 Métricas de Evaluación

### Métricas Implementadas
- **Diversidad**: Variedad de categorías en las recomendaciones
- **Novedad**: Basada en popularidad inversa (rating)
- **Cobertura**: Porcentaje de items únicos recomendados
- **Precision@K**: Precisión en los top-K elementos
- **Tiempo de Ejecución**: Performance de cada método

### Interpretación de Resultados
```
Diversidad: 0.0 - 1.0 (mayor es mejor)
Novedad: 0.0 - 1.0 (mayor es mejor)
Cobertura: 0.0 - 1.0 (mayor es mejor)
Precision@5: 0.0 - 1.0 (mayor es mejor)
```

## 🎯 Ejemplos de Uso

### Búsqueda por Descripción
```python
# Ejemplo de búsqueda
descripcion = "Wireless bluetooth headphones with noise cancelling"
results = description_searcher.search_method_1_knn(descripcion, n_results=5)
```

### Recomendaciones Colaborativas
```python
# Ejemplo de recomendación colaborativa
target_asin = "B07NTK7T5P"
recommendations = collaborative_recommender.recommend_method_1_svd(target_asin)
```

### Recomendaciones Basadas en Cliente
```python
# Ejemplo de recomendación por cliente
selected_asins = ["B07NTK7T5P", "B0751M85FV", "B01HYNE114"]
recommendations = client_recommender.recommend_method_3_hybrid_approach(selected_asins)
```

## 🔍 Casos de Uso Recomendados

### Para E-commerce
- **Búsqueda de productos**: Usuarios buscan productos por descripción
- **Productos relacionados**: Mostrar productos similares en páginas de detalle
- **Recomendaciones personalizadas**: Basadas en historial de compras

### Para Análisis de Datos
- **Comparación de algoritmos**: Evaluar efectividad de diferentes métodos
- **Análisis de comportamiento**: Estudiar patrones de similaridad
- **Optimización de catálogo**: Identificar gaps en el inventario

## 📈 Optimización y Rendimiento

### Mejores Prácticas
1. **Preprocesamiento**: Mantener embeddings precalculados
2. **Filtrado**: Usar umbrales de rating mínimos
3. **Caché**: Implementar caché para consultas frecuentes
4. **Batch Processing**: Procesar múltiples consultas en lotes

### Escalabilidad
- Soporta hasta 100K+ productos
- Embeddings optimizados para memoria
- Matrices sparse para eficiencia
- Clustering para reducir complejidad

## 🛡️ Manejo de Errores

### Errores Comunes
- **Embeddings no encontrados**: El sistema genera embeddings básicos
- **Datos faltantes**: Usa valores por defecto y continúa
- **Inconsistencias**: Reporta warnings y sugiere regenerar embeddings

### Validaciones
- Verificación de consistencia entre embeddings y DataFrame
- Validación de formato de ASINs
- Comprobación de disponibilidad de ratings

## 🔧 Troubleshooting

### Problemas Frecuentes

**1. Error de consistencia de embeddings**
```python
# Solución: Regenerar embeddings
description_embeddings = model.encode(df_similars["description"].tolist())
np.save("embeddings.npy", description_embeddings)
```

**2. Sistema colaborativo no disponible**
```python
# Verificar estructura de ratings_detailed.csv
# Debe contener: user_id, parent_asin, rating
```

**3. Memoria insuficiente**
```python
# Reducir dimensiones de embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")  # Modelo más ligero
```

## 📝 Logs y Debugging

### Información de Debug
```python
# Activar warnings detallados
import warnings
warnings.filterwarnings('default')

# Verificar carga de datos
print(f"Productos cargados: {len(df_similars)}")
print(f"Embeddings: {description_embeddings.shape}")
print(f"Ratings: {len(ratings_dict)}")
```

## 🤝 Contribuciones

### Estructura del Código
- **Clases principales**: `DescriptionSearcher`, `CollaborativeRecommender`, `ClientBasedRecommender`
- **Métricas**: `RecommendationMetrics`
- **Interfaz**: Funciones Gradio para UI
- **Utilidades**: Funciones de limpieza y formato

### Extensiones Posibles
- Agregar más algoritmos de ML
- Implementar deep learning
- Añadir filtros por precio/categoría
- Integrar con APIs externas

## 📄 Licencia

Este proyecto está disponible bajo licencia MIT. Ver archivo LICENSE para más detalles.

## 📞 Soporte

Para reportar bugs o solicitar características:
- Crear un issue en el repositorio
- Incluir logs de error y configuración
- Proporcionar datos de ejemplo si es posible

---

**¡Disfruta explorando el mundo de las recomendaciones inteligentes!** 🚀