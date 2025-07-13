import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
import gradio as gr
import json
import re
from collections import defaultdict, Counter
import csv
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== CARGA DE DATOS ====================
df_productos = pd.read_csv("productos.csv")
df_productos = df_productos.reset_index(drop=True)
df_mapping = pd.read_csv("embedding_index_mapping.csv")

# Cargar ratings (adaptable a ambos formatos)
try:
    # Intentar cargar ratings agregados (V2)
    df_ratings_aggregated = pd.read_csv("ratings_aggregated.csv")
    df_ratings_detailed = pd.read_csv("ratings_detailed.csv")
    ratings_dict = df_ratings_aggregated.set_index('parent_asin')['average_rating'].to_dict()
    print(f"Ratings V2 cargados: {len(ratings_dict):,} productos con ratings")
    HAS_DETAILED_RATINGS = True
except FileNotFoundError:
    # Fallback a ratings V1
    df_ratings = pd.read_csv("ratings.csv")
    ratings_dict = df_ratings.set_index('parent_asin')['rating'].to_dict()
    df_ratings_detailed = df_ratings  # Para compatibilidad
    print(f"Ratings V1 cargados: {len(ratings_dict):,} productos con ratings")
    HAS_DETAILED_RATINGS = False

# ==================== PREPARACIÓN DE DATOS (SIN MERGE) ====================
# CRÍTICO: No hacer merge para preservar embeddings precargados
df_similars = df_productos[df_productos["parent_asin"].isin(df_mapping["parent_asin"])].reset_index(drop=True)

# Asegurarte de que el orden coincida
df_similars = df_similars.merge(df_mapping, on="parent_asin").sort_values("index").reset_index(drop=True)

# CRÍTICO: Resetear índices ANTES de cualquier operación
df_similars["description"] = df_similars["description"].fillna("").astype(str)
df_similars = df_similars.reset_index(drop=True)

print(f"Total de productos en df_similars: {len(df_similars):,}")
print(f"Productos únicos: {df_similars['parent_asin'].nunique():,}")

# ==================== CARGA DE EMBEDDINGS ====================
model = SentenceTransformer("all-MiniLM-L6-v2")

try:
    description_embeddings = np.load("embeddings.npy")
    descriptions = np.load("descriptions.npy", allow_pickle=True)
    print(f"Embeddings precalculados cargados: {description_embeddings.shape}")

    # VERIFICACIÓN CRÍTICA: Asegurar consistencia
    if len(description_embeddings) != len(df_similars):
        print(f"WARNING: Mismatch detectado!")
        print(f"   Embeddings: {len(description_embeddings)}")
        print(f"   df_similars: {len(df_similars)}")
        print("   Recomiendo regenerar embeddings con el nuevo df_similars")
    else:
        print("Consistencia verificada: embeddings y df_similars coinciden")

except FileNotFoundError:
    print("Embeddings no encontrados. Generando embeddings básicos...")
    description_embeddings = model.encode(df_similars["description"].tolist())
    descriptions = df_similars["description"].values

# ==================== SISTEMA DE MÉTRICAS Y EVALUACIÓN ====================
class RecommendationMetrics:
    """Sistema de métricas para evaluar y comparar diferentes enfoques de recomendación"""

    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.execution_times = defaultdict(list)

    def calculate_diversity(self, recommendations_asins):
        """Calcula la diversidad de las recomendaciones basada en categorías"""
        if not recommendations_asins:
            return 0.0

        categories = []
        for asin in recommendations_asins:
            product_row = df_similars[df_similars['parent_asin'] == asin]
            if len(product_row) > 0:
                category = product_row.iloc[0].get('main_category', 'Unknown')
                categories.append(category)

        if not categories:
            return 0.0

        unique_categories = len(set(categories))
        total_items = len(categories)
        return unique_categories / total_items

    def calculate_novelty(self, recommendations_asins):
        """Calcula la novedad basada en popularidad (rating y frecuencia)"""
        if not recommendations_asins:
            return 0.0

        novelty_scores = []
        for asin in recommendations_asins:
            rating = ratings_dict.get(asin, 0.0)
            # Mayor rating = menor novedad (productos populares)
            novelty_score = max(0, 5.0 - rating) / 5.0
            novelty_scores.append(novelty_score)

        return np.mean(novelty_scores) if novelty_scores else 0.0

    def calculate_coverage(self, recommendations_asins, total_available_items):
        """Calcula el coverage como porcentaje de items únicos recomendados"""
        unique_recommendations = len(set(recommendations_asins))
        return unique_recommendations / min(total_available_items, 100)  # Normalizar

    def calculate_precision_at_k(self, recommendations_asins, relevant_items, k=5):
        """Calcula precision@k"""
        if not recommendations_asins or not relevant_items:
            return 0.0

        top_k = recommendations_asins[:k]
        relevant_in_top_k = len(set(top_k) & set(relevant_items))
        return relevant_in_top_k / min(k, len(top_k))

    def evaluate_recommendations(self, method_name, recommendations_asins, execution_time,
                               relevant_items=None, total_available=1000):
        """Evalúa un conjunto de recomendaciones con múltiples métricas"""
        metrics = {
            'method': method_name,
            'timestamp': datetime.now(),
            'execution_time': execution_time,
            'num_recommendations': len(recommendations_asins),
            'diversity': self.calculate_diversity(recommendations_asins),
            'novelty': self.calculate_novelty(recommendations_asins),
            'coverage': self.calculate_coverage(recommendations_asins, total_available)
        }

        if relevant_items:
            metrics['precision_at_5'] = self.calculate_precision_at_k(recommendations_asins, relevant_items, 5)

        # Almacenar métricas
        for key, value in metrics.items():
            if key not in ['method', 'timestamp']:
                self.metrics_history[f"{method_name}_{key}"].append(value)

        return metrics

    def get_comparison_report(self):
        """Genera un reporte comparativo de todos los métodos evaluados"""
        if not self.metrics_history:
            return "No hay métricas disponibles"

        report = "# 📊 REPORTE COMPARATIVO DE MÉTODOS\n\n"

        # Agrupar métricas por tipo
        methods = set()
        for key in self.metrics_history.keys():
            method = key.split('_')[0] + '_' + key.split('_')[1]
            methods.add(method)

        for method in sorted(methods):
            report += f"## {method.replace('_', ' ').title()}\n"

            # Buscar métricas de este método
            method_metrics = {}
            for key, values in self.metrics_history.items():
                if key.startswith(method):
                    metric_name = '_'.join(key.split('_')[2:])
                    if values:
                        method_metrics[metric_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'count': len(values)
                        }

            for metric, stats in method_metrics.items():
                report += f"- **{metric.replace('_', ' ').title()}**: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})\n"

            report += "\n"

        return report

# Instancia global de métricas
metrics_evaluator = RecommendationMetrics()

# ==================== FUNCIONALIDAD 1: BÚSQUEDA POR DESCRIPCIÓN (3 MÉTODOS) ====================

class DescriptionSearcher:
    """Sistema de búsqueda por descripción con múltiples enfoques"""

    def __init__(self, df_products, embeddings, model):
        self.df_products = df_products
        self.embeddings = embeddings
        self.model = model
        self.setup_methods()

    def setup_methods(self):
        """Configura los diferentes métodos de búsqueda"""
        # Método 1: KNN con embeddings (original)
        self.knn = NearestNeighbors(n_neighbors=50, metric="cosine")
        self.knn.fit(self.embeddings)

        # Método 2: TF-IDF + Cosine Similarity
        descriptions_text = self.df_products["description"].fillna("").tolist()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(descriptions_text)

        # Método 3: Clustering + Embedding similarity
        self.n_clusters = min(100, len(self.df_products) // 10)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(self.embeddings)

    def search_method_1_knn(self, query, n_results=5):
        """Método 1: KNN con embeddings (original mejorado)"""
        start_time = time.time()

        query_embedding = self.model.encode([query])
        distances, indices = self.knn.kneighbors(query_embedding, n_neighbors=min(50, len(self.df_products)))

        results = []
        seen_asins = set()

        for i, idx in enumerate(indices[0]):
            if len(results) >= n_results:
                break

            if idx >= len(self.df_products):
                continue

            row = self.df_products.iloc[idx]
            asin = row.get("parent_asin", "N/A")

            if asin in seen_asins:
                continue
            seen_asins.add(asin)

            similarity_score = 1 - distances[0][i]  # Convertir distancia a similitud
            results.append({
                'asin': asin,
                'similarity_score': similarity_score,
                'method': 'KNN_Embeddings'
            })

        execution_time = time.time() - start_time

        # Evaluar con métricas
        result_asins = [r['asin'] for r in results]
        metrics = metrics_evaluator.evaluate_recommendations(
            'search_knn', result_asins, execution_time
        )

        return results, metrics

    def search_method_2_tfidf(self, query, n_results=5):
        """Método 2: TF-IDF + Cosine Similarity"""
        start_time = time.time()

        query_tfidf = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()

        # Obtener top resultados
        top_indices = np.argsort(similarities)[::-1]

        results = []
        seen_asins = set()

        for idx in top_indices:
            if len(results) >= n_results:
                break

            if similarities[idx] < 0.01:  # Umbral mínimo de similitud
                continue

            row = self.df_products.iloc[idx]
            asin = row.get("parent_asin", "N/A")

            if asin in seen_asins:
                continue
            seen_asins.add(asin)

            results.append({
                'asin': asin,
                'similarity_score': similarities[idx],
                'method': 'TF_IDF'
            })

        execution_time = time.time() - start_time

        # Evaluar con métricas
        result_asins = [r['asin'] for r in results]
        metrics = metrics_evaluator.evaluate_recommendations(
            'search_tfidf', result_asins, execution_time
        )

        return results, metrics

    def search_method_3_cluster(self, query, n_results=5):
        """Método 3: Clustering + Embedding similarity"""
        start_time = time.time()

        query_embedding = self.model.encode([query])

        # Encontrar cluster más similar
        query_cluster = self.kmeans.predict(query_embedding)[0]

        # Filtrar productos del mismo cluster
        cluster_mask = self.cluster_labels == query_cluster
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            execution_time = time.time() - start_time
            return [], {'method': 'Cluster_Search', 'execution_time': execution_time}

        # Calcular similitudes dentro del cluster
        cluster_embeddings = self.embeddings[cluster_indices]
        similarities = cosine_similarity(query_embedding, cluster_embeddings).flatten()

        # Ordenar por similitud
        sorted_cluster_indices = cluster_indices[np.argsort(similarities)[::-1]]

        results = []
        seen_asins = set()

        for idx in sorted_cluster_indices:
            if len(results) >= n_results:
                break

            row = self.df_products.iloc[idx]
            asin = row.get("parent_asin", "N/A")

            if asin in seen_asins:
                continue
            seen_asins.add(asin)

            similarity_idx = np.where(cluster_indices == idx)[0][0]
            similarity_score = similarities[similarity_idx]

            results.append({
                'asin': asin,
                'similarity_score': similarity_score,
                'method': 'Cluster_Search'
            })

        execution_time = time.time() - start_time

        # Evaluar con métricas
        result_asins = [r['asin'] for r in results]
        metrics = metrics_evaluator.evaluate_recommendations(
            'search_cluster', result_asins, execution_time
        )

        return results, metrics

# ==================== FUNCIONALIDAD 2: RECOMENDACIÓN COLABORATIVA (3 MÉTODOS) ====================

class CollaborativeRecommender:
    """Sistema de recomendación colaborativa con múltiples enfoques"""

    def __init__(self, ratings_df, min_ratings_per_user=5, min_ratings_per_item=5):
        self.ratings_df = ratings_df.copy()
        self.min_ratings_per_user = min_ratings_per_user
        self.min_ratings_per_item = min_ratings_per_item
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.svd_model = None
        self.nmf_model = None
        self.user_encoder = {}
        self.item_encoder = {}
        self.user_decoder = {}
        self.item_decoder = {}

        self._prepare_data()
        self._build_matrices()

    def _prepare_data(self):
        """Prepara los datos filtrando usuarios e items con pocas interacciones"""
        print("Preparando datos para filtrado colaborativo...")

        # Filtrar usuarios con al menos min_ratings_per_user ratings
        user_counts = self.ratings_df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= self.min_ratings_per_user].index

        # Filtrar items con al menos min_ratings_per_item ratings
        item_counts = self.ratings_df['parent_asin'].value_counts()
        valid_items = item_counts[item_counts >= self.min_ratings_per_item].index

        # Aplicar filtros
        self.ratings_df = self.ratings_df[
            (self.ratings_df['user_id'].isin(valid_users)) &
            (self.ratings_df['parent_asin'].isin(valid_items))
        ]

        print(f"Datos filtrados: {len(self.ratings_df):,} ratings, "
              f"{self.ratings_df['user_id'].nunique():,} usuarios, "
              f"{self.ratings_df['parent_asin'].nunique():,} productos")

        # Crear encoders
        unique_users = self.ratings_df['user_id'].unique()
        unique_items = self.ratings_df['parent_asin'].unique()

        self.user_encoder = {user: idx for idx, user in enumerate(unique_users)}
        self.item_encoder = {item: idx for idx, item in enumerate(unique_items)}
        self.user_decoder = {idx: user for user, idx in self.user_encoder.items()}
        self.item_decoder = {idx: item for item, idx in self.item_encoder.items()}

    def _build_matrices(self):
        """Construye las matrices necesarias para la recomendación"""
        print("Construyendo matrices de interacción...")

        n_users = len(self.user_encoder)
        n_items = len(self.item_encoder)

        # Mapear a índices numéricos
        user_indices = self.ratings_df['user_id'].map(self.user_encoder)
        item_indices = self.ratings_df['parent_asin'].map(self.item_encoder)
        ratings = self.ratings_df['rating'].values

        # Crear matriz sparse
        self.user_item_matrix = csr_matrix(
            (ratings, (user_indices, item_indices)),
            shape=(n_users, n_items)
        )

        # Método 1: SVD para similitud entre items
        print("Calculando similitudes SVD...")
        self.svd_model = TruncatedSVD(
            n_components=min(50, min(n_users, n_items)-1),
            random_state=42
        )
        item_features_svd = self.svd_model.fit_transform(self.user_item_matrix.T)
        self.item_similarity_matrix = cosine_similarity(item_features_svd)

        # Método 2: NMF para factorización
        print("Calculando factorización NMF...")
        self.nmf_model = NMF(
            n_components=min(30, min(n_users, n_items)-1),
            random_state=42,
            max_iter=200
        )
        self.user_features_nmf = self.nmf_model.fit_transform(self.user_item_matrix)
        self.item_features_nmf = self.nmf_model.components_.T

        # Método 3: Item-based cosine similarity directo
        print("Calculando similitud directa...")
        self.item_similarity_direct = cosine_similarity(self.user_item_matrix.T)

        print("Matrices construidas exitosamente")

    def recommend_method_1_svd(self, target_item, n_recommendations=4):
        """Método 1: Recomendaciones basadas en SVD"""
        start_time = time.time()

        if target_item not in self.item_encoder:
            return [], {'method': 'SVD_Collaborative', 'execution_time': 0}

        target_idx = self.item_encoder[target_item]
        similarities = self.item_similarity_matrix[target_idx]

        # Obtener items más similares (excluyendo el item objetivo)
        similar_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]

        recommendations = []
        for idx in similar_indices:
            item_id = self.item_decoder[idx]
            similarity_score = similarities[idx]
            recommendations.append({
                'asin': item_id,
                'similarity_score': similarity_score,
                'method': 'SVD_Collaborative'
            })

        execution_time = time.time() - start_time

        # Evaluar con métricas
        result_asins = [r['asin'] for r in recommendations]
        metrics = metrics_evaluator.evaluate_recommendations(
            'collab_svd', result_asins, execution_time
        )

        return recommendations, metrics

    def recommend_method_2_nmf(self, target_item, n_recommendations=4):
        """Método 2: Recomendaciones basadas en NMF"""
        start_time = time.time()

        if target_item not in self.item_encoder:
            return [], {'method': 'NMF_Collaborative', 'execution_time': 0}

        target_idx = self.item_encoder[target_item]
        target_features = self.item_features_nmf[target_idx]

        # Calcular similitudes con todos los items
        similarities = cosine_similarity([target_features], self.item_features_nmf).flatten()

        # Obtener items más similares (excluyendo el item objetivo)
        similar_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]

        recommendations = []
        for idx in similar_indices:
            item_id = self.item_decoder[idx]
            similarity_score = similarities[idx]
            recommendations.append({
                'asin': item_id,
                'similarity_score': similarity_score,
                'method': 'NMF_Collaborative'
            })

        execution_time = time.time() - start_time

        # Evaluar con métricas
        result_asins = [r['asin'] for r in recommendations]
        metrics = metrics_evaluator.evaluate_recommendations(
            'collab_nmf', result_asins, execution_time
        )

        return recommendations, metrics

    def recommend_method_3_direct(self, target_item, n_recommendations=4):
        """Método 3: Similitud directa item-to-item"""
        start_time = time.time()

        if target_item not in self.item_encoder:
            return [], {'method': 'Direct_Collaborative', 'execution_time': 0}

        target_idx = self.item_encoder[target_item]
        similarities = self.item_similarity_direct[target_idx]

        # Obtener items más similares (excluyendo el item objetivo)
        similar_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]

        recommendations = []
        for idx in similar_indices:
            item_id = self.item_decoder[idx]
            similarity_score = similarities[idx]
            recommendations.append({
                'asin': item_id,
                'similarity_score': similarity_score,
                'method': 'Direct_Collaborative'
            })

        execution_time = time.time() - start_time

        # Evaluar con métricas
        result_asins = [r['asin'] for r in recommendations]
        metrics = metrics_evaluator.evaluate_recommendations(
            'collab_direct', result_asins, execution_time
        )

        return recommendations, metrics

    def get_available_items(self):
        """Retorna lista de items disponibles para recomendación"""
        return list(self.item_encoder.keys())

# ==================== FUNCIONALIDAD 3: RECOMENDACIÓN BASADA EN CLIENTE (3 MÉTODOS) ====================

class ClientBasedRecommender:
    """Sistema de recomendación basado en productos seleccionados por un cliente"""

    def __init__(self, df_products, embeddings, ratings_dict):
        self.df_products = df_products
        self.embeddings = embeddings
        self.ratings_dict = ratings_dict
        self.setup_methods()

    def setup_methods(self):
        """Configura mapeo ASIN -> índice posicional"""
        self.asin_to_idx = {}

        parent_asins = self.df_products["parent_asin"].values

        for idx, asin in enumerate(parent_asins):
            if pd.notna(asin):
                self.asin_to_idx[asin] = idx

        self.prepare_content_features()

    # Verificación crítica
        assert len(self.embeddings) == len(self.df_products), \
            f"ERROR: embeddings ({len(self.embeddings)}) y dataframe ({len(self.df_products)}) NO coinciden."


    def prepare_content_features(self):
        """Prepara características de contenido para recomendaciones"""
        # Extraer categorías principales
        categories = self.df_products.get('main_category', pd.Series(['Unknown'] * len(self.df_products)))
        self.unique_categories = list(set(categories.fillna('Unknown')))

        # Crear matriz de características categóricas
        self.category_features = np.zeros((len(self.df_products), len(self.unique_categories)))
        for idx, category in enumerate(categories.fillna('Unknown')):
            if category in self.unique_categories:
                cat_idx = self.unique_categories.index(category)
                self.category_features[idx, cat_idx] = 1

    def recommend_method_1_profile_similarity(self, selected_asins, n_recommendations=5):
        """Método 1: Perfil de usuario basado en similitud de embeddings"""
        start_time = time.time()

        if not selected_asins:
            return [], {'method': 'Profile_Similarity', 'execution_time': 0}

        # Obtener embeddings de productos seleccionados
        selected_embeddings = []
        valid_asins = []

        for asin in selected_asins:
            if asin in self.asin_to_idx:
                idx = self.asin_to_idx[asin]
                selected_embeddings.append(self.embeddings[idx])
                valid_asins.append(asin)

        if not selected_embeddings:
            return [], {'method': 'Profile_Similarity', 'execution_time': 0}

        # Crear perfil de usuario como promedio de embeddings
        user_profile = np.mean(selected_embeddings, axis=0)

        # Calcular similitudes con todos los productos
        similarities = cosine_similarity([user_profile], self.embeddings).flatten()

        # Excluir productos ya seleccionados
        excluded_indices = [self.asin_to_idx[asin] for asin in valid_asins if asin in self.asin_to_idx]
        for idx in excluded_indices:
            similarities[idx] = -1

        # Obtener top recomendaciones
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]

        recommendations = []
        for idx in top_indices:
            if similarities[idx] <= 0:
                continue

            row = self.df_products.iloc[idx]
            asin = row.get('parent_asin')
            if asin:
                recommendations.append({
                    'asin': asin,
                    'similarity_score': similarities[idx],
                    'method': 'Profile_Similarity'
                })

        execution_time = time.time() - start_time

        # Evaluar con métricas
        result_asins = [r['asin'] for r in recommendations]
        metrics = metrics_evaluator.evaluate_recommendations(
            'client_profile', result_asins, execution_time
        )

        return recommendations, metrics

    def recommend_method_2_weighted_categories(self, selected_asins, n_recommendations=5):
        """Método 2: Recomendación basada en categorías ponderadas"""
        start_time = time.time()

        if not selected_asins:
            return [], {'method': 'Weighted_Categories', 'execution_time': 0}

        # Contar categorías en productos seleccionados
        category_weights = defaultdict(float)
        valid_selections = 0

        for asin in selected_asins:
            if asin in self.asin_to_idx:
                idx = self.asin_to_idx[asin]
                row = self.df_products.iloc[idx]
                category = row.get('main_category', 'Unknown')

                # Ponderar por rating del producto
                rating = self.ratings_dict.get(asin, 3.0)
                category_weights[category] += rating / 5.0  # Normalizar rating
                valid_selections += 1

        if not category_weights:
            return [], {'method': 'Weighted_Categories', 'execution_time': 0}

        # Normalizar pesos
        total_weight = sum(category_weights.values())
        for category in category_weights:
            category_weights[category] /= total_weight

        # Calcular scores para todos los productos
        product_scores = []
        excluded_asins = set(selected_asins)

        for idx, row in self.df_products.iterrows():
            asin = row.get('parent_asin')
            if not asin or asin in excluded_asins:
                continue

            category = row.get('main_category', 'Unknown')
            category_score = category_weights.get(category, 0.0)

            # Combinar con rating del producto
            product_rating = self.ratings_dict.get(asin, 0.0)
            final_score = category_score * 0.7 + (product_rating / 5.0) * 0.3

            product_scores.append({
                'asin': asin,
                'similarity_score': final_score,
                'method': 'Weighted_Categories'
            })

        # Ordenar por score y tomar top N
        product_scores.sort(key=lambda x: x['similarity_score'], reverse=True)
        recommendations = product_scores[:n_recommendations]

        execution_time = time.time() - start_time

        # Evaluar con métricas
        result_asins = [r['asin'] for r in recommendations]
        metrics = metrics_evaluator.evaluate_recommendations(
            'client_categories', result_asins, execution_time
        )

        return recommendations, metrics

    def recommend_method_3_hybrid_approach(self, selected_asins, n_recommendations=5):
        """Método 3: Enfoque híbrido combinando embeddings, categorías y ratings"""
        start_time = time.time()

        if not selected_asins:
            return [], {'method': 'Hybrid_Approach', 'execution_time': 0}

        # Paso 1: Crear perfil de embeddings
        selected_embeddings = []
        selected_categories = []
        selected_ratings = []
        valid_asins = []

        for asin in selected_asins:
            if asin in self.asin_to_idx:
                idx = self.asin_to_idx[asin]
                row = self.df_products.iloc[idx]

                selected_embeddings.append(self.embeddings[idx])
                selected_categories.append(row.get('main_category', 'Unknown'))
                selected_ratings.append(self.ratings_dict.get(asin, 3.0))
                valid_asins.append(asin)

        if not selected_embeddings:
            return [], {'method': 'Hybrid_Approach', 'execution_time': 0}

        # Crear perfil promedio ponderado por rating
        weights = np.array(selected_ratings) / 5.0  # Normalizar ratings
        weights = weights / np.sum(weights)  # Normalizar pesos

        user_profile = np.average(selected_embeddings, axis=0, weights=weights)

        # Paso 2: Calcular preferencias de categoría
        category_preferences = Counter(selected_categories)
        total_selections = len(selected_categories)

        # Paso 3: Evaluar todos los productos candidatos
        candidate_scores = []
        excluded_asins = set(selected_asins)

        for idx, row in self.df_products.iterrows():
            asin = row.get('parent_asin')
            if not asin or asin in excluded_asins:
                continue

            # Score de similitud de embedding
            embedding_similarity = cosine_similarity([user_profile], [self.embeddings[idx]])[0][0]

            # Score de categoría
            category = row.get('main_category', 'Unknown')
            category_score = category_preferences.get(category, 0) / total_selections

            # Score de rating
            product_rating = self.ratings_dict.get(asin, 0.0)
            rating_score = product_rating / 5.0

            # Combinación ponderada
            hybrid_score = (
                embedding_similarity * 0.5 +
                category_score * 0.3 +
                rating_score * 0.2
            )

            candidate_scores.append({
                'asin': asin,
                'similarity_score': hybrid_score,
                'method': 'Hybrid_Approach',
                'embedding_sim': embedding_similarity,
                'category_score': category_score,
                'rating_score': rating_score
            })

        # Ordenar y tomar top N
        candidate_scores.sort(key=lambda x: x['similarity_score'], reverse=True)
        recommendations = candidate_scores[:n_recommendations]

        execution_time = time.time() - start_time

        # Evaluar con métricas
        result_asins = [r['asin'] for r in recommendations]
        metrics = metrics_evaluator.evaluate_recommendations(
            'client_hybrid', result_asins, execution_time
        )

        return recommendations, metrics

# ==================== FUNCIONES DE UTILIDAD ====================
def clean_description(description):
    """Limpia la descripción eliminando corchetes y su contenido"""
    if not description or description == "":
        return "Sin descripción"

    if description.strip().startswith('[') and description.strip().endswith(']'):
        cleaned = description.strip()[1:-1].strip()
    else:
        cleaned = re.sub(r'\[.*?\]', '', description)

    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned if cleaned else "Sin descripción"

def get_best_image_url(row):
    """Extrae la mejor URL de imagen disponible"""
    image_columns = ['image_urls_best', 'image_urls_large', 'image_urls_all']

    for col in image_columns:
        if col in row:
            try:
                images = json.loads(row[col]) if isinstance(row[col], str) else row[col]
                if isinstance(images, list) and images:
                    for img_url in images:
                        if img_url and isinstance(img_url, str) and img_url.startswith("http"):
                            return img_url
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

    return "https://via.placeholder.com/300x300.png?text=No+Image"

def get_product_rating(asin):
    """Obtiene el rating de un producto desde el diccionario de ratings"""
    return ratings_dict.get(asin, 0.0)

def get_product_info_by_asin(asin):
    """Obtiene información de un producto por su ASIN"""
    product_row = df_similars[df_similars['parent_asin'] == asin]
    if len(product_row) == 0:
        return None

    row = product_row.iloc[0]
    return {
        'asin': asin,
        'title': row.get('title', 'Sin título'),
        'description': clean_description(row.get('description', '')),
        'rating': get_product_rating(asin),
        'image_url': get_best_image_url(row),
        'category': row.get('main_category', 'Unknown')
    }

# ==================== INICIALIZACIÓN DE SISTEMAS ====================

# Inicializar sistema de búsqueda por descripción
description_searcher = DescriptionSearcher(df_similars, description_embeddings, model)

# Inicializar sistema colaborativo solo si hay ratings detallados
if HAS_DETAILED_RATINGS:
    print("Inicializando sistema de recomendación colaborativo...")
    collaborative_recommender = CollaborativeRecommender(df_ratings_detailed)
else:
    print("Sistema colaborativo no disponible (requiere ratings detallados)")
    collaborative_recommender = None

# Inicializar sistema basado en cliente
client_recommender = ClientBasedRecommender(df_similars, description_embeddings, ratings_dict)

# ==================== FUNCIONES DE INTERFAZ MEJORADAS ====================

def search_products_enhanced(descripcion_input, method_choice, max_images_per_product=2, target_products=5):
    """Búsqueda mejorada con selección de método"""
    if not descripcion_input.strip():
        return [("https://via.placeholder.com/300.png?text=Vacío", "Por favor escribe algo para buscar...")]

    # Seleccionar método de búsqueda
    if method_choice == "KNN + Embeddings":
        results, metrics = description_searcher.search_method_1_knn(descripcion_input, target_products)
    elif method_choice == "TF-IDF + Cosine":
        results, metrics = description_searcher.search_method_2_tfidf(descripcion_input, target_products)
    elif method_choice == "Clustering + Embeddings":
        results, metrics = description_searcher.search_method_3_cluster(descripcion_input, target_products)
    else:
        # Comparar todos los métodos
        results_knn, metrics_knn = description_searcher.search_method_1_knn(descripcion_input, 2)
        results_tfidf, metrics_tfidf = description_searcher.search_method_2_tfidf(descripcion_input, 2)
        results_cluster, metrics_cluster = description_searcher.search_method_3_cluster(descripcion_input, 2)

        # Combinar resultados
        all_results = results_knn + results_tfidf + results_cluster
        results = sorted(all_results, key=lambda x: x['similarity_score'], reverse=True)[:target_products]

        metrics = {
            'method': 'All_Methods_Combined',
            'knn_time': metrics_knn.get('execution_time', 0),
            'tfidf_time': metrics_tfidf.get('execution_time', 0),
            'cluster_time': metrics_cluster.get('execution_time', 0)
        }

    # Convertir resultados a formato de galería
    gallery_results = []
    for result in results:
        product_info = get_product_info_by_asin(result['asin'])
        if product_info:
            texto = f"🔍 Método: {result['method']}\n"
            texto += f"📦 {product_info['title']}\n"
            texto += f"⭐ Rating: {product_info['rating']:.2f}\n"
            texto += f"🎯 Similitud: {result['similarity_score']:.3f}\n"
            texto += f"📂 Categoría: {product_info['category']}\n\n"
            texto += f"📝 {product_info['description'][:200]}{'...' if len(product_info['description']) > 200 else ''}"

            gallery_results.append((product_info['image_url'], texto))

    return gallery_results

def get_collaborative_recommendations_enhanced(selected_product_asin, method_choice):
    """Recomendaciones colaborativas mejoradas con selección de método"""
    if not collaborative_recommender:
        return [("https://via.placeholder.com/300.png?text=No+Disponible", "Sistema colaborativo no disponible")]

    if not selected_product_asin:
        return [("https://via.placeholder.com/300.png?text=Vacío", "Por favor selecciona un producto...")]

    # Seleccionar método colaborativo
    if method_choice == "SVD":
        recommendations, metrics = collaborative_recommender.recommend_method_1_svd(selected_product_asin)
    elif method_choice == "NMF":
        recommendations, metrics = collaborative_recommender.recommend_method_2_nmf(selected_product_asin)
    elif method_choice == "Direct Similarity":
        recommendations, metrics = collaborative_recommender.recommend_method_3_direct(selected_product_asin)
    else:
        # Comparar todos los métodos
        rec_svd, met_svd = collaborative_recommender.recommend_method_1_svd(selected_product_asin, 2)
        rec_nmf, met_nmf = collaborative_recommender.recommend_method_2_nmf(selected_product_asin, 2)
        rec_direct, met_direct = collaborative_recommender.recommend_method_3_direct(selected_product_asin, 1)

        recommendations = rec_svd + rec_nmf + rec_direct
        metrics = {'method': 'All_Collaborative_Methods'}

    if not recommendations:
        return [("https://via.placeholder.com/300.png?text=Sin+Recomendaciones", "No se encontraron recomendaciones para este producto.")]

    # Convertir a formato de galería
    gallery_results = []
    for rec in recommendations:
        product_info = get_product_info_by_asin(rec['asin'])
        if product_info:
            texto = f"🤝 Método: {rec['method']}\n"
            texto += f"📦 {product_info['title']}\n"
            texto += f"⭐ Rating: {product_info['rating']:.2f}\n"
            texto += f"🎯 Similitud: {rec['similarity_score']:.3f}\n"
            texto += f"📂 Categoría: {product_info['category']}\n\n"
            texto += f"📝 {product_info['description'][:200]}{'...' if len(product_info['description']) > 200 else ''}"

            gallery_results.append((product_info['image_url'], texto))

    return gallery_results

def get_client_recommendations(selected_asins_text, method_choice, n_recommendations=5):
    """Recomendaciones basadas en cliente con selección de método"""
    if not selected_asins_text.strip():
        return [("https://via.placeholder.com/300.png?text=Vacío", "Por favor ingresa ASINs de productos...")]

    # Parsear ASINs (separados por comas, espacios o nuevas líneas)
    selected_asins = []
    for asin in re.split(r'[,\s\n]+', selected_asins_text.strip()):
        asin = asin.strip()
        if asin:
            selected_asins.append(asin)

    if not selected_asins:
        return [("https://via.placeholder.com/300.png?text=Error", "No se pudieron parsear los ASINs")]

    # Seleccionar método de recomendación basada en cliente
    if method_choice == "Profile Similarity":
        recommendations, metrics = client_recommender.recommend_method_1_profile_similarity(selected_asins, n_recommendations)
    elif method_choice == "Weighted Categories":
        recommendations, metrics = client_recommender.recommend_method_2_weighted_categories(selected_asins, n_recommendations)
    elif method_choice == "Hybrid Approach":
        recommendations, metrics = client_recommender.recommend_method_3_hybrid_approach(selected_asins, n_recommendations)
    else:
        # Comparar todos los métodos
        rec_profile, met_profile = client_recommender.recommend_method_1_profile_similarity(selected_asins, 2)
        rec_categories, met_categories = client_recommender.recommend_method_2_weighted_categories(selected_asins, 2)
        rec_hybrid, met_hybrid = client_recommender.recommend_method_3_hybrid_approach(selected_asins, 1)

        recommendations = rec_profile + rec_categories + rec_hybrid
        metrics = {'method': 'All_Client_Methods'}

    if not recommendations:
        return [("https://via.placeholder.com/300.png?text=Sin+Recomendaciones", "No se encontraron recomendaciones para los productos seleccionados.")]

    # Convertir a formato de galería
    gallery_results = []
    for rec in recommendations:
        product_info = get_product_info_by_asin(rec['asin'])
        if product_info:
            texto = f"👤 Método: {rec['method']}\n"
            texto += f"📦 {product_info['title']}\n"
            texto += f"⭐ Rating: {product_info['rating']:.2f}\n"
            texto += f"🎯 Score: {rec['similarity_score']:.3f}\n"
            texto += f"📂 Categoría: {product_info['category']}\n\n"

            # Información adicional para método híbrido
            if 'embedding_sim' in rec:
                texto += f"🔗 Sim. Embedding: {rec['embedding_sim']:.3f}\n"
                texto += f"📂 Score Categoría: {rec['category_score']:.3f}\n"
                texto += f"⭐ Score Rating: {rec['rating_score']:.3f}\n\n"

            texto += f"📝 {product_info['description'][:150]}{'...' if len(product_info['description']) > 150 else ''}"

            gallery_results.append((product_info['image_url'], texto))

    return gallery_results

def get_product_options():
    """Obtiene lista de productos disponibles para el dropdown"""
    if not collaborative_recommender:
        return [("Sistema no disponible", "")]

    available_asins = collaborative_recommender.get_available_items()
    options = []

    for asin in available_asins[:100]:  # Limitar para performance
        product_info = get_product_info_by_asin(asin)
        if product_info and product_info['rating'] > 0:
            label = f"{product_info['title'][:50]}... (Rating: {product_info['rating']:.1f})"
            options.append((label, asin))

    return options

def get_metrics_report():
    """Genera reporte de métricas para mostrar en la interfaz"""
    return metrics_evaluator.get_comparison_report()

# ==================== INTERFAZ GRADIO MEJORADA ====================
def create_enhanced_interface():
    """Crea la interfaz mejorada con todas las funcionalidades y métricas"""

    with gr.Blocks(title="🚀 Advanced Product Recommendaation System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🚀 Advanced Product Recommendaation System

        **Funcionalidades disponibles:**
        - 🔍 **Búsqueda por Descripción** (3 métodos: KNN+Embeddings, TF-IDF+Cosine, Clustering+Embeddings)
        - 🤝 **Recomendación Colaborativa** (3 métodos: SVD, NMF, Direct Similarity)
        - 👤 **Recomendación Basada en Cliente** (3 métodos: Profile Similarity, Weighted Categories, Hybrid Approach)
        - 📊 **Métricas y Comparación** en tiempo real
        """)

        with gr.Tabs():
            # TAB 1: Búsqueda por descripción mejorada
            with gr.TabItem("🔍 Búsqueda por Descripción"):
                gr.Markdown("### Describe el producto que buscas en inglés y selecciona el método de búsqueda")

                with gr.Row():
                    with gr.Column(scale=1):
                        descripcion_input = gr.Textbox(
                            label="Describe your ideal product",
                            placeholder="exp: Handmade shungite bead bracelet, Silver necklace, etc."
                        )
                        search_method = gr.Dropdown(
                            choices=["KNN + Embeddings", "TF-IDF + Cosine", "Clustering + Embeddings", "Comparar Todos"],
                            value="KNN + Embeddings",
                            label="Método de búsqueda"
                        )
                        max_images = gr.Slider(
                            minimum=1, maximum=3, value=2, step=1,
                            label="Máximo de imágenes por producto"
                        )
                        num_products = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="Número de productos a mostrar"
                        )
                        search_btn = gr.Button("🔍 Buscar Productos", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        search_gallery = gr.Gallery(
                            label="Productos Encontrados",
                            columns=3,
                            rows=2,
                            height="auto"
                        )

            # TAB 2: Recomendaciones colaborativas mejoradas
            if collaborative_recommender:
                with gr.TabItem("🤝 Recomendaciones Colaborativas"):
                    gr.Markdown("### Selecciona un producto base y el método de recomendación colaborativa")

                    with gr.Row():
                        with gr.Column(scale=1):
                            product_dropdown = gr.Dropdown(
                                choices=get_product_options(),
                                label="Selecciona un producto base",
                                value=None
                            )
                            collab_method = gr.Dropdown(
                                choices=["SVD", "NMF", "Direct Similarity", "Comparar Todos"],
                                value="SVD",
                                label="Método colaborativo"
                            )
                            recommend_btn = gr.Button("🤝 Obtener Recomendaciones", variant="primary", size="lg")
                            refresh_products_btn = gr.Button("🔄 Actualizar Lista")

                        with gr.Column(scale=2):
                            recommendations_gallery = gr.Gallery(
                                label="Recomendaciones Colaborativas",
                                columns=2,
                                rows=2,
                                height="auto"
                            )

            # TAB 3: Recomendaciones basadas en cliente (NUEVA FUNCIONALIDAD)
            with gr.TabItem("👤 Recomendaciones Basadas en Cliente"):
                gr.Markdown("""
                ### Ingresa los ASINs de productos que un cliente ha seleccionado
                **Formato:** Separa los ASINs con comas, espacios o nuevas líneas
                **Ejemplo 1:** B07NTK7T5P, B0751M85FV, B01HYNE114, B0BKBJT5MM.
                **Ejemplo 2:** B01BAN3CBE, B0754TWHPT, B079KM6HDM, B097B8WH61.
                **Ejemplo 3:** B0B8WK62Z3, B01BYCH44W, B0BGNQ3CLH, B084L4PF4M.
                            
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        client_asins_input = gr.Textbox(
                            label="ASINs de productos seleccionados por el cliente",
                            placeholder="Insert here the product´s ID´s to get other products you might enjoy!",
                            lines=3
                        )
                        client_method = gr.Dropdown(
                            choices=["Profile Similarity", "Weighted Categories", "Hybrid Approach", "Comparar Todos"],
                            value="Hybrid Approach",
                            label="Método de recomendación"
                        )
                        client_num_recs = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="Número de recomendaciones"
                        )
                        client_recommend_btn = gr.Button("👤 Generar Recomendaciones", variant="primary", size="lg")

                        with gr.Accordion("ℹ️ Información de Métodos", open=False):
                            gr.Markdown("""
                            **Profile Similarity:** Crea un perfil promedio basado en los embeddings de los productos seleccionados

                            **Weighted Categories:** Recomienda basándose en las categorías más frecuentes, ponderadas por rating

                            **Hybrid Approach:** Combina embeddings, categorías y ratings con pesos optimizados
                            """)

                    with gr.Column(scale=2):
                        client_gallery = gr.Gallery(
                            label="Recomendaciones para el Cliente",
                            columns=3,
                            rows=2,
                            height="auto"
                        )

            # TAB 4: Métricas y comparación
            with gr.TabItem("📊 Métricas y Comparación"):
                gr.Markdown("### Análisis de rendimiento y comparación de métodos")

                with gr.Row():
                    with gr.Column():
                        metrics_btn = gr.Button("📊 Actualizar Métricas", variant="secondary")
                        clear_metrics_btn = gr.Button("🗑️ Limpiar Historial")

                metrics_output = gr.Markdown("Ejecuta algunas recomendaciones para ver las métricas...")

        # Estadísticas del sistema
        with gr.Accordion("📈 Estadísticas del Sistema", open=False):
            collab_stats = ""
            if collaborative_recommender:
                collab_stats = f"""
                - 🤝 Productos disponibles para recomendación colaborativa: {len(collaborative_recommender.get_available_items()):,}
                - 🧮 Dimensiones de matriz SVD: {collaborative_recommender.item_similarity_matrix.shape}
                """

            stats_text = f"""
            **Estadísticas del Sistema Completo:**
            - 📊 Total de productos: {len(df_similars):,}
            - ⭐ Productos con ratings: {len(ratings_dict):,}
            - 🔍 Embeddings precalculados: {len(description_embeddings):,}
            - ✅ Consistencia verificada: {len(description_embeddings) == len(df_similars)}
            - 🎯 Métodos de búsqueda: 3 implementados
            - 🤝 Métodos colaborativos: {"3 implementados" if collaborative_recommender else "No disponible"}
            - 👤 Métodos basados en cliente: 3 implementados
            {collab_stats}
            """
            gr.Markdown(stats_text)

        # ==================== EVENTOS ====================

        # Búsqueda por descripción
        search_btn.click(
            fn=search_products_enhanced,
            inputs=[descripcion_input, search_method, max_images, num_products],
            outputs=search_gallery
        )

        # Recomendaciones colaborativas (solo si está disponible)
        if collaborative_recommender:
            recommend_btn.click(
                fn=get_collaborative_recommendations_enhanced,
                inputs=[product_dropdown, collab_method],
                outputs=recommendations_gallery
            )

            refresh_products_btn.click(
                fn=lambda: gr.Dropdown(choices=get_product_options()),
                outputs=product_dropdown
            )

        # Recomendaciones basadas en cliente
        client_recommend_btn.click(
            fn=get_client_recommendations,
            inputs=[client_asins_input, client_method, client_num_recs],
            outputs=client_gallery
        )

        # Métricas
        metrics_btn.click(
            fn=get_metrics_report,
            outputs=metrics_output
        )

        def clear_metrics():
            global metrics_evaluator
            metrics_evaluator = RecommendationMetrics()
            return "Historial de métricas limpiado."

        clear_metrics_btn.click(
            fn=clear_metrics,
            outputs=metrics_output
        )

    return demo

# ==================== LANZAMIENTO ====================
if __name__ == "__main__":
    print("🚀 Iniciando sistema avanzado de recomendación...")

    # Verificar configuración
    print(f"✅ DataFrame: {len(df_similars):,} productos")
    print(f"✅ Embeddings: {description_embeddings.shape}")
    print(f"✅ Consistencia: {len(description_embeddings) == len(df_similars)}")
    print(f"✅ Búsqueda por descripción: 3 métodos disponibles")

    if collaborative_recommender:
        print(f"✅ Sistema colaborativo: 3 métodos con {len(collaborative_recommender.get_available_items()):,} productos")
    else:
        print("⚠️ Sistema colaborativo no disponible")

    print(f"✅ Sistema basado en cliente: 3 métodos disponibles")
    print(f"✅ Sistema de métricas: Inicializado")

    # Crear y lanzar interfaz
    demo = create_enhanced_interface()
    demo.launch(
        share=False,
        debug=False,
        show_error=True,
        server_name="0.0.0.0",
        server_port=7860
    )