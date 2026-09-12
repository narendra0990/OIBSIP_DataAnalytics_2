import numpy as np
import pandas as pd

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X_arr, axis=0)
        self.scale_ = np.std(X_arr, axis=0)
        # Avoid division by zero
        self.scale_[self.scale_ == 0.0] = 1.0
        return self
        
    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return (X_arr - self.mean_) / self.scale_
        
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

class KMeans:
    def __init__(self, n_clusters=4, max_iter=300, random_state=42, n_init=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_init = n_init
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        
    def _init_centroids(self, X, rng):
        # K-Means++ initialization
        n_samples, n_features = X.shape
        centers = [X[rng.choice(n_samples)]]
        
        for _ in range(1, self.n_clusters):
            dist_sq = np.array([min([np.sum((x - c) ** 2) for c in centers]) for x in X])
            prob = dist_sq / dist_sq.sum()
            next_idx = rng.choice(n_samples, p=prob)
            centers.append(X[next_idx])
            
        return np.array(centers)
        
    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X_arr.shape
        rng = np.random.RandomState(self.random_state)
        
        best_inertia = np.inf
        best_centers = None
        best_labels = None
        
        for _ in range(self.n_init):
            centers = self._init_centroids(X_arr, rng)
            
            for _ in range(self.max_iter):
                # Assign to nearest centroid
                distances = np.linalg.norm(X_arr[:, np.newaxis] - centers, axis=2)
                labels = np.argmin(distances, axis=1)
                
                # Recompute centroids
                new_centers = np.array([
                    X_arr[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else centers[k]
                    for k in range(self.n_clusters)
                ])
                
                if np.allclose(centers, new_centers, atol=1e-6):
                    break
                centers = new_centers
                
            # Compute inertia
            min_dist_sq = np.min(np.sum((X_arr[:, np.newaxis] - centers) ** 2, axis=2), axis=1)
            inertia = np.sum(min_dist_sq)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
                
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self
        
    def predict(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        distances = np.linalg.norm(X_arr[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)
        
    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_

def silhouette_score(X, labels):
    X_arr = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    n_samples = len(X_arr)
    
    if len(unique_labels) < 2 or len(unique_labels) >= n_samples:
        return 0.0
        
    # Sample if dataset is large to maintain fast computation
    if n_samples > 1000:
        idx = np.random.RandomState(42).choice(n_samples, size=1000, replace=False)
        X_arr = X_arr[idx]
        labels = labels[idx]
        n_samples = len(X_arr)
        
    s_vals = []
    for i in range(n_samples):
        same_cluster_mask = (labels == labels[i])
        diff_cluster_masks = [labels == k for k in unique_labels if k != labels[i]]
        
        # Intra-cluster mean distance a(i)
        if np.sum(same_cluster_mask) > 1:
            a_i = np.mean(np.linalg.norm(X_arr[same_cluster_mask] - X_arr[i], axis=1))
        else:
            a_i = 0.0
            
        # Inter-cluster mean minimum distance b(i)
        b_i = min([np.mean(np.linalg.norm(X_arr[m] - X_arr[i], axis=1)) for m in diff_cluster_masks])
        
        denom = max(a_i, b_i)
        s_i = (b_i - a_i) / denom if denom > 0 else 0.0
        s_vals.append(s_i)
        
    return float(np.mean(s_vals))

class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0
        
    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        
        if self.fit_intercept:
            X_b = np.hstack([np.ones((X_arr.shape[0], 1)), X_arr])
        else:
            X_b = X_arr
            
        # Normal equation with pseudo-inverse for high numerical stability
        weights = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y_arr
        
        if self.fit_intercept:
            self.intercept_ = float(weights[0])
            self.coef_ = weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = weights
            
        return self
        
    def predict(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr @ self.coef_ + self.intercept_
        
    def score(self, X, y):
        return r2_score(y, self.predict(X))

class Ridge:
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0
        
    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        n_features = X_arr.shape[1]
        
        if self.fit_intercept:
            X_b = np.hstack([np.ones((X_arr.shape[0], 1)), X_arr])
            I = np.eye(n_features + 1)
            I[0, 0] = 0 # Do not regularize intercept
        else:
            X_b = X_arr
            I = np.eye(n_features)
            
        weights = np.linalg.pinv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y_arr
        
        if self.fit_intercept:
            self.intercept_ = float(weights[0])
            self.coef_ = weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = weights
            
        return self
        
    def predict(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr @ self.coef_ + self.intercept_

class Lasso:
    def __init__(self, alpha=0.1, max_iter=1000, tol=1e-4, fit_intercept=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0
        
    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X_arr.shape
        
        if self.fit_intercept:
            mean_x = np.mean(X_arr, axis=0)
            mean_y = np.mean(y_arr)
            X_c = X_arr - mean_x
            y_c = y_arr - mean_y
        else:
            X_c = X_arr
            y_c = y_arr
            mean_x = np.zeros(n_features)
            mean_y = 0.0
            
        beta = np.zeros(n_features)
        
        for _ in range(self.max_iter):
            beta_old = beta.copy()
            for j in range(n_features):
                r_j = y_c - (X_c @ beta - X_c[:, j] * beta[j])
                rho_j = np.dot(X_c[:, j], r_j)
                z_j = np.dot(X_c[:, j], X_c[:, j])
                
                if z_j == 0:
                    continue
                    
                # Soft-thresholding
                if rho_j < -self.alpha * n_samples / 2:
                    beta[j] = (rho_j + self.alpha * n_samples / 2) / z_j
                elif rho_j > self.alpha * n_samples / 2:
                    beta[j] = (rho_j - self.alpha * n_samples / 2) / z_j
                else:
                    beta[j] = 0.0
                    
            if np.max(np.abs(beta - beta_old)) < self.tol:
                break
                
        self.coef_ = beta
        if self.fit_intercept:
            self.intercept_ = float(mean_y - np.dot(mean_x, beta))
        else:
            self.intercept_ = 0.0
            
        return self
        
    def predict(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr @ self.coef_ + self.intercept_

def train_test_split(*arrays, test_size=0.2, random_state=42, shuffle=True):
    rng = np.random.RandomState(random_state)
    n_samples = len(arrays[0])
    indices = np.arange(n_samples)
    if shuffle:
        rng.shuffle(indices)
    split_idx = int(n_samples * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    res = []
    for arr in arrays:
        if isinstance(arr, pd.DataFrame) or isinstance(arr, pd.Series):
            res.append(arr.iloc[train_idx])
            res.append(arr.iloc[test_idx])
        else:
            arr_np = np.asarray(arr)
            res.append(arr_np[train_idx])
            res.append(arr_np[test_idx])
    return res

def mean_squared_error(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

def root_mean_squared_error(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mean_absolute_error(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

def r2_score(y_true, y_pred):
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    return float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
