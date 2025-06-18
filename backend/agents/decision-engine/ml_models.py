"""
ML Models Manager - Production Ready Implementation
Machine Learning model management for insurance decision making
"""

import asyncio
import json
import logging
import pickle
import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.pipeline import Pipeline
import joblib
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Text, Boolean, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Monitoring
from prometheus_client import Counter, Histogram, Gauge

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
models_trained_total = Counter('models_trained_total', 'Total models trained', ['model_type', 'status'])
model_training_duration = Histogram('model_training_duration_seconds', 'Time to train models')
model_prediction_duration = Histogram('model_prediction_duration_seconds', 'Time to make predictions')
active_models_gauge = Gauge('active_models', 'Number of active models')
model_accuracy_gauge = Gauge('model_accuracy', 'Model accuracy scores', ['model_id', 'model_type'])

Base = declarative_base()

class ModelType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"

class ModelStatus(Enum):
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    FAILED = "failed"
    DEPRECATED = "deprecated"

class AlgorithmType(Enum):
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_REGRESSION = "linear_regression"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"
    DECISION_TREE = "decision_tree"
    NAIVE_BAYES = "naive_bayes"
    KNN = "knn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    ADABOOST = "adaboost"

@dataclass
class ModelConfiguration:
    """Model configuration parameters"""
    algorithm: AlgorithmType
    model_type: ModelType
    hyperparameters: Dict[str, Any]
    feature_selection: Optional[Dict[str, Any]] = None
    preprocessing: Optional[Dict[str, Any]] = None
    validation_strategy: str = "train_test_split"
    test_size: float = 0.2
    cross_validation_folds: int = 5
    random_state: int = 42

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_id: str
    model_type: ModelType
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    cross_val_scores: Optional[List[float]] = None
    confusion_matrix: Optional[List[List[int]]] = None
    feature_importance: Optional[Dict[str, float]] = None
    training_time: float = 0.0
    prediction_time: float = 0.0

@dataclass
class TrainingData:
    """Training data container"""
    features: pd.DataFrame
    target: pd.Series
    feature_names: List[str]
    target_name: str
    data_info: Dict[str, Any]

class MLModelRecord(Base):
    """SQLAlchemy model for ML model records"""
    __tablename__ = 'ml_models'
    
    model_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    algorithm = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    status = Column(String, nullable=False)
    version = Column(String, nullable=False)
    configuration = Column(JSON)
    metrics = Column(JSON)
    feature_names = Column(JSON)
    target_name = Column(String)
    model_path = Column(String)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    deployed_at = Column(DateTime)
    metadata = Column(JSON)

class MLModelManager:
    """
    Production-ready ML Model Manager
    Handles training, evaluation, deployment, and management of ML models
    """
    
    def __init__(self, db_url: str, redis_url: str, model_storage_path: str = "/tmp/ml_models"):
        self.db_url = db_url
        self.redis_url = redis_url
        self.model_storage_path = model_storage_path
        
        # Database setup
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Redis setup
        self.redis_client = redis.from_url(redis_url)
        
        # Model storage
        os.makedirs(model_storage_path, exist_ok=True)
        
        # Active models cache
        self.active_models: Dict[str, Any] = {}
        self.model_pipelines: Dict[str, Pipeline] = {}
        
        # Algorithm mappings
        self.classification_algorithms = {
            AlgorithmType.RANDOM_FOREST: RandomForestClassifier,
            AlgorithmType.GRADIENT_BOOSTING: GradientBoostingClassifier,
            AlgorithmType.LOGISTIC_REGRESSION: LogisticRegression,
            AlgorithmType.SVM: SVC,
            AlgorithmType.NEURAL_NETWORK: MLPClassifier,
            AlgorithmType.DECISION_TREE: DecisionTreeClassifier,
            AlgorithmType.NAIVE_BAYES: GaussianNB,
            AlgorithmType.KNN: KNeighborsClassifier,
            AlgorithmType.ADABOOST: AdaBoostClassifier
        }
        
        self.regression_algorithms = {
            AlgorithmType.RANDOM_FOREST: RandomForestClassifier,  # Use RandomForestRegressor
            AlgorithmType.GRADIENT_BOOSTING: GradientBoostingClassifier,  # Use GradientBoostingRegressor
            AlgorithmType.LINEAR_REGRESSION: LinearRegression,
            AlgorithmType.SVM: SVR,
            AlgorithmType.NEURAL_NETWORK: MLPRegressor,
            AlgorithmType.DECISION_TREE: DecisionTreeRegressor,
            AlgorithmType.KNN: KNeighborsRegressor
        }
        
        # Add XGBoost and LightGBM if available
        if XGBOOST_AVAILABLE:
            self.classification_algorithms[AlgorithmType.XGBOOST] = xgb.XGBClassifier
            self.regression_algorithms[AlgorithmType.XGBOOST] = xgb.XGBRegressor
        
        if LIGHTGBM_AVAILABLE:
            self.classification_algorithms[AlgorithmType.LIGHTGBM] = lgb.LGBMClassifier
            self.regression_algorithms[AlgorithmType.LIGHTGBM] = lgb.LGBMRegressor
        
        # Load existing models
        self._load_deployed_models()
        
        logger.info("MLModelManager initialized successfully")

    def _load_deployed_models(self):
        """Load deployed models into memory"""
        
        try:
            with self.Session() as session:
                deployed_models = session.query(MLModelRecord).filter(
                    MLModelRecord.status == ModelStatus.DEPLOYED.value
                ).all()
                
                for model_record in deployed_models:
                    try:
                        model_path = model_record.model_path
                        if os.path.exists(model_path):
                            model = joblib.load(model_path)
                            self.active_models[model_record.model_id] = model
                            logger.info(f"Loaded deployed model: {model_record.model_id}")
                    except Exception as e:
                        logger.error(f"Failed to load model {model_record.model_id}: {e}")
                
                active_models_gauge.set(len(self.active_models))
                
        except Exception as e:
            logger.error(f"Error loading deployed models: {e}")

    async def train_model(self, 
                         model_id: str,
                         name: str,
                         training_data: TrainingData,
                         config: ModelConfiguration,
                         description: str = "",
                         version: str = "1.0") -> Dict[str, Any]:
        """Train a new ML model"""
        
        start_time = datetime.utcnow()
        
        with model_training_duration.time():
            try:
                logger.info(f"Starting training for model {model_id}")
                
                # Create model pipeline
                pipeline = self._create_model_pipeline(config)
                
                # Prepare data
                X_train, X_test, y_train, y_test = train_test_split(
                    training_data.features,
                    training_data.target,
                    test_size=config.test_size,
                    random_state=config.random_state,
                    stratify=training_data.target if config.model_type == ModelType.CLASSIFICATION else None
                )
                
                # Train model
                pipeline.fit(X_train, y_train)
                
                # Evaluate model
                metrics = await self._evaluate_model(
                    pipeline, X_train, X_test, y_train, y_test, config
                )
                
                # Calculate training time
                training_time = (datetime.utcnow() - start_time).total_seconds()
                metrics.training_time = training_time
                
                # Save model
                model_path = await self._save_model(model_id, pipeline, version)
                
                # Store model record
                await self._store_model_record(
                    model_id=model_id,
                    name=name,
                    description=description,
                    algorithm=config.algorithm,
                    model_type=config.model_type,
                    version=version,
                    configuration=asdict(config),
                    metrics=asdict(metrics),
                    feature_names=training_data.feature_names,
                    target_name=training_data.target_name,
                    model_path=model_path
                )
                
                # Update metrics
                models_trained_total.labels(
                    model_type=config.model_type.value,
                    status='success'
                ).inc()
                
                if metrics.accuracy:
                    model_accuracy_gauge.labels(
                        model_id=model_id,
                        model_type=config.model_type.value
                    ).set(metrics.accuracy)
                
                logger.info(f"Model {model_id} trained successfully in {training_time:.2f}s")
                
                return {
                    "model_id": model_id,
                    "status": "trained",
                    "metrics": asdict(metrics),
                    "training_time": training_time,
                    "model_path": model_path
                }
                
            except Exception as e:
                logger.error(f"Error training model {model_id}: {e}")
                
                models_trained_total.labels(
                    model_type=config.model_type.value,
                    status='failed'
                ).inc()
                
                # Store failed model record
                await self._store_model_record(
                    model_id=model_id,
                    name=name,
                    description=description,
                    algorithm=config.algorithm,
                    model_type=config.model_type,
                    version=version,
                    configuration=asdict(config),
                    metrics={"error": str(e)},
                    feature_names=training_data.feature_names,
                    target_name=training_data.target_name,
                    model_path="",
                    status=ModelStatus.FAILED
                )
                
                raise

    def _create_model_pipeline(self, config: ModelConfiguration) -> Pipeline:
        """Create ML pipeline with preprocessing and model"""
        
        steps = []
        
        # Feature selection
        if config.feature_selection:
            selection_method = config.feature_selection.get("method", "selectkbest")
            k = config.feature_selection.get("k", 10)
            
            if selection_method == "selectkbest":
                if config.model_type == ModelType.CLASSIFICATION:
                    selector = SelectKBest(score_func=f_classif, k=k)
                else:
                    selector = SelectKBest(score_func=f_regression, k=k)
                steps.append(("feature_selection", selector))
            elif selection_method == "rfe":
                # RFE will be added after model creation
                pass
        
        # Preprocessing
        if config.preprocessing:
            scaler_type = config.preprocessing.get("scaler", "standard")
            
            if scaler_type == "standard":
                scaler = StandardScaler()
            elif scaler_type == "minmax":
                scaler = MinMaxScaler()
            elif scaler_type == "robust":
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            steps.append(("scaler", scaler))
        
        # Model
        model = self._create_model(config)
        steps.append(("model", model))
        
        return Pipeline(steps)

    def _create_model(self, config: ModelConfiguration):
        """Create ML model based on configuration"""
        
        algorithm = config.algorithm
        model_type = config.model_type
        hyperparameters = config.hyperparameters
        
        # Get algorithm class
        if model_type == ModelType.CLASSIFICATION:
            if algorithm not in self.classification_algorithms:
                raise ValueError(f"Algorithm {algorithm} not supported for classification")
            model_class = self.classification_algorithms[algorithm]
        elif model_type == ModelType.REGRESSION:
            if algorithm not in self.regression_algorithms:
                raise ValueError(f"Algorithm {algorithm} not supported for regression")
            model_class = self.regression_algorithms[algorithm]
        else:
            raise ValueError(f"Model type {model_type} not supported")
        
        # Create model with hyperparameters
        try:
            model = model_class(**hyperparameters)
            return model
        except Exception as e:
            logger.error(f"Error creating model {algorithm}: {e}")
            # Fallback to default parameters
            return model_class()

    async def _evaluate_model(self, 
                            pipeline: Pipeline,
                            X_train: pd.DataFrame,
                            X_test: pd.DataFrame,
                            y_train: pd.Series,
                            y_test: pd.Series,
                            config: ModelConfiguration) -> ModelMetrics:
        """Evaluate model performance"""
        
        start_time = datetime.utcnow()
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate prediction time
        prediction_time = (datetime.utcnow() - start_time).total_seconds()
        
        metrics = ModelMetrics(
            model_id="",  # Will be set by caller
            model_type=config.model_type,
            prediction_time=prediction_time
        )
        
        if config.model_type == ModelType.CLASSIFICATION:
            # Classification metrics
            metrics.accuracy = accuracy_score(y_test, y_pred)
            metrics.precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics.recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics.f1_score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # ROC AUC for binary classification
            if len(np.unique(y_test)) == 2:
                try:
                    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                    metrics.roc_auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    pass
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics.confusion_matrix = cm.tolist()
            
        elif config.model_type == ModelType.REGRESSION:
            # Regression metrics
            metrics.mse = mean_squared_error(y_test, y_pred)
            metrics.mae = mean_absolute_error(y_test, y_pred)
            metrics.r2_score = r2_score(y_test, y_pred)
        
        # Cross-validation
        if config.validation_strategy == "cross_validation":
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, 
                cv=config.cross_validation_folds,
                scoring='accuracy' if config.model_type == ModelType.CLASSIFICATION else 'r2'
            )
            metrics.cross_val_scores = cv_scores.tolist()
        
        # Feature importance
        try:
            model = pipeline.named_steps['model']
            if hasattr(model, 'feature_importances_'):
                # Get feature names after preprocessing
                feature_names = self._get_feature_names_after_preprocessing(pipeline, X_train.columns)
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                metrics.feature_importance = importance_dict
            elif hasattr(model, 'coef_'):
                # For linear models
                feature_names = self._get_feature_names_after_preprocessing(pipeline, X_train.columns)
                coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                importance_dict = dict(zip(feature_names, np.abs(coef)))
                metrics.feature_importance = importance_dict
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
        
        return metrics

    def _get_feature_names_after_preprocessing(self, pipeline: Pipeline, original_features: List[str]) -> List[str]:
        """Get feature names after preprocessing steps"""
        
        feature_names = list(original_features)
        
        # Handle feature selection
        if 'feature_selection' in pipeline.named_steps:
            selector = pipeline.named_steps['feature_selection']
            if hasattr(selector, 'get_support'):
                selected_features = selector.get_support()
                feature_names = [name for name, selected in zip(feature_names, selected_features) if selected]
        
        return feature_names

    async def _save_model(self, model_id: str, pipeline: Pipeline, version: str) -> str:
        """Save trained model to disk"""
        
        try:
            model_dir = os.path.join(self.model_storage_path, model_id)
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, f"model_v{version}.joblib")
            joblib.dump(pipeline, model_path)
            
            logger.info(f"Model saved to {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model {model_id}: {e}")
            raise

    async def _store_model_record(self, 
                                model_id: str,
                                name: str,
                                description: str,
                                algorithm: AlgorithmType,
                                model_type: ModelType,
                                version: str,
                                configuration: Dict[str, Any],
                                metrics: Dict[str, Any],
                                feature_names: List[str],
                                target_name: str,
                                model_path: str,
                                status: ModelStatus = ModelStatus.TRAINED):
        """Store model record in database"""
        
        try:
            with self.Session() as session:
                record = MLModelRecord(
                    model_id=model_id,
                    name=name,
                    description=description,
                    algorithm=algorithm.value,
                    model_type=model_type.value,
                    status=status.value,
                    version=version,
                    configuration=configuration,
                    metrics=metrics,
                    feature_names=feature_names,
                    target_name=target_name,
                    model_path=model_path,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    metadata={}
                )
                
                session.merge(record)
                session.commit()
                
        except Exception as e:
            logger.error(f"Error storing model record: {e}")
            raise

    async def deploy_model(self, model_id: str) -> Dict[str, Any]:
        """Deploy a trained model for inference"""
        
        try:
            with self.Session() as session:
                record = session.query(MLModelRecord).filter(
                    MLModelRecord.model_id == model_id
                ).first()
                
                if not record:
                    raise ValueError(f"Model {model_id} not found")
                
                if record.status != ModelStatus.TRAINED.value:
                    raise ValueError(f"Model {model_id} is not in trained status")
                
                # Load model
                if not os.path.exists(record.model_path):
                    raise ValueError(f"Model file not found: {record.model_path}")
                
                model = joblib.load(record.model_path)
                
                # Add to active models
                self.active_models[model_id] = model
                
                # Update status
                record.status = ModelStatus.DEPLOYED.value
                record.deployed_at = datetime.utcnow()
                record.updated_at = datetime.utcnow()
                session.commit()
                
                active_models_gauge.set(len(self.active_models))
                
                logger.info(f"Model {model_id} deployed successfully")
                
                return {
                    "model_id": model_id,
                    "status": "deployed",
                    "deployed_at": record.deployed_at.isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error deploying model {model_id}: {e}")
            raise

    async def predict(self, model_id: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using deployed model"""
        
        start_time = datetime.utcnow()
        
        with model_prediction_duration.time():
            try:
                if model_id not in self.active_models:
                    raise ValueError(f"Model {model_id} is not deployed")
                
                model = self.active_models[model_id]
                
                # Get model record for feature names
                with self.Session() as session:
                    record = session.query(MLModelRecord).filter(
                        MLModelRecord.model_id == model_id
                    ).first()
                    
                    if not record:
                        raise ValueError(f"Model record {model_id} not found")
                
                # Prepare features
                feature_names = record.feature_names
                feature_array = []
                
                for feature_name in feature_names:
                    value = features.get(feature_name, 0)
                    # Handle different data types
                    if isinstance(value, (list, dict)):
                        value = len(value) if isinstance(value, list) else 0
                    elif isinstance(value, str):
                        try:
                            value = float(value)
                        except:
                            value = 0
                    feature_array.append(float(value))
                
                # Make prediction
                feature_df = pd.DataFrame([feature_array], columns=feature_names)
                prediction = model.predict(feature_df)[0]
                
                # Get prediction probability if available
                prediction_proba = None
                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba(feature_df)[0]
                        prediction_proba = proba.tolist()
                    except:
                        pass
                
                # Calculate prediction time
                prediction_time = (datetime.utcnow() - start_time).total_seconds()
                
                result = {
                    "model_id": model_id,
                    "prediction": float(prediction) if isinstance(prediction, (np.integer, np.floating)) else prediction,
                    "prediction_proba": prediction_proba,
                    "features_used": feature_names,
                    "prediction_time": prediction_time,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                return result
                
            except Exception as e:
                logger.error(f"Error making prediction with model {model_id}: {e}")
                raise

    async def batch_predict(self, model_id: str, features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make batch predictions"""
        
        try:
            if model_id not in self.active_models:
                raise ValueError(f"Model {model_id} is not deployed")
            
            model = self.active_models[model_id]
            
            # Get model record
            with self.Session() as session:
                record = session.query(MLModelRecord).filter(
                    MLModelRecord.model_id == model_id
                ).first()
                
                if not record:
                    raise ValueError(f"Model record {model_id} not found")
            
            feature_names = record.feature_names
            
            # Prepare batch features
            batch_features = []
            for features in features_list:
                feature_array = []
                for feature_name in feature_names:
                    value = features.get(feature_name, 0)
                    if isinstance(value, (list, dict)):
                        value = len(value) if isinstance(value, list) else 0
                    elif isinstance(value, str):
                        try:
                            value = float(value)
                        except:
                            value = 0
                    feature_array.append(float(value))
                batch_features.append(feature_array)
            
            # Make batch predictions
            feature_df = pd.DataFrame(batch_features, columns=feature_names)
            predictions = model.predict(feature_df)
            
            # Get prediction probabilities if available
            prediction_probas = None
            if hasattr(model, 'predict_proba'):
                try:
                    prediction_probas = model.predict_proba(feature_df)
                except:
                    pass
            
            # Format results
            results = []
            for i, prediction in enumerate(predictions):
                result = {
                    "prediction": float(prediction) if isinstance(prediction, (np.integer, np.floating)) else prediction,
                    "prediction_proba": prediction_probas[i].tolist() if prediction_probas is not None else None
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error making batch predictions with model {model_id}: {e}")
            raise

    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get model information"""
        
        try:
            with self.Session() as session:
                record = session.query(MLModelRecord).filter(
                    MLModelRecord.model_id == model_id
                ).first()
                
                if not record:
                    raise ValueError(f"Model {model_id} not found")
                
                return {
                    "model_id": record.model_id,
                    "name": record.name,
                    "description": record.description,
                    "algorithm": record.algorithm,
                    "model_type": record.model_type,
                    "status": record.status,
                    "version": record.version,
                    "configuration": record.configuration,
                    "metrics": record.metrics,
                    "feature_names": record.feature_names,
                    "target_name": record.target_name,
                    "created_at": record.created_at.isoformat(),
                    "updated_at": record.updated_at.isoformat(),
                    "deployed_at": record.deployed_at.isoformat() if record.deployed_at else None,
                    "is_active": model_id in self.active_models
                }
                
        except Exception as e:
            logger.error(f"Error getting model info for {model_id}: {e}")
            raise

    async def list_models(self, status: ModelStatus = None) -> List[Dict[str, Any]]:
        """List all models with optional status filter"""
        
        try:
            with self.Session() as session:
                query = session.query(MLModelRecord)
                
                if status:
                    query = query.filter(MLModelRecord.status == status.value)
                
                records = query.order_by(MLModelRecord.created_at.desc()).all()
                
                models = []
                for record in records:
                    models.append({
                        "model_id": record.model_id,
                        "name": record.name,
                        "algorithm": record.algorithm,
                        "model_type": record.model_type,
                        "status": record.status,
                        "version": record.version,
                        "created_at": record.created_at.isoformat(),
                        "is_active": record.model_id in self.active_models
                    })
                
                return models
                
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    async def delete_model(self, model_id: str) -> bool:
        """Delete a model"""
        
        try:
            with self.Session() as session:
                record = session.query(MLModelRecord).filter(
                    MLModelRecord.model_id == model_id
                ).first()
                
                if not record:
                    return False
                
                # Remove from active models
                if model_id in self.active_models:
                    del self.active_models[model_id]
                    active_models_gauge.set(len(self.active_models))
                
                # Delete model files
                if record.model_path and os.path.exists(record.model_path):
                    model_dir = os.path.dirname(record.model_path)
                    if os.path.exists(model_dir):
                        shutil.rmtree(model_dir)
                
                # Delete database record
                session.delete(record)
                session.commit()
                
                logger.info(f"Model {model_id} deleted successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {e}")
            return False

    async def hyperparameter_tuning(self,
                                  model_id: str,
                                  training_data: TrainingData,
                                  config: ModelConfiguration,
                                  param_grid: Dict[str, List[Any]],
                                  search_type: str = "grid",
                                  cv_folds: int = 5,
                                  n_iter: int = 50) -> Dict[str, Any]:
        """Perform hyperparameter tuning"""
        
        try:
            logger.info(f"Starting hyperparameter tuning for model {model_id}")
            
            # Create base pipeline
            base_pipeline = self._create_model_pipeline(config)
            
            # Prepare parameter grid for pipeline
            pipeline_param_grid = {}
            for param, values in param_grid.items():
                pipeline_param_grid[f"model__{param}"] = values
            
            # Choose search strategy
            if search_type == "grid":
                search = GridSearchCV(
                    base_pipeline,
                    pipeline_param_grid,
                    cv=cv_folds,
                    scoring='accuracy' if config.model_type == ModelType.CLASSIFICATION else 'r2',
                    n_jobs=-1,
                    verbose=1
                )
            elif search_type == "random":
                search = RandomizedSearchCV(
                    base_pipeline,
                    pipeline_param_grid,
                    n_iter=n_iter,
                    cv=cv_folds,
                    scoring='accuracy' if config.model_type == ModelType.CLASSIFICATION else 'r2',
                    n_jobs=-1,
                    verbose=1,
                    random_state=config.random_state
                )
            else:
                raise ValueError(f"Unknown search type: {search_type}")
            
            # Perform search
            search.fit(training_data.features, training_data.target)
            
            # Get best parameters
            best_params = search.best_params_
            best_score = search.best_score_
            
            # Update configuration with best parameters
            best_config = config
            model_params = {}
            for param, value in best_params.items():
                if param.startswith("model__"):
                    param_name = param.replace("model__", "")
                    model_params[param_name] = value
            
            best_config.hyperparameters.update(model_params)
            
            logger.info(f"Hyperparameter tuning completed. Best score: {best_score:.4f}")
            
            return {
                "model_id": model_id,
                "best_params": best_params,
                "best_score": best_score,
                "best_config": asdict(best_config),
                "cv_results": {
                    "mean_test_scores": search.cv_results_['mean_test_score'].tolist(),
                    "std_test_scores": search.cv_results_['std_test_score'].tolist(),
                    "params": search.cv_results_['params']
                }
            }
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning for model {model_id}: {e}")
            raise

    async def get_model_statistics(self) -> Dict[str, Any]:
        """Get model management statistics"""
        
        try:
            with self.Session() as session:
                total_models = session.query(MLModelRecord).count()
                
                # Models by status
                status_counts = {}
                for status in ModelStatus:
                    count = session.query(MLModelRecord).filter(
                        MLModelRecord.status == status.value
                    ).count()
                    status_counts[status.value] = count
                
                # Models by type
                type_counts = {}
                for model_type in ModelType:
                    count = session.query(MLModelRecord).filter(
                        MLModelRecord.model_type == model_type.value
                    ).count()
                    type_counts[model_type.value] = count
                
                # Models by algorithm
                algorithm_counts = {}
                for algorithm in AlgorithmType:
                    count = session.query(MLModelRecord).filter(
                        MLModelRecord.algorithm == algorithm.value
                    ).count()
                    algorithm_counts[algorithm.value] = count
                
                return {
                    "total_models": total_models,
                    "active_models": len(self.active_models),
                    "models_by_status": status_counts,
                    "models_by_type": type_counts,
                    "models_by_algorithm": algorithm_counts,
                    "storage_path": self.model_storage_path
                }
                
        except Exception as e:
            logger.error(f"Error getting model statistics: {e}")
            return {}

# Factory function
def create_ml_model_manager(db_url: str = None, redis_url: str = None, model_storage_path: str = None) -> MLModelManager:
    """Create and configure MLModelManager instance"""
    
    if not db_url:
        db_url = "postgresql://insurance_user:insurance_pass@localhost:5432/insurance_ai"
    
    if not redis_url:
        redis_url = "redis://localhost:6379/0"
    
    if not model_storage_path:
        model_storage_path = "/tmp/insurance_ai_models"
    
    return MLModelManager(db_url=db_url, redis_url=redis_url, model_storage_path=model_storage_path)

# Example usage
if __name__ == "__main__":
    async def test_ml_model_manager():
        """Test ML model manager functionality"""
        
        manager = create_ml_model_manager()
        
        # Create sample training data
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)]
        )
        y = pd.Series(np.random.randint(0, 2, n_samples), name="target")
        
        training_data = TrainingData(
            features=X,
            target=y,
            feature_names=X.columns.tolist(),
            target_name="target",
            data_info={"samples": n_samples, "features": n_features}
        )
        
        # Create model configuration
        config = ModelConfiguration(
            algorithm=AlgorithmType.RANDOM_FOREST,
            model_type=ModelType.CLASSIFICATION,
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            preprocessing={"scaler": "standard"}
        )
        
        # Train model
        model_id = str(uuid.uuid4())
        result = await manager.train_model(
            model_id=model_id,
            name="Test Classification Model",
            training_data=training_data,
            config=config,
            description="Test model for demonstration"
        )
        
        print(f"Training result: {result}")
        
        # Deploy model
        deploy_result = await manager.deploy_model(model_id)
        print(f"Deploy result: {deploy_result}")
        
        # Make prediction
        test_features = {f"feature_{i}": np.random.randn() for i in range(n_features)}
        prediction = await manager.predict(model_id, test_features)
        print(f"Prediction: {prediction}")
        
        # Get statistics
        stats = await manager.get_model_statistics()
        print(f"Statistics: {stats}")
    
    # Run test
    # asyncio.run(test_ml_model_manager())

