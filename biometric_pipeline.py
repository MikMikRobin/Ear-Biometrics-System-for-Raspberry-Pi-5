#!/usr/bin/env python3
"""
Ear Biometrics System V2

Streamlined single-window ear biometrics system using:
- YOLO for ear detection and cropping
- EfficientNet-Lite4 for feature extraction
- kNN for matching
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import queue
import sqlite3
import pickle
import json
from pathlib import Path
from PIL import Image, ImageTk
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b4
import uuid
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# Import Ultimate model architecture
try:
    from ultimate_ear_training import UltimateEarFeatureExtractor
    ULTIMATE_AVAILABLE = True
except ImportError:
    ULTIMATE_AVAILABLE = False
    print("Ultimate model architecture not available")

class UltimateFeatureExtractor:
    """Feature extractor using Ultimate EfficientNet model for ears"""
    
    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        self.model = None
        self.feature_dim = 4096  # Ultimate models use 4096D features
        self.model_path = model_path
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        self.load_model()
    
    def load_model(self):
        """Load Ultimate model"""
        if not ULTIMATE_AVAILABLE:
            print("❌ Ultimate model architecture not available")
            return False
            
        try:
            if self.model_path and Path(self.model_path).exists():
                return self.load_ultimate_model()
            
            # Look for default Ultimate models
            default_paths = [
                "ultimate_ear_model_final_hard.pth",
                "ultimate_ear_model_final_medium.pth", 
                "ultimate_ear_model_final_easy.pth",
                "ultimate_ear_model_best_hard.pth",
                "ultimate_ear_model_best_medium.pth",
                "ultimate_ear_model_best_easy.pth"
            ]
            
            for path in default_paths:
                if Path(path).exists():
                    print(f"Found Ultimate model: {path}")
                    return self.load_ultimate_model(path)
            
            print("❌ No Ultimate models found")
            return False
            
        except Exception as e:
            print(f"Error loading Ultimate model: {e}")
            return False
    
    def load_ultimate_model(self, model_path=None):
        """Load Ultimate model"""
        try:
            path = model_path or self.model_path
            checkpoint = torch.load(path, map_location=self.device)
            
            # Create Ultimate model
            self.model = UltimateEarFeatureExtractor(
                feature_dim=4096,
                backbone='efficientnet_b4'  # Most Ultimate models use B4
            ).to(self.device)
            
            # Load trained weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"✅ Ultimate model loaded: {path} (epoch {epoch})")
            return True
            
        except Exception as e:
            print(f"Error loading Ultimate model: {e}")
            return False
    
    def extract_features(self, image, skip_normalization=False):
        """Extract features from ear crop using Ultimate model"""
        if self.model is None:
            print("❌ Ultimate model not loaded")
            return None
        
        try:
            # Ensure model is in evaluation mode
            self.model.eval()
            
            print(f"🔍 Ultimate model extraction:")
            print(f"  Input type: {type(image)}")
            print(f"  Input shape: {image.shape if hasattr(image, 'shape') else 'No shape'}")
            
            # Handle different input types
            if isinstance(image, Image.Image):
                # Convert PIL Image to numpy array for ToPILImage transform
                image = np.array(image)
                print(f"  Converted PIL to numpy: {image.shape}")
            
            # Ensure correct color format (RGB)
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Assume input is BGR from OpenCV, convert to RGB
                    original_mean = image.mean()
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    print(f"  BGR->RGB conversion: {original_mean:.2f} -> {image.mean():.2f}")
            
            print(f"  Pre-transform image: shape={image.shape}, dtype={image.dtype}")
            print(f"  Pre-transform range: [{image.min()}, {image.max()}]")
            
            # Apply transforms (ToPILImage expects numpy array)
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            print(f"  Post-transform tensor: shape={input_tensor.shape}, device={input_tensor.device}")
            print(f"  Post-transform range: [{input_tensor.min():.6f}, {input_tensor.max():.6f}]")
            print(f"  Post-transform mean: {input_tensor.mean():.6f}")
            
            # Extract features
            with torch.no_grad():
                print(f"  Running model inference...")
                features = self.model(input_tensor)
                print(f"  Raw model output: shape={features.shape}, device={features.device}")
                print(f"  Raw output range: [{features.min():.6f}, {features.max():.6f}]")
                print(f"  Raw output mean: {features.mean():.6f}, std: {features.std():.6f}")
            
            final_features = features.cpu().numpy().flatten()
            print(f"  Final features: shape={final_features.shape}")
            print(f"  Final range: [{final_features.min():.6f}, {final_features.max():.6f}]")
            
            # Ultimate models may not have built-in normalization, so we normalize here
            # But only if not skipping (for consistency with EfficientNet models)
            if not skip_normalization:
                norm = np.linalg.norm(final_features)
                final_features = final_features / (norm + 1e-8)
                print(f"  Normalized features: norm={norm:.6f}")
            else:
                print(f"  Features (no normalization): norm={np.linalg.norm(final_features):.6f}")
            
            return final_features
            
        except Exception as e:
            print(f"Ultimate feature extraction error: {e}")
            import traceback
            traceback.print_exc()
            return None

class EfficientNetFeatureExtractor:
    """Feature extractor using trained EfficientNet for ears"""
    
    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        self.model = None
        self.feature_dim = 512  # Default, will be overridden by checkpoint
        self.model_path = model_path
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        self.load_model()
        
    def load_model(self):
        """Load trained EfficientNet model or pre-trained if no trained model available"""
        try:
            # Try to load trained model first
            if self.model_path and Path(self.model_path).exists():
                return self.load_trained_model()
            
            # Look for default trained model (prioritize final model)
            default_paths = [
                "ear_efficientnet_epoch_50.pth",
                "ear_efficientnet_epoch_40.pth",
                "ear_efficientnet_epoch_30.pth",
                "ear_efficientnet_epoch_20.pth",
                "ear_efficientnet_epoch_10.pth"
            ]
            
            for path in default_paths:
                if Path(path).exists():
                    print(f"Found trained model: {path}")
                    return self.load_trained_model(path)
            
            # Fall back to pre-trained ImageNet model
            print("No trained ear model found, using pre-trained ImageNet EfficientNet-B4")
            return self.load_pretrained_model()
            
        except Exception as e:
            print(f"Error loading EfficientNet: {e}")
            return False
    
    def load_trained_model(self, model_path=None):
        """Load trained ear-specific EfficientNet model"""
        try:
            path = model_path or self.model_path
            checkpoint = torch.load(path, map_location=self.device)
            
            # Get feature dimension from checkpoint
            self.feature_dim = checkpoint.get('feature_dim', 512)
            
            # Detect if this is an "Excellent" model trained with timm (attention + feature_processor)
            state_dict_keys = list(checkpoint['model_state_dict'].keys())
            is_excellent_model = (
                any('attentions' in k for k in state_dict_keys) or
                any('feature_processor' in k for k in state_dict_keys) or
                any('backbone.blocks' in k for k in state_dict_keys)
            )

            if is_excellent_model:
                # Define Excellent architecture inline to avoid external deps
                try:
                    import timm  # timm backbone used during training
                except ImportError:
                    raise ImportError("timm is required to load Excellent model checkpoints. Please install timm.")

                class ChannelAttention(nn.Module):
                    def __init__(self, in_planes, ratio=16):
                        super().__init__()
                        self.avg_pool = nn.AdaptiveAvgPool2d(1)
                        self.max_pool = nn.AdaptiveMaxPool2d(1)
                        self.fc = nn.Sequential(
                            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                            nn.ReLU(),
                            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
                        )
                        self.sigmoid = nn.Sigmoid()
                    def forward(self, x):
                        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

                class SpatialAttention(nn.Module):
                    def __init__(self, kernel_size=7):
                        super().__init__()
                        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
                        self.sigmoid = nn.Sigmoid()
                    def forward(self, x):
                        avg_out = torch.mean(x, dim=1, keepdim=True)
                        max_out, _ = torch.max(x, dim=1, keepdim=True)
                        x = torch.cat([avg_out, max_out], dim=1)
                        return self.sigmoid(self.conv1(x))

                class CBAM(nn.Module):
                    def __init__(self, in_planes, ratio=16, kernel_size=7):
                        super().__init__()
                        self.ca = ChannelAttention(in_planes, ratio)
                        self.sa = SpatialAttention(kernel_size)
                    def forward(self, x):
                        x = x * self.ca(x)
                        x = x * self.sa(x)
                        return x

                class ExcellentEarFeatureExtractor(nn.Module):
                    def __init__(self, feature_dim=2048, dropout_rate=0.2):
                        super().__init__()
                        # Use EfficientNet-B4 features (timm) with last feature map
                        self.backbone = timm.create_model('efficientnet_b4', pretrained=True, features_only=True, out_indices=[4])
                        feature_info = self.backbone.feature_info
                        num_chs = [info['num_chs'] for info in feature_info][4]  # 448
                        self.attentions = nn.ModuleList([CBAM(num_chs)])
                        self.feature_processor = nn.Sequential(
                            nn.Conv2d(num_chs, feature_dim // 2, 1),
                            nn.BatchNorm2d(feature_dim // 2),
                            nn.ReLU(inplace=True),
                            nn.AdaptiveAvgPool2d(1),
                            nn.Flatten(),
                            nn.Dropout(dropout_rate),
                            nn.Linear(feature_dim // 2, feature_dim),
                            nn.BatchNorm1d(feature_dim),
                            nn.ReLU(inplace=True),
                            nn.Dropout(dropout_rate / 2),
                            nn.Linear(feature_dim, feature_dim),
                            nn.BatchNorm1d(feature_dim),
                        )
                        self.feature_dim = feature_dim
                        # Init
                        for m in self.modules():
                            if isinstance(m, nn.Linear):
                                nn.init.xavier_uniform_(m.weight)
                                if m.bias is not None:
                                    nn.init.constant_(m.bias, 0)
                            elif isinstance(m, nn.Conv2d):
                                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                                nn.init.constant_(m.weight, 1)
                                nn.init.constant_(m.bias, 0)
                    def forward(self, x):
                        feats = self.backbone(x)
                        attended = self.attentions[0](feats[0])
                        out = self.feature_processor(attended)
                        return torch.nn.functional.normalize(out, p=2, dim=1)

                # Create Excellent model
                # Override feature_dim from checkpoint if present (commonly 2048)
                self.feature_dim = checkpoint.get('feature_dim', self.feature_dim)
                self.model = ExcellentEarFeatureExtractor(feature_dim=self.feature_dim)
            else:
                # Create the EXACT same model architecture as in simple EfficientNet training
                class EarFeatureExtractor(nn.Module):
                    def __init__(self, feature_dim=512, pretrained=True):
                        super(EarFeatureExtractor, self).__init__()
                        # torchvision EfficientNet-B4
                        if pretrained:
                            from torchvision.models import EfficientNet_B4_Weights
                            self.backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
                        else:
                            self.backbone = efficientnet_b4(weights=None)
                        # Classifier head replacement
                        in_features = getattr(self.backbone.classifier, 'in_features', 1792)
                        self.backbone.classifier = nn.Sequential(
                            nn.Dropout(0.2),
                            nn.Linear(in_features, feature_dim),
                            nn.BatchNorm1d(feature_dim),
                            nn.ReLU(),
                            nn.Linear(feature_dim, feature_dim)
                        )
                        self.feature_dim = feature_dim
                    def forward(self, x):
                        features = self.backbone(x)
                        return torch.nn.functional.normalize(features, p=2, dim=1)

                # Create model with standard architecture
                self.model = EarFeatureExtractor(feature_dim=self.feature_dim, pretrained=True)
            
            # Move to device first, then load weights
            self.model.to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            model_type = "Excellent" if is_excellent_model else "Standard"
            print(f"✓ Trained {model_type} ear model loaded: {path}")
            print(f"  Feature dimension: {self.feature_dim}")
            return True
            
        except Exception as e:
            print(f"Error loading trained model: {e}")
            return False
    
    def load_pretrained_model(self):
        """Load pre-trained EfficientNet-B4 model (fallback)"""
        try:
            # Load pre-trained EfficientNet-B4
            from torchvision.models import EfficientNet_B4_Weights
            self.model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
            
            # Remove classifier to get features
            self.model.classifier = nn.Identity()
            self.feature_dim = 1792  # EfficientNet-B4 feature dimension
            
            # Set to evaluation mode
            self.model.eval()
            self.model.to(self.device)
            
            print("✓ Pre-trained EfficientNet-B4 loaded (ImageNet weights)")
            return True
            
        except Exception as e:
            print(f"Error loading pre-trained model: {e}")
            return False
    
    def extract_features(self, image, skip_normalization=False):
        """
        Extract features from ear crop
        
        Args:
            image: numpy array (BGR format from OpenCV)
            skip_normalization: If True, skip L2 normalization (for database features)
            
        Returns:
            numpy array: feature vector (512 for trained model, 1792 for pretrained)
        """
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Preprocess
            input_tensor = self.transform(image_rgb).unsqueeze(0)
            
            # Ensure model and input are on the same device
            if self.model is not None:
                # Ensure model is in evaluation mode
                self.model.eval()
                
                # Move input to same device as model
                model_device = next(self.model.parameters()).device
                input_tensor = input_tensor.to(model_device)
                
                # Extract features
                with torch.no_grad():
                    features = self.model(input_tensor)
                    features = features.cpu().numpy().flatten()
            else:
                # Fallback if model not loaded
                features = np.random.rand(self.feature_dim).astype(np.float32)
            
            # The model's forward() already does L2 normalization
            # This ensures consistent normalization between enrollment and identification
            print(f"🔍 Features from model (L2 normalized): norm={np.linalg.norm(features):.6f}")
            
            return features.astype(np.float32)
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            # Return random features as fallback
            return np.random.rand(self.feature_dim).astype(np.float32)

class BiometricDatabase:
    """Handle biometric database operations"""
    
    def __init__(self, db_path="ear_biometrics_v2.db"):
        self.db_path = db_path
        self.setup_database()
        
    def setup_database(self):
        """Initialize database with migration support"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create persons table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                num_samples INTEGER DEFAULT 0
            )
        ''')
        
        # Create features table (will be migrated if needed)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT,
                feature_vector BLOB,
                confidence REAL,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons (id)
            )
        ''')
        
        # Create model metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT NOT NULL,
                feature_dim INTEGER NOT NULL,
                model_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create current model state table to track which model is currently active
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS current_model_state (
                id INTEGER PRIMARY KEY,
                model_type TEXT NOT NULL,
                feature_dim INTEGER NOT NULL,
                model_path TEXT NOT NULL,
                model_name TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create face photos table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT UNIQUE,
                face_photo BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons (id)
            )
        ''')
        
        # Create migration tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS migration_status (
                id INTEGER PRIMARY KEY,
                migration_name TEXT UNIQUE,
                completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Migrate existing database if needed
        self._migrate_database(cursor)
        
        conn.commit()
        conn.close()
    
    def _migrate_database(self, cursor):
        """Migrate existing database to new schema"""
        try:
            # Check if migration has already been completed
            cursor.execute("SELECT COUNT(*) FROM migration_status WHERE migration_name = 'schema_v2'")
            migration_completed = cursor.fetchone()[0] > 0
            
            if migration_completed:
                print("✅ Database schema is up to date (migration already completed)")
                return
            
            # Check if features table has the new columns
            cursor.execute("PRAGMA table_info(features)")
            columns = [column[1] for column in cursor.fetchall()]
            
            migration_needed = False
            
            # Add missing columns if they don't exist
            if 'feature_dim' not in columns:
                print("🔄 Migrating database: Adding feature_dim column...")
                cursor.execute("ALTER TABLE features ADD COLUMN feature_dim INTEGER")
                migration_needed = True
            
            if 'model_type' not in columns:
                print("🔄 Migrating database: Adding model_type column...")
                cursor.execute("ALTER TABLE features ADD COLUMN model_type TEXT")
                migration_needed = True
            
            # Check if current_model_state table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='current_model_state'")
            if not cursor.fetchone():
                print("🔄 Migrating database: Creating current_model_state table...")
                cursor.execute('''
                    CREATE TABLE current_model_state (
                        id INTEGER PRIMARY KEY,
                        model_type TEXT NOT NULL,
                        feature_dim INTEGER NOT NULL,
                        model_path TEXT NOT NULL,
                        model_name TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                migration_needed = True
            else:
                # Check if model_path column exists
                cursor.execute("PRAGMA table_info(current_model_state)")
                columns = [column[1] for column in cursor.fetchall()]
                if 'model_path' not in columns:
                    print("🔄 Migrating database: Adding model_path column to current_model_state...")
                    cursor.execute("ALTER TABLE current_model_state ADD COLUMN model_path TEXT")
                    migration_needed = True
            
            if migration_needed:
                # Mark migration as completed
                cursor.execute("INSERT OR IGNORE INTO migration_status (migration_name) VALUES ('schema_v2')")
                print("✅ Database migration completed")
            else:
                # Mark migration as completed even if no changes were needed
                cursor.execute("INSERT OR IGNORE INTO migration_status (migration_name) VALUES ('schema_v2')")
                print("✅ Database schema is up to date")
            
        except Exception as e:
            print(f"⚠️ Database migration error: {e}")
            # If migration fails, we'll continue with the old schema
    
    def add_person(self, person_id, name, feature_vectors, confidences=None, model_type=None, feature_dim=None, image_paths=None):
        """Add person with feature vectors and model metadata"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert person
            cursor.execute(
                "INSERT OR REPLACE INTO persons (id, name, num_samples) VALUES (?, ?, ?)",
                (person_id, name, len(feature_vectors))
            )
            
            # Insert features with model metadata (backward compatible)
            for i, features in enumerate(feature_vectors):
                confidence = confidences[i] if confidences else 0.8
                feature_blob = pickle.dumps(features)
                image_path = image_paths[i] if image_paths and i < len(image_paths) else None
                
                # Check if new columns exist
                cursor.execute("PRAGMA table_info(features)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'feature_dim' in columns and 'model_type' in columns:
                    if 'image_path' in columns:
                        # Full new schema - include all metadata
                        cursor.execute(
                            "INSERT INTO features (person_id, feature_vector, confidence, feature_dim, model_type, image_path) VALUES (?, ?, ?, ?, ?, ?)",
                            (person_id, feature_blob, confidence, feature_dim, model_type, image_path)
                        )
                    else:
                        # Partial new schema - exclude image_path
                        cursor.execute(
                            "INSERT INTO features (person_id, feature_vector, confidence, feature_dim, model_type) VALUES (?, ?, ?, ?, ?)",
                            (person_id, feature_blob, confidence, feature_dim, model_type)
                        )
                else:
                    # Old schema - exclude model metadata
                    cursor.execute(
                        "INSERT INTO features (person_id, feature_vector, confidence) VALUES (?, ?, ?)",
                        (person_id, feature_blob, confidence)
                    )
            
            # Store model metadata (only if it doesn't exist)
            if model_type and feature_dim:
                # Check if this model metadata already exists
                cursor.execute(
                    "SELECT COUNT(*) FROM model_metadata WHERE model_type = ? AND feature_dim = ?",
                    (model_type, feature_dim)
                )
                count = cursor.fetchone()[0]
                
                if count == 0:
                    cursor.execute(
                        "INSERT INTO model_metadata (model_type, feature_dim, model_name) VALUES (?, ?, ?)",
                        (model_type, feature_dim, f"{model_type}_{feature_dim}d")
                    )
                    print(f"📊 Added new model metadata: {model_type}_{feature_dim}d")
                else:
                    print(f"📊 Model metadata already exists: {model_type}_{feature_dim}d")
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Database error: {e}")
            return False
    
    def get_all_features(self, model_type=None, feature_dim=None, current_model_dim=None):
        """Get all features for matching, optionally filtered by model type and dimension"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if new columns exist
            cursor.execute("PRAGMA table_info(features)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'feature_dim' in columns and 'model_type' in columns:
                # New schema - can filter by model
                if model_type and feature_dim:
                    # First try to get features with matching model type and dimension
                    cursor.execute(
                        "SELECT person_id, feature_vector FROM features WHERE model_type = ? AND feature_dim = ?",
                        (model_type, feature_dim)
                    )
                    results = cursor.fetchall()
                    
                    # If no exact match, try to find compatible features
                    if not results:
                        print(f"📊 No exact match for {model_type} ({feature_dim}D), searching for compatible features...")
                        
                        # Try same model_type with different feature_dim
                        cursor.execute(
                            "SELECT person_id, feature_vector, feature_dim FROM features WHERE model_type = ?",
                            (model_type,)
                        )
                        same_type_results = cursor.fetchall()
                        
                        if same_type_results:
                            print(f"📊 Found {len(same_type_results)} features with same model_type but different dims")
                            # Check if dimensions are consistent
                            dims = [pickle.loads(row[1]).shape[0] if len(pickle.loads(row[1]).shape) > 0 else len(pickle.loads(row[1])) for row in same_type_results]
                            if len(set(dims)) == 1:
                                print(f"📊 All features have consistent dimension: {dims[0]}D")
                                results = [(row[0], row[1]) for row in same_type_results]
                            else:
                                print(f"⚠️ Inconsistent dimensions found: {set(dims)}")
                                print("⚠️ Cannot use mixed-dimension features")
                        
                        # If still no results, try legacy features
                        if not results:
                            print(f"📊 Trying legacy features...")
                            cursor.execute(
                                "SELECT person_id, feature_vector FROM features WHERE model_type IS NULL OR model_type = ''"
                            )
                            results = cursor.fetchall()
                            print(f"📊 Legacy query returned {len(results)} results")
                        
                        # If STILL no results, try ANY features (last resort)
                        if not results:
                            print(f"📊 No compatible features found, trying ANY available features...")
                            cursor.execute("SELECT person_id, feature_vector FROM features")
                            all_results = cursor.fetchall()
                            print(f"📊 Found {len(all_results)} total features in database")
                            
                            if all_results:
                                # Check if all features have consistent dimensions
                                dims = []
                                for row in all_results:
                                    try:
                                        feature = pickle.loads(row[1])
                                        dim = feature.shape[0] if len(feature.shape) > 0 else len(feature)
                                        dims.append(dim)
                                    except:
                                        continue
                                
                                if dims and len(set(dims)) == 1:
                                    print(f"📊 All features have consistent dimension: {dims[0]}D")
                                    results = all_results
                                else:
                                    print(f"⚠️ Mixed dimensions found: {set(dims)} - cannot use")
                else:
                    # Get all features
                    cursor.execute("SELECT person_id, feature_vector FROM features")
                    results = cursor.fetchall()
            else:
                # Old schema - get all features (no filtering possible)
                cursor.execute("SELECT person_id, feature_vector FROM features")
                results = cursor.fetchall()
            
            if not results:
                conn.close()
                return [], []
            
            # Group features by person_id
            person_features = {}
            
            for person_id, feature_blob in results:
                feature_vector = pickle.loads(feature_blob)
                if person_id not in person_features:
                    person_features[person_id] = []
                person_features[person_id].append(feature_vector)
            
            # Use individual features instead of averaging (averaging corrupts features)
            person_ids = []
            features = []
            
            for person_id, feature_list in person_features.items():
                # Use individual features instead of averaging
                for i, feature in enumerate(feature_list):
                    person_ids.append(person_id)
                    features.append(feature)
                
                # Debug: Show individual feature fingerprints
                import hashlib
                individual_fingerprints = []
                for i, feature in enumerate(feature_list):
                    fingerprint = hashlib.md5(feature[:10].tobytes()).hexdigest()[:8]
                    individual_fingerprints.append(fingerprint)
                
                print(f"📊 Person {person_id}: using {len(feature_list)} individual features")
                print(f"📊 Individual fingerprints: {individual_fingerprints}")
                print(f"📊 Features already normalized by model (no double normalization)")
            
            conn.close()
            
            # Check if all features have the same dimension
            if features:
                feature_dims = [len(f) for f in features]
                unique_dims = set(feature_dims)
                
                if len(unique_dims) > 1:
                    print(f"⚠️ Mixed feature dimensions found: {unique_dims}")
                    
                    # Check if current model can handle any of these dimensions
                    current_dim = current_model_dim
                    
                    if current_dim and current_dim in unique_dims:
                        # Filter to only features with the current model's expected dimension
                        print(f"📊 Filtering to features with current model dimension: {current_dim}D")
                        filtered_person_ids = []
                        filtered_features = []
                        
                        for i, (person_id, feature) in enumerate(zip(person_ids, features)):
                            if len(feature) == current_dim:
                                filtered_person_ids.append(person_id)
                                filtered_features.append(feature)
                        
                        if filtered_features:
                            print(f"📊 Using {len(filtered_features)} features with dimension {current_dim}D")
                            person_ids = filtered_person_ids
                            features = filtered_features
                        else:
                            print(f"⚠️ No features found with current model dimension {current_dim}D")
                            return [], []
                    else:
                        print(f"⚠️ Current model dimension {current_dim}D not compatible with available dimensions {unique_dims}")
                        print("⚠️ Please load a model that matches the database features or clear the database")
                    return [], []
                
                # Convert to numpy array only if dimensions are consistent
                try:
                    features_array = np.array(features)
                    return person_ids, features_array
                except ValueError as ve:
                    print(f"⚠️ Cannot create feature array: {ve}")
                    return [], []
            
            return person_ids, features
            
        except Exception as e:
            print(f"Database query error: {e}")
            return [], []
    
    def get_persons(self):
        """Get all enrolled persons"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, name, num_samples, created_at FROM persons ORDER BY created_at DESC")
            results = cursor.fetchall()
            
            conn.close()
            return results
            
        except Exception as e:
            print(f"Database query error: {e}")
            return []
    
    def get_person_name(self, person_id):
        """Get person name by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM persons WHERE id = ?", (person_id,))
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else "Unknown"
        except Exception as e:
            print(f"❌ Error getting person name: {e}")
            return "Unknown"
    
    def get_person_samples(self, person_id):
        """Get all samples for a specific person"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT feature_vector, confidence, image_path, created_at, feature_dim, model_type 
                FROM features 
                WHERE person_id = ? 
                ORDER BY created_at DESC
            """, (person_id,))
            
            results = cursor.fetchall()
            conn.close()
            
            samples = []
            for feature_blob, confidence, image_path, created_at, feature_dim, model_type in results:
                try:
                    feature_vector = pickle.loads(feature_blob)
                    samples.append({
                        'features': feature_vector,
                        'confidence': confidence,
                        'image_path': image_path,
                        'created_at': created_at,
                        'feature_dim': feature_dim,
                        'model_type': model_type
                    })
                except Exception as e:
                    print(f"Error loading sample: {e}")
                    continue
            
            return samples
            
        except Exception as e:
            print(f"Error getting person samples: {e}")
            return []
    
    def get_person_face_photo(self, person_id):
        """Get person face photo by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT face_photo FROM face_photos WHERE person_id = ?", (person_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0]:
                # Deserialize face photo
                face_photo = pickle.loads(result[0])
                return face_photo
            return None
        except Exception as e:
            print(f"❌ Error getting face photo: {e}")
            return None
    
    def delete_person(self, person_id):
        """Delete a person and all their features from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete features first (foreign key constraint)
            cursor.execute("DELETE FROM features WHERE person_id = ?", (person_id,))
            
            # Delete person
            cursor.execute("DELETE FROM persons WHERE id = ?", (person_id,))
            
            conn.commit()
            conn.close()
            
            print(f"✓ Deleted person {person_id} from database")
            return True
            
        except Exception as e:
            print(f"Error deleting person: {e}")
            return False
    
    def clear_all_persons(self):
        """Clear all persons and features from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete all features first
            cursor.execute("DELETE FROM features")
            
            # Delete all persons
            cursor.execute("DELETE FROM persons")
            
            conn.commit()
            conn.close()
            
            print("✓ Cleared all persons from database")
            return True
            
        except Exception as e:
            print(f"Error clearing database: {e}")
            return False
    
    def get_database_status(self):
        """Get database status for debugging"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get person count
            cursor.execute("SELECT COUNT(*) FROM persons")
            person_count = cursor.fetchone()[0]
            
            # Check if new columns exist
            cursor.execute("PRAGMA table_info(features)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'feature_dim' in columns and 'model_type' in columns:
                # New schema - get feature count by model type
                cursor.execute("SELECT model_type, feature_dim, COUNT(*) FROM features GROUP BY model_type, feature_dim")
                feature_stats = cursor.fetchall()
                
                # Get model metadata
                cursor.execute("SELECT model_type, feature_dim, model_name FROM model_metadata")
                model_metadata = cursor.fetchall()
            else:
                # Old schema - no model information available
                cursor.execute("SELECT COUNT(*) FROM features")
                total_features = cursor.fetchone()[0]
                feature_stats = [("Unknown", "Unknown", total_features)] if total_features > 0 else []
                model_metadata = []
            
            conn.close()
            
            return {
                'person_count': person_count,
                'feature_stats': feature_stats,
                'model_metadata': model_metadata
            }
            
        except Exception as e:
            print(f"Database status error: {e}")
            return None
    
    def cleanup_duplicate_metadata(self):
        """Clean up duplicate model metadata entries"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Remove duplicates, keeping only the first occurrence
            cursor.execute("""
                DELETE FROM model_metadata 
                WHERE rowid NOT IN (
                    SELECT MIN(rowid) 
                    FROM model_metadata 
                    GROUP BY model_type, feature_dim
                )
            """)
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                print(f"🧹 Cleaned up {deleted_count} duplicate model metadata entries")
            else:
                print("🧹 No duplicate model metadata found")
                
            return deleted_count
            
        except Exception as e:
            print(f"Cleanup error: {e}")
            return 0
    
    def save_current_model_state(self, model_type, feature_dim, model_path, model_name):
        """Save the current model state to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clear existing state and insert new one
            cursor.execute("DELETE FROM current_model_state")
            cursor.execute(
                "INSERT INTO current_model_state (model_type, feature_dim, model_path, model_name) VALUES (?, ?, ?, ?)",
                (model_type, feature_dim, model_path, model_name)
            )
            
            conn.commit()
            conn.close()
            
            print(f"💾 Saved current model state: {model_type} ({feature_dim}D) - {model_path}")
            print(f"💾 DEBUG: Exact model path saved to database")
            return True
            
        except Exception as e:
            print(f"Error saving model state: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_current_model_state(self):
        """Get the current model state from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT model_type, feature_dim, model_path, model_name FROM current_model_state LIMIT 1")
            result = cursor.fetchone()
            
            conn.close()
            
            if result:
                model_type, feature_dim, model_path, model_name = result
                print(f"📖 Loaded model state: {model_type} ({feature_dim}D) - {model_path}")
                print(f"📖 DEBUG: Exact model path loaded from database")
                return model_type, feature_dim, model_path, model_name
            else:
                print("📖 No saved model state found")
                print("📖 DEBUG: No model state in database")
                return None, None, None, None
            
        except Exception as e:
            print(f"Error loading model state: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None

class EarBiometricsV2:
    def __init__(self, root):
        self.root = root
        self.root.title("Ear Biometrics System V2 - YOLO + EfficientNet")
        self.root.geometry("1200x800")
        
        # Models
        self.yolo_model = None
        self.feature_extractor = None
        self.database = BiometricDatabase()
        
        # Camera
        self.camera = None
        self.is_running = False
        self.current_camera_index = 0
        self.available_cameras = []
        
        # Processing
        self.frame_queue = queue.Queue(maxsize=2)
        self.mode = "identify"  # "enroll" or "identify"
        self.enrollment_frames = []
        self.enrollment_features = []
        
        # Quality control settings
        self.min_ear_size = 80  # Minimum width/height for ear bounding box
        self.target_ear_size = 150  # Ideal ear size for best quality
        self.show_guidelines = True  # Show visual guidelines
        
        # Matching
        self.knn_model = None
        self.person_ids = []
        self.feature_database = None
        
        # Model paths tracking
        self.model_paths = []
        
        # Setup
        self.setup_gui()
        self.find_cameras()
        self.update_model_selector()  # Initialize feature model selector
        self.update_yolo_selector()   # Initialize YOLO model selector
        
        # Auto-load default model on startup
        self.root.after(1000, self.auto_load_default_model)  # Delay to ensure GUI is ready
    
    def auto_load_default_model(self):
        """Automatically load the correct model on startup based on saved state"""
        try:
            print("🔄 Auto-loading model on startup...")
            print(f"🔄 DEBUG: Starting auto-load process")
            
            # Check if there's a saved model state
            saved_model_type, saved_feature_dim, saved_model_path, saved_model_name = self.database.get_current_model_state()
            
            if saved_model_type and saved_feature_dim and saved_model_path:
                print(f"🔄 Found saved model state: {saved_model_type} ({saved_feature_dim}D) - {saved_model_path}")
                print(f"🔄 DEBUG: Setting GUI to match saved model type")
                
                # Set the GUI to match the saved model type
                self.model_type_var.set(saved_model_type)
                self.update_model_selector()
                
                # Try to find the saved model by exact path match first
                model_names = self.model_selector['values']
                print(f"🔄 DEBUG: Available models: {list(model_names)}")
                print(f"🔄 DEBUG: Available paths: {self.model_paths}")
                print(f"🔄 DEBUG: Looking for saved path: {saved_model_path}")
                
                # First try to find exact path match
                exact_match_found = False
                for i, model_path in enumerate(self.model_paths):
                    if model_path == saved_model_path:
                        self.model_selector.current(i)
                        print(f"🔄 Selected saved model by exact path: {saved_model_path}")
                        print(f"🔄 DEBUG: Model index set to {i}")
                        exact_match_found = True
                        break
                
                # If no exact path match, try to find by name
                if not exact_match_found:
                    print(f"⚠️ Saved model path {saved_model_path} not found in current list")
                    for i, model_name in enumerate(model_names):
                        if model_name == saved_model_name:
                            self.model_selector.current(i)
                            print(f"🔄 Selected saved model by exact name: {saved_model_name}")
                            print(f"🔄 DEBUG: Model index set to {i}")
                            exact_match_found = True
                            break
                
                if not exact_match_found:
                    print(f"⚠️ Saved model {saved_model_name} not found in current list")
                    if model_names:
                        self.model_selector.current(0)
                        print(f"🔄 Using first available model: {model_names[0]}")
            else:
                print("🔄 No saved model state found, using default")
                print(f"🔄 Current model type: {self.model_type_var.get()}")
            
            # Load the models
            print(f"🔄 DEBUG: About to call load_models()")
            success = self.load_models()
            if success:
                print("✅ Model loaded successfully on startup")
                print(f"✅ Feature extractor: {self.feature_extractor}")
                print(f"✅ Feature dimension: {self.feature_extractor.feature_dim if self.feature_extractor else 'None'}")
            else:
                print("⚠️ Failed to auto-load model - user will need to load manually")
        except Exception as e:
            print(f"⚠️ Auto-load failed: {e}")
            import traceback
            traceback.print_exc()
        
    def setup_gui(self):
        """Setup simplified single-window GUI"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Left panel - Video
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Video display
        video_frame = ttk.LabelFrame(left_panel, text="Live Video Feed", padding="10")
        video_frame.pack(fill='both', expand=True)
        
        self.video_label = ttk.Label(video_frame, text="Load models and start camera", anchor='center')
        self.video_label.pack(expand=True)
        
        # Right panel - Controls and Results
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side='right', fill='y')
        right_panel.configure(width=350)
        
        # Model selection and status
        model_frame = ttk.LabelFrame(right_panel, text="Model Selection", padding="10")
        model_frame.pack(fill='x', pady=(0, 10))
        
        # Model type selector
        ttk.Label(model_frame, text="Feature Model:").pack(anchor='w')
        self.model_type_var = tk.StringVar(value="Excellent")
        model_type_frame = ttk.Frame(model_frame)
        model_type_frame.pack(fill='x', pady=(0, 5))
        
        ttk.Radiobutton(model_type_frame, text="Excellent", variable=self.model_type_var, 
                       value="Excellent", command=self.update_model_selector).pack(side='left')
        ttk.Radiobutton(model_type_frame, text="Ultimate", variable=self.model_type_var, 
                       value="Ultimate", command=self.update_model_selector).pack(side='left')
        ttk.Radiobutton(model_type_frame, text="Simple", variable=self.model_type_var, 
                       value="Simple", command=self.update_model_selector).pack(side='left')
        
        # Feature model selector
        ttk.Label(model_frame, text="Select Feature Model:").pack(anchor='w')
        self.model_selector = ttk.Combobox(model_frame, state="readonly", width=40)
        self.model_selector.pack(fill='x', pady=(0, 10))
        
        # YOLO model selector
        ttk.Label(model_frame, text="Detection Model:").pack(anchor='w')
        self.yolo_selector = ttk.Combobox(model_frame, state="readonly", width=40)
        self.yolo_selector.pack(fill='x', pady=(0, 5))
        
        # Status labels
        self.yolo_status = ttk.Label(model_frame, text="YOLO: Not loaded", foreground='red')
        self.yolo_status.pack(anchor='w')
        
        self.efficientnet_status = ttk.Label(model_frame, text="Feature Model: Not loaded", foreground='red')
        self.efficientnet_status.pack(anchor='w')
        
        ttk.Button(model_frame, text="Load Models", command=self.load_models).pack(pady=5)
        
        # Camera controls
        camera_frame = ttk.LabelFrame(right_panel, text="Camera Controls", padding="10")
        camera_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(camera_frame, text="Camera:").pack(anchor='w')
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(camera_frame, textvariable=self.camera_var, state="readonly")
        self.camera_combo.pack(fill='x', pady=2)
        
        # Mode selection
        mode_frame = ttk.Frame(camera_frame)
        mode_frame.pack(fill='x', pady=5)
        
        self.mode_var = tk.StringVar(value="identify")
        ttk.Radiobutton(mode_frame, text="Identify", variable=self.mode_var, 
                       value="identify", command=self.on_mode_change).pack(side='left')
        ttk.Radiobutton(mode_frame, text="Enroll", variable=self.mode_var, 
                       value="enroll", command=self.on_mode_change).pack(side='left', padx=(10, 0))
        
        # Enrollment controls
        self.enroll_frame = ttk.Frame(camera_frame)
        self.enroll_frame.pack(fill='x', pady=5)
        
        ttk.Label(self.enroll_frame, text="Person Name:").pack(anchor='w')
        self.person_name_var = tk.StringVar()
        ttk.Entry(self.enroll_frame, textvariable=self.person_name_var).pack(fill='x', pady=2)
        
        # Start/Stop buttons
        button_frame = ttk.Frame(camera_frame)
        button_frame.pack(fill='x', pady=5)
        
        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_detection)
        self.start_button.pack(side='left', padx=(0, 5))
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_detection, state='disabled')
        self.stop_button.pack(side='left')
        
        # Status
        status_frame = ttk.LabelFrame(right_panel, text="Status", padding="10")
        status_frame.pack(fill='x', pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="Ready", font=('Arial', 10, 'bold'))
        self.status_label.pack(anchor='w')
        
        self.fps_label = ttk.Label(status_frame, text="FPS: 0")
        self.fps_label.pack(anchor='w')
        
        # Results
        results_frame = ttk.LabelFrame(right_panel, text="Results", padding="10")
        results_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Results text area
        self.results_text = tk.Text(results_frame, height=8, width=40, font=('Courier', 9))
        results_scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.pack(side='left', fill='both', expand=True)
        results_scrollbar.pack(side='right', fill='y')
        
        # Enrolled persons
        persons_frame = ttk.LabelFrame(right_panel, text="Enrolled Persons", padding="10")
        persons_frame.pack(fill='x')
        
        # Simple listbox for persons
        persons_list_frame = ttk.Frame(persons_frame)
        persons_list_frame.pack(fill='both', expand=True)
        
        self.persons_listbox = tk.Listbox(persons_list_frame, height=6, font=('Arial', 9))
        persons_list_scrollbar = ttk.Scrollbar(persons_list_frame, orient='vertical', command=self.persons_listbox.yview)
        self.persons_listbox.configure(yscrollcommand=persons_list_scrollbar.set)
        
        self.persons_listbox.pack(side='left', fill='both', expand=True)
        persons_list_scrollbar.pack(side='right', fill='y')
        
        # Database management buttons
        db_buttons_frame = ttk.Frame(persons_frame)
        db_buttons_frame.pack(fill='x', pady=(5, 0))
        
        ttk.Button(db_buttons_frame, text="De-enroll Selected", 
                  command=self.de_enroll_person, width=15).pack(side='left', padx=(0, 5))
        ttk.Button(db_buttons_frame, text="Clear All", 
                  command=self.clear_all_persons, width=12).pack(side='left', padx=(5, 0))
        ttk.Button(db_buttons_frame, text="Debug DB", 
                  command=self.show_database_status, width=10).pack(side='left', padx=(5, 0))
        ttk.Button(db_buttons_frame, text="Cleanup", 
                  command=self.cleanup_database, width=10).pack(side='left', padx=(5, 0))
        
        # Quality control settings
        quality_frame = ttk.LabelFrame(right_panel, text="Quality Settings", padding="10")
        quality_frame.pack(fill='x', pady=(10, 0))
        
        # Minimum ear size
        size_frame = ttk.Frame(quality_frame)
        size_frame.pack(fill='x', pady=2)
        ttk.Label(size_frame, text="Min Ear Size:").pack(side='left')
        self.min_size_var = tk.IntVar(value=self.min_ear_size)
        size_scale = ttk.Scale(size_frame, from_=50, to=200, variable=self.min_size_var, 
                              orient='horizontal', command=self.update_quality_settings)
        size_scale.pack(side='left', fill='x', expand=True, padx=(5, 5))
        self.size_label = ttk.Label(size_frame, text=f"{self.min_ear_size}px")
        self.size_label.pack(side='right')
        
        # Visual guidelines toggle
        guidelines_frame = ttk.Frame(quality_frame)
        guidelines_frame.pack(fill='x', pady=2)
        self.guidelines_var = tk.BooleanVar(value=self.show_guidelines)
        ttk.Checkbutton(guidelines_frame, text="Show Guidelines", 
                       variable=self.guidelines_var, command=self.update_quality_settings).pack(side='left')
        
        # Initially hide enrollment controls
        self.on_mode_change()
        
    def find_cameras(self):
        """Find available cameras"""
        self.available_cameras = []
        camera_names = []
        
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.available_cameras.append(i)
                    camera_names.append(f"Camera {i}")
                cap.release()
        
        self.camera_combo['values'] = camera_names
        if camera_names:
            self.camera_combo.current(0)
            self.current_camera_index = self.available_cameras[0]
    
    def update_quality_settings(self, *args):
        """Update quality control settings"""
        self.min_ear_size = self.min_size_var.get()
        self.show_guidelines = self.guidelines_var.get()
        self.size_label.config(text=f"{self.min_ear_size}px")
    
    def update_model_selector(self):
        """Update model selector based on selected model type"""
        model_type = self.model_type_var.get()
        
        if model_type == "Excellent":
            # Find Excellent models (advanced training with attention mechanisms)
            excellent_models = []
            excellent_patterns = [
                "excellent_ear_model_best.pth",
                "excellent_ear_model_*.pth"
            ]
            
            for pattern in excellent_patterns:
                excellent_models.extend(Path(".").glob(pattern))
            
            # Sort by preference (best > others)
            excellent_models.sort(key=lambda x: (
                0 if 'best' in x.name else 1,
                x.name
            ))
            
            # Store full paths, display names for combobox
            self.model_paths = [str(model) for model in excellent_models]
            model_names = [model.name for model in excellent_models]
            
        elif model_type == "Ultimate":
            # Find Ultimate models
            ultimate_models = []
            ultimate_patterns = [
                "ultimate_ear_model_final_*.pth",
                "ultimate_ear_model_best_*.pth",
                "ultimate_ear_model_checkpoint_*.pth"
            ]
            
            for pattern in ultimate_patterns:
                ultimate_models.extend(Path(".").glob(pattern))
            
            # Sort by preference (final > best > checkpoint)
            ultimate_models.sort(key=lambda x: (
                0 if 'final' in x.name else 1 if 'best' in x.name else 2,
                x.name
            ))
            
            # Store full paths, display names for combobox
            self.model_paths = [str(model) for model in ultimate_models]
            model_names = [model.name for model in ultimate_models]
            
        else:  # Simple models
            # Find Simple EfficientNet models
            simple_models = []
            simple_patterns = [
                "*efficientnet*.pth",
                "ear_*_final.pth",
                "kinear_*.pth",
                "excellent_ear_model_best.pth"
            ]
            
            for pattern in simple_patterns:
                models = Path(".").glob(pattern)
                for model in models:
                    if "ultimate" not in model.name.lower():
                        simple_models.append(model)
            
            # Sort by preference: excellent > final > best > others
            def model_priority(model_path):
                name = model_path.name.lower()
                if 'excellent' in name:
                    return 0  # Highest priority
                elif 'final' in name:
                    return 1
                elif 'best' in name:
                    return 2
                else:
                    return 3
            
            simple_models.sort(key=lambda x: (model_priority(x), x.name))
            
            # Store full paths, display names for combobox
            self.model_paths = [str(model) for model in simple_models]
            model_names = [model.name for model in simple_models]
        
        # Update combobox with display names
        self.model_selector['values'] = model_names
        if model_names:
            self.model_selector.current(0)
        else:
            self.model_selector.set("No models found")
            self.model_paths = []
    
    def update_yolo_selector(self):
        """Update YOLO model selector with available YOLO models"""
        yolo_models = []
        
        # Find YOLO models in common locations
        yolo_patterns = [
            "best.pt",
            "*.pt",
            "runs/finetune/*/weights/best.pt",
            "runs/finetune/*/weights/*.pt",
            "weights/*.pt"
        ]
        
        for pattern in yolo_patterns:
            for path in Path(".").glob(pattern):
                if path.suffix == '.pt' and path.is_file():
                    yolo_models.append(path)
        
        # Remove duplicates and sort
        yolo_models = list(set(yolo_models))
        yolo_models.sort(key=lambda x: (
            0 if x.name == 'best.pt' else 1,  # Prioritize 'best.pt'
            str(x)
        ))
        
        # Create display names with relative paths
        model_names = []
        for model in yolo_models:
            if len(str(model)) > 50:
                # Shorten very long paths
                display_name = f".../{model.parent.name}/{model.name}"
            else:
                display_name = str(model)
            model_names.append(display_name)
        
        # Update combobox
        self.yolo_selector['values'] = model_names
        if model_names:
            self.yolo_selector.current(0)
        else:
            self.yolo_selector.set("No YOLO models found")
    
    def load_models(self):
        """Load YOLO and selected feature model"""
        try:
            # Load selected YOLO model
            selected_yolo = self.yolo_selector.get()
            yolo_loaded = False
            
            if selected_yolo and selected_yolo != "No YOLO models found":
                try:
                    # Handle shortened display names
                    if selected_yolo.startswith("..."):
                        # Find the actual path
                        model_name = selected_yolo.split("/")[-1]
                        for pattern in ["best.pt", "*.pt", "runs/finetune/*/weights/*.pt"]:
                            for path in Path(".").glob(pattern):
                                if path.name == model_name:
                                    selected_yolo = str(path)
                                    break
                    
                    self.yolo_model = YOLO(selected_yolo)
                    model_name = Path(selected_yolo).name
                    self.yolo_status.config(text=f"YOLO: {model_name} ✓", foreground='green')
                    yolo_loaded = True
                    print(f"✓ YOLO model loaded: {selected_yolo}")
                    
                except Exception as e:
                    print(f"✗ YOLO model loading failed: {e}")
                    self.yolo_status.config(text="YOLO: Failed to load", foreground='red')
            
            if not yolo_loaded:
                self.yolo_status.config(text="YOLO: No model selected", foreground='red')
                messagebox.showerror("Error", "Please select a YOLO model")
            
            # Load selected feature model
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_type = self.model_type_var.get()
            selected_model_name = self.model_selector.get()
            
            if not selected_model_name or selected_model_name == "No models found":
                self.efficientnet_status.config(text="Feature Model: No model selected", foreground='red')
                messagebox.showerror("Error", "Please select a feature model")
                return False
            
            # Get the full path from the stored paths
            selected_index = self.model_selector.current()
            if selected_index >= 0 and selected_index < len(self.model_paths):
                selected_model = self.model_paths[selected_index]
                print(f"🔍 Using full model path: {selected_model}")
            else:
                # Fallback to just the name (for backward compatibility)
                selected_model = selected_model_name
                print(f"⚠️ Using model name as fallback: {selected_model}")
            
            feature_loaded = False
            
            if model_type == "Excellent":
                print(f"Loading Excellent model: {selected_model}")
                self.feature_extractor = EfficientNetFeatureExtractor(
                    model_path=selected_model, 
                    device=device
                )
                if self.feature_extractor.model is not None:
                    self.efficientnet_status.config(
                        text=f"Excellent: {selected_model[:25]}... ✓", 
                        foreground='green'
                    )
                    feature_loaded = True
                    print(f"✓ Excellent model loaded: {selected_model}")
                else:
                    self.efficientnet_status.config(text="Excellent: Failed", foreground='red')
                    print(f"✗ Excellent model loading failed: {selected_model}")
            elif model_type == "Ultimate":
                print(f"Loading Ultimate model: {selected_model}")
                self.feature_extractor = UltimateFeatureExtractor(
                    model_path=selected_model, 
                    device=device
                )
                if self.feature_extractor.model is not None:
                    self.efficientnet_status.config(
                        text=f"Ultimate: {selected_model[:25]}... ✓", 
                        foreground='green'
                    )
                    feature_loaded = True
                    print(f"✓ Ultimate model loaded: {selected_model}")
                else:
                    self.efficientnet_status.config(text="Ultimate: Failed", foreground='red')
                    print(f"✗ Ultimate model loading failed: {selected_model}")
            else:
                print(f"Loading Simple model: {selected_model}")
                self.feature_extractor = EfficientNetFeatureExtractor(
                    model_path=selected_model, 
                    device=device
                )
                if self.feature_extractor.model is not None:
                    self.efficientnet_status.config(
                        text=f"Simple: {selected_model[:25]}... ✓", 
                        foreground='green'
                    )
                    feature_loaded = True
                    print(f"✓ Simple model loaded: {selected_model}")
                else:
                    self.efficientnet_status.config(text="Simple: Failed", foreground='red')
                    print(f"✗ Simple model loading failed: {selected_model}")
            
            # Show result
            if yolo_loaded and feature_loaded:
                # Save current model state to database FIRST (before updating database)
                model_name = Path(selected_model).name if selected_model else "Unknown"
                model_path = selected_model if selected_model else "Unknown"
                self.database.save_current_model_state(model_type, self.feature_extractor.feature_dim, model_path, model_name)
                
                # Update database to check feature dimension compatibility
                print("🔄 Updating database after model load...")
                self.update_database()
                print("✅ Database update completed")
                yolo_name = Path(selected_yolo).name if selected_yolo else "Unknown"
                messagebox.showinfo("Success", 
                    f"Models loaded successfully!\n"
                    f"YOLO: {yolo_name}\n"
                    f"Feature: {model_type} - {selected_model}")
                return True
            else:
                error_msg = "Failed to load:\n"
                if not yolo_loaded:
                    error_msg += "- YOLO model\n"
                if not feature_loaded:
                    error_msg += "- Feature model\n"
                messagebox.showerror("Error", error_msg)
                return False
                
        except Exception as e:
            messagebox.showerror("Error", f"Model loading failed: {str(e)}")
            return False
    
    def on_mode_change(self):
        """Handle mode change"""
        self.mode = self.mode_var.get()
        
        if self.mode == "enroll":
            self.enroll_frame.pack(fill='x', pady=5)
        else:
            self.enroll_frame.pack_forget()
    
    def start_detection(self):
        """Start camera and detection"""
        if not self.yolo_model or not self.feature_extractor:
            messagebox.showerror("Error", "Please load models first")
            return
        
        if not self.available_cameras:
            messagebox.showerror("Error", "No cameras available")
            return
        
        try:
            # Get selected camera
            if self.camera_combo.current() >= 0:
                self.current_camera_index = self.available_cameras[self.camera_combo.current()]
            
            # Open camera
            self.camera = cv2.VideoCapture(self.current_camera_index)
            if not self.camera.isOpened():
                raise Exception(f"Cannot open camera {self.current_camera_index}")
            
            # Configure camera
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Reset state
            self.is_running = True
            self.enrollment_frames = []
            self.enrollment_features = []
            
            # Update UI
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.status_label.config(text=f"Running - {self.mode.title()} Mode")
            
            # Start processing thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
            # Start display update
            self.update_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start: {str(e)}")
    
    def stop_detection(self):
        """Stop detection"""
        self.is_running = False
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        # Update UI
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.status_label.config(text="Ready")
        self.video_label.config(image="", text="Load models and start camera")
        
        # Handle enrollment completion
        if self.mode == "enroll" and self.enrollment_features:
            self.complete_enrollment()
    
    def camera_loop(self):
        """Camera processing loop"""
        fps_counter = 0
        fps_start = time.time()
        
        while self.is_running and self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if not ret:
                break
            
            try:
                # Run YOLO detection
                results = self.yolo_model(frame, conf=0.3, verbose=False)
                
                # Process detections
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    self.process_detections(frame, results[0])
                
                # Create display frame
                display_frame = results[0].plot(conf=True, labels=True)
                
                # Add visual guidelines
                if self.show_guidelines:
                    display_frame = self.add_guidelines(display_frame)
                
                # Add to display queue
                try:
                    self.frame_queue.put_nowait(display_frame)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(display_frame)
                    except queue.Empty:
                        pass
                
                # Update FPS
                fps_counter += 1
                if time.time() - fps_start >= 1.0:
                    fps = fps_counter / (time.time() - fps_start)
                    self.fps_label.config(text=f"FPS: {fps:.1f}")
                    fps_counter = 0
                    fps_start = time.time()
                
            except Exception as e:
                print(f"Processing error: {e}")
    
    def add_guidelines(self, frame):
        """Add visual guidelines to help with ear positioning"""
        h, w = frame.shape[:2]
        
        # Center area where ear should be positioned
        center_x, center_y = w // 2, h // 2
        
        # Target ear size box (ideal size)
        target_size = self.target_ear_size
        half_target = target_size // 2
        
        # Minimum ear size box
        min_size = self.min_ear_size
        half_min = min_size // 2
        
        # Draw target area (green box)
        cv2.rectangle(frame, 
                     (center_x - half_target, center_y - half_target),
                     (center_x + half_target, center_y + half_target),
                     (0, 255, 0), 2)  # Green
        
        # Draw minimum area (orange box)
        cv2.rectangle(frame, 
                     (center_x - half_min, center_y - half_min),
                     (center_x + half_min, center_y + half_min),
                     (0, 165, 255), 1)  # Orange
        
        # Add text labels
        cv2.putText(frame, f"Target: {target_size}px", 
                   (center_x - half_target, center_y - half_target - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Min: {min_size}px", 
                   (center_x - half_min, center_y + half_min + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        # Add positioning help text
        cv2.putText(frame, "Position ear within green box for best quality", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def process_detections(self, frame, results):
        """Process ear detections with quality control"""
        # Get best detection
        best_box = results.boxes[0]
        confidence = best_box.conf.item()
        box_coords = best_box.xyxy[0].cpu().numpy()
        
        # Calculate ear size
        x1, y1, x2, y2 = map(int, box_coords)
        ear_width = x2 - x1
        ear_height = y2 - y1
        ear_size = min(ear_width, ear_height)
        
        # Quality check: minimum size requirement
        if ear_size < self.min_ear_size:
            return
        
        # Quality indicator
        if ear_size >= self.target_ear_size:
            quality_icon = "🟢"  # Excellent
        elif ear_size >= self.min_ear_size * 1.2:
            quality_icon = "🟡"  # Good
        else:
            quality_icon = "🟠"  # Acceptable
        
        # Crop ear
        ear_crop = frame[y1:y2, x1:x2]
        
        if ear_crop.size == 0:
            return
        
        if self.mode == "enroll":
            self.process_enrollment(ear_crop, confidence, ear_size, quality_icon)
        else:
            self.process_identification(ear_crop, confidence, ear_size, quality_icon)
    
    def process_enrollment(self, ear_crop, confidence, ear_size, quality_icon):
        """Process enrollment with quality feedback"""
        if len(self.enrollment_features) >= 5:  # Collect max 5 samples
            return
        
        # Extract features (enrollment - let model normalize)
        features = self.feature_extractor.extract_features(ear_crop, skip_normalization=False)
        
        if features is None:
            self.results_text.insert(tk.END, "❌ Feature extraction failed\n")
            self.results_text.see(tk.END)
            return
        
        # Create fingerprint for this enrollment sample
        import hashlib
        fingerprint = hashlib.md5(features[:10].tobytes()).hexdigest()[:8]
        print(f"📝 ENROLLMENT - Sample {len(self.enrollment_features)+1} fingerprint: {fingerprint} (first 10 values: {features[:10]})")
        
        # Store
        self.enrollment_frames.append(ear_crop.copy())
        self.enrollment_features.append(features)
        
        # Clean enrollment feedback
        sample_num = len(self.enrollment_features)
        self.results_text.insert(tk.END, f"Sample {sample_num}/5 captured\n")
        self.results_text.see(tk.END)
        
        if len(self.enrollment_features) >= 5:
            self.results_text.insert(tk.END, "✅ Ready for enrollment! Stop to save.\n")
            self.results_text.see(tk.END)
    
    def process_identification(self, ear_crop, confidence, ear_size, quality_icon):
        """Process identification with clean output"""
        if not self.knn_model or len(self.person_ids) == 0:
            self.results_text.insert(tk.END, "No enrolled persons\n")
            self.results_text.see(tk.END)
            return
        
        # Extract features (live extraction - let model normalize)
        features = self.feature_extractor.extract_features(ear_crop, skip_normalization=False)
        
        if features is None:
            self.results_text.insert(tk.END, "❌ Feature extraction failed\n")
            self.results_text.see(tk.END)
            return
        
        # Match
        try:
            print(f"🔍 DEBUG - kNN model exists: {self.knn_model is not None}")
            print(f"🔍 DEBUG - person_ids: {self.person_ids}")
            print(f"🔍 DEBUG - person_ids length: {len(self.person_ids)}")
            
            distances, indices = self.knn_model.kneighbors([features], n_neighbors=min(3, len(self.person_ids)))
            print(f"🔍 DEBUG - kNN distances: {distances[0]}")
            print(f"🔍 DEBUG - kNN indices: {indices[0]}")
            
            # Get best match
            best_distance = distances[0][0]
            best_idx = indices[0][0]
            best_person_id = self.person_ids[best_idx]
            print(f"🔍 DEBUG - best_idx: {best_idx}, best_person_id: {best_person_id}")
            
            # Convert distance to similarity (cosine distance -> similarity)
            similarity = 1 - best_distance
            
            # Debug: Show feature comparison with fingerprints
            import hashlib
            query_fingerprint = hashlib.md5(features[:10].tobytes()).hexdigest()[:8]
            print(f"🔍 DEBUG - Query features: mean={features.mean():.6f}, std={features.std():.6f}, shape={features.shape}")
            print(f"🔍 DEBUG - Query fingerprint: {query_fingerprint} (first 10 values: {features[:10]})")
            print(f"🔍 DEBUG - Current model: {self.model_type_var.get()}")
            print(f"🔍 DEBUG - Feature extractor dim: {self.feature_extractor.feature_dim if self.feature_extractor else 'None'}")
            print(f"🔍 DEBUG - Database has {len(self.feature_database)} features")
            if len(self.feature_database) > 0:
                db_feature = self.feature_database[best_idx]
                db_fingerprint = hashlib.md5(db_feature[:10].tobytes()).hexdigest()[:8]
                print(f"🔍 DEBUG - DB features: mean={db_feature.mean():.6f}, std={db_feature.std():.6f}, shape={db_feature.shape}")
                print(f"🔍 DEBUG - DB fingerprint: {db_fingerprint} (first 10 values: {db_feature[:10]})")
                print(f"🔍 DEBUG - Distance: {best_distance:.6f}, Similarity: {similarity:.6f}")
                print(f"🔍 DEBUG - Best match person: {best_person_id}")
                print(f"🔍 DEBUG - Fingerprint match: {'YES' if query_fingerprint == db_fingerprint else 'NO'}")
            else:
                print("🔍 DEBUG - No features in database!")
            
            # Use 0.825 confidence threshold as requested
            print(f"🔍 DEBUG - Threshold check: {similarity:.6f} > 0.825 = {similarity > 0.825}")
            
            if similarity > 0.75:
                # Get person name
                persons = self.database.get_persons()
                print(f"🔍 DEBUG - Found {len(persons)} persons in database")
                person_name = "Unknown"
                for p_id, name, _, _ in persons:
                    print(f"🔍 DEBUG - Checking person: {p_id} vs {best_person_id}")
                    if p_id == best_person_id:
                        person_name = name
                        print(f"🔍 DEBUG - Found matching person: {name}")
                        break
                
                result = f"✓ MATCH: {person_name} (confidence: {similarity:.3f})\n"
                print(f"🔍 DEBUG - Result: {result.strip()}")
                self.results_text.insert(tk.END, result)
            else:
                result = f"✗ No match (best: {similarity:.3f})\n"
                print(f"🔍 DEBUG - Below threshold: {result.strip()}")
                self.results_text.insert(tk.END, result)
            
            self.results_text.see(tk.END)
            
        except Exception as e:
            print(f"Identification error: {e}")
    
    def complete_enrollment(self):
        """Complete enrollment process"""
        person_name = self.person_name_var.get().strip()
        
        if not person_name:
            messagebox.showerror("Error", "Please enter person name")
            return
        
        if len(self.enrollment_features) < 3:
            messagebox.showerror("Error", "Need at least 3 samples for enrollment")
            return
        
        try:
            # Save to database with model metadata
            person_id = str(uuid.uuid4())
            model_type = self.model_type_var.get()
            feature_dim = self.feature_extractor.feature_dim if self.feature_extractor else None
            
            # Create fingerprints for all enrollment features before saving
            import hashlib
            enrollment_fingerprints = []
            for i, feature in enumerate(self.enrollment_features):
                fingerprint = hashlib.md5(feature[:10].tobytes()).hexdigest()[:8]
                enrollment_fingerprints.append(fingerprint)
                print(f"📝 SAVING - Sample {i+1} fingerprint: {fingerprint}")
            
            print(f"📝 SAVING - All enrollment fingerprints: {enrollment_fingerprints}")
            print(f"📝 SAVING - Person ID: {person_id}, Name: {person_name}")
            
            success = self.database.add_person(
                person_id, person_name, self.enrollment_features,
                model_type=model_type, feature_dim=feature_dim
            )
            
            if success:
                messagebox.showinfo("Success", f"Enrolled {person_name} with {len(self.enrollment_features)} samples")
                self.person_name_var.set("")
                self.update_database()
            else:
                messagebox.showerror("Error", "Failed to save enrollment")
                
        except Exception as e:
            messagebox.showerror("Error", f"Enrollment error: {str(e)}")
    
    def de_enroll_person(self):
        """De-enroll selected person from database"""
        selection = self.persons_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a person to de-enroll")
            return
        
        # Get selected person name
        selected_index = selection[0]
        person_info = self.persons_listbox.get(selected_index)
        person_name = person_info.split(' (')[0]  # Extract name before sample count
        
        # Confirm deletion
        result = messagebox.askyesno(
            "Confirm De-enrollment", 
            f"Are you sure you want to de-enroll '{person_name}'?\n\nThis will permanently remove all their biometric data."
        )
        
        if result:
            try:
                # Find person ID by name
                persons = self.database.get_persons()
                person_id = None
                for p_id, name, _, _ in persons:
                    if name == person_name:
                        person_id = p_id
                        break
                
                if person_id:
                    # Delete person from database
                    success = self.database.delete_person(person_id)
                    
                    if success:
                        messagebox.showinfo("Success", f"Successfully de-enrolled '{person_name}'")
                        self.update_database()  # Refresh the display and kNN model
                    else:
                        messagebox.showerror("Error", f"Failed to de-enroll '{person_name}'")
                else:
                    messagebox.showerror("Error", f"Person '{person_name}' not found in database")
                    
            except Exception as e:
                messagebox.showerror("Error", f"De-enrollment failed: {str(e)}")
    
    def clear_all_persons(self):
        """Clear all enrolled persons from database"""
        persons = self.database.get_persons()
        if not persons:
            messagebox.showinfo("Info", "Database is already empty")
            return
        
        # Confirm deletion
        result = messagebox.askyesno(
            "Confirm Clear All", 
            f"Are you sure you want to clear ALL {len(persons)} enrolled persons?\n\n⚠️ This action cannot be undone!"
        )
        
        if result:
            try:
                # Clear all persons
                success = self.database.clear_all_persons()
                
                if success:
                    messagebox.showinfo("Success", "All persons have been cleared from the database")
                    self.update_database()  # Refresh the display and kNN model
                else:
                    messagebox.showerror("Error", "Failed to clear database")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Clear operation failed: {str(e)}")
    
    def show_database_status(self):
        """Show database status for debugging"""
        try:
            status = self.database.get_database_status()
            if status:
                # Create status message
                msg = f"Database Status:\n\n"
                msg += f"Total Persons: {status['person_count']}\n\n"
                
                if status['feature_stats']:
                    msg += "Features by Model:\n"
                    for model_type, feature_dim, count in status['feature_stats']:
                        msg += f"  {model_type} ({feature_dim}D): {count} features\n"
                else:
                    msg += "No features found\n"
                
                if status['model_metadata']:
                    msg += "\nModel Metadata:\n"
                    for model_type, feature_dim, model_name in status['model_metadata']:
                        msg += f"  {model_name}: {model_type} ({feature_dim}D)\n"
                
                # Show current model info
                current_model = self.model_type_var.get()
                current_dim = self.feature_extractor.feature_dim if self.feature_extractor else "Unknown"
                msg += f"\nCurrent Model: {current_model} ({current_dim}D)"
                
                messagebox.showinfo("Database Status", msg)
            else:
                messagebox.showerror("Error", "Could not retrieve database status")
                
        except Exception as e:
            messagebox.showerror("Error", f"Database status error: {str(e)}")
    
    def cleanup_database(self):
        """Clean up duplicate model metadata entries"""
        try:
            deleted_count = self.database.cleanup_duplicate_metadata()
            if deleted_count > 0:
                messagebox.showinfo("Cleanup Complete", f"Cleaned up {deleted_count} duplicate model metadata entries")
            else:
                messagebox.showinfo("Cleanup Complete", "No duplicate model metadata found")
        except Exception as e:
            print(f"Error during cleanup: {e}")
            messagebox.showerror("Error", f"Cleanup failed: {e}")
    
    def update_database(self):
        """Update database and kNN model"""
        try:
            print("📊 Starting database update...")
            # Load features filtered by current model
            model_type = self.model_type_var.get()
            feature_dim = self.feature_extractor.feature_dim if self.feature_extractor else None
            print(f"📊 Attempting to load features for model: {model_type}, dim: {feature_dim}")
            person_ids, features = self.database.get_all_features(model_type=model_type, feature_dim=feature_dim, current_model_dim=feature_dim)
            print(f"📊 Loaded {len(person_ids)} persons from database (model: {model_type}, dim: {feature_dim})")
            
            # Debug: Show what we actually loaded
            if len(person_ids) > 0:
                print(f"📊 Loaded person IDs: {person_ids}")
                print(f"📊 Feature shapes: {[f.shape for f in features]}")
            else:
                print("📊 No features loaded with current model filter - checking for other models...")
                
                # Check what models are actually in the database
                status = self.database.get_database_status()
                if status and status['feature_stats']:
                    print("📊 Available models in database:")
                    for model_type_db, feature_dim_db, count in status['feature_stats']:
                        print(f"📊   {model_type_db} ({feature_dim_db}D): {count} features")
                
                # Try loading without model filter to see all available features
                person_ids, features = self.database.get_all_features(model_type=None, feature_dim=None)
                print(f"📊 Legacy load result: {len(person_ids)} persons")
                
                if len(person_ids) > 0:
                    print("⚠️ WARNING: Current model doesn't match database features!")
                    print("⚠️ This will cause recognition to fail.")
                    
                    # Suggest the correct model to load
                    if status and status['feature_stats']:
                        # Find the most common model in the database
                        most_common = max(status['feature_stats'], key=lambda x: x[2])  # x[2] is count
                        suggested_model_type, suggested_feature_dim, suggested_count = most_common
                        
                        print(f"⚠️ SUGGESTION: Load {suggested_model_type} model ({suggested_feature_dim}D)")
                        print(f"⚠️ This model has {suggested_count} features in the database")
                        
                        # Ask user if they want to switch to the suggested model
                        result = messagebox.askyesno(
                            "Model Mismatch Detected",
                            f"The current model ({model_type}, {feature_dim}D) doesn't match the database.\n\n"
                            f"Database contains {suggested_count} features from {suggested_model_type} model ({suggested_feature_dim}D).\n\n"
                            f"Would you like to switch to the {suggested_model_type} model to recognize existing people?"
                        )
                        
                        if result:
                            # Switch to the suggested model
                            self.model_type_var.set(suggested_model_type)
                            self.update_model_selector()
                            
                            # Try to find a matching model in the list
                            model_names = self.model_selector['values']
                            # Look for models that might match the suggested type
                            matching_models = [name for name in model_names if suggested_model_type.lower() in name.lower()]
                            if matching_models:
                                self.model_selector.set(matching_models[0])
                                print(f"🔄 Switched to suggested model: {matching_models[0]}")
                                
                                # Reload models with the suggested model
                                success = self.load_models()
                                if success:
                                    print("✅ Successfully switched to matching model")
                                    return  # Exit early since we reloaded
                                else:
                                    print("❌ Failed to load suggested model")
                            else:
                                print(f"❌ No {suggested_model_type} models found in current list")
                    else:
                        print("⚠️ Consider:")
                        print("⚠️   1. Clear database and re-enroll with current model, OR")
                        print("⚠️   2. Load the model that was used for enrollment")
            
            # Debug: Show feature statistics and fingerprints
            if len(features) > 0:
                if isinstance(features, np.ndarray):
                    print(f"📊 Feature array stats: shape={features.shape}, mean={features.mean():.6f}, std={features.std():.6f}")
                    
                    # Create feature fingerprints for tracking
                    import hashlib
                    feature_fingerprints = []
                    for i, feature in enumerate(features):
                        # Create a hash of the first 10 values for fingerprinting
                        fingerprint = hashlib.md5(feature[:10].tobytes()).hexdigest()[:8]
                        feature_fingerprints.append(fingerprint)
                        print(f"📊 Person {i+1} feature fingerprint: {fingerprint} (first 10 values: {feature[:10]})")
                    
                    print(f"📊 All feature fingerprints: {feature_fingerprints}")
                else:
                    print(f"📊 Feature list: {len(features)} features")
                    if len(features) > 0:
                        first_feature = features[0]
                        print(f"📊 First feature: shape={first_feature.shape}, mean={first_feature.mean():.6f}, std={first_feature.std():.6f}")
            
            # Check if we have any features
            has_features = False
            if isinstance(features, np.ndarray):
                has_features = features.size > 0
            elif isinstance(features, list):
                has_features = len(features) > 0
            
            print(f"📊 Has features: {has_features}")
            
            # Check if feature extractor is loaded
            if self.feature_extractor is None:
                print("⚠️ No feature extractor loaded, skipping database update")
                return
            
            if has_features:
                # Check feature dimension compatibility
                current_feature_dim = self.feature_extractor.feature_dim
                
                # Check if we have valid features array
                if isinstance(features, np.ndarray) and features.size > 0:
                    existing_feature_dim = features.shape[1] if len(features.shape) > 1 else len(features[0])
                    
                    if existing_feature_dim != current_feature_dim:
                        print(f"⚠️ Feature dimension mismatch!")
                        print(f"   Database has {existing_feature_dim}D features")
                        print(f"   Current model expects {current_feature_dim}D features")
                        
                        # Ask user to clear database
                        result = messagebox.askyesno(
                            "Feature Dimension Mismatch",
                            f"The database contains {existing_feature_dim}D features,\n"
                            f"but the current model produces {current_feature_dim}D features.\n\n"
                            f"Would you like to clear the database to use the new model?\n"
                            f"(All enrolled persons will be removed)"
                        )
                        
                        if result:
                            self.clear_all_persons()
                            return
                        else:
                            print("❌ Cannot use current model with existing database")
                            return
                elif isinstance(features, list) and len(features) > 0:
                    # Features couldn't be converted to numpy array (mixed dimensions)
                    print(f"⚠️ Database contains mixed or corrupted feature dimensions!")
                    print(f"   Current model expects {current_feature_dim}D features")
                    
                    # Ask user to clear database
                    result = messagebox.askyesno(
                        "Corrupted Database",
                        f"The database contains mixed or corrupted feature dimensions.\n"
                        f"This usually happens when switching between different model types.\n\n"
                        f"Would you like to clear the database to fix this issue?\n"
                        f"(All enrolled persons will be removed)"
                    )
                    
                    if result:
                        self.clear_all_persons()
                        return
                    else:
                        print("❌ Cannot use database with mixed feature dimensions")
                        return
                
                # If we reach here, features are valid - build kNN model
                self.person_ids = person_ids
                self.feature_database = features
                
                # Update kNN model
                n_samples = features.shape[0] if isinstance(features, np.ndarray) else len(features)
                print(f"📊 Updating kNN model with {n_samples} samples, feature dim: {current_feature_dim}")
                self.knn_model = NearestNeighbors(
                    n_neighbors=min(5, n_samples), 
                    metric='cosine'
                )
                self.knn_model.fit(features)
                print("✅ kNN model updated successfully")
            else:
                # No features available - clear kNN model
                self.knn_model = None
                self.person_ids = []
                self.feature_database = None
                print("⚠️ No features available, kNN model cleared")
            
            # Update persons list
            self.persons_listbox.delete(0, tk.END)
            persons = self.database.get_persons()
            for person_id, name, num_samples, created_at in persons:
                self.persons_listbox.insert(tk.END, f"{name} ({num_samples} samples)")
                
        except Exception as e:
            print(f"Database update error: {e}")
    
    def update_display(self):
        """Update video display"""
        if not self.is_running:
            return
        
        try:
            frame = self.frame_queue.get_nowait()
            
            # Convert and resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (640, 480))
            
            # Convert to PhotoImage
            image = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image)
            
            # Update display
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo
            
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Display error: {e}")
        
        # Schedule next update
        self.root.after(30, self.update_display)

def main():
    root = tk.Tk()
    app = EarBiometricsV2(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
