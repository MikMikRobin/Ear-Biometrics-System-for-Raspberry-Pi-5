#!/usr/bin/env python3
"""
Ear Biometrics System GUI v3 - Liquid Glass Theme
Modern, sleek GUI implementation with enhanced user experience
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import queue
import sqlite3
import pickle
import json
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw, ImageFilter, ImageEnhance
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b4
import uuid
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import os

# Import the model loader for robust model handling
try:
    from model_loader import UniversalFeatureExtractor
    UNIVERSAL_LOADER_AVAILABLE = True
    print("Successfully imported UniversalFeatureExtractor")
except ImportError as e:
    UNIVERSAL_LOADER_AVAILABLE = False
    print(f"Warning: Could not import UniversalFeatureExtractor - {e}")

# Import the core biometrics classes from the biometric pipeline
try:
    from biometric_pipeline import (
        UltimateFeatureExtractor, 
        EfficientNetFeatureExtractor, 
        BiometricDatabase,
        ULTIMATE_AVAILABLE
    )
    print("Successfully imported from biometric_pipeline.py")
except ImportError as e:
    # If import fails, we'll define the classes here
    print(f"Warning: Could not import from biometric_pipeline.py - {e}")
    print("Using embedded classes instead")
    
    # Import Ultimate model architecture
    try:
        from ultimate_ear_training import UltimateEarFeatureExtractor
        ULTIMATE_AVAILABLE = True
    except ImportError:
        ULTIMATE_AVAILABLE = False
        print("Ultimate model architecture not available")

    # Import the actual Ultimate architecture from training script
    try:
        from ultimate_ear_training import UltimateEarFeatureExtractor
        ULTIMATE_AVAILABLE = True
        print("✓ Ultimate model architecture imported successfully")
    except ImportError as e:
        ULTIMATE_AVAILABLE = False
        print(f"⚠️ Ultimate model architecture not available: {e}")

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
                    "efficientnet_b4_ultimate_best.pth",
                    "efficientnet_b4_ultimate_final.pth",
                    "efficientnet_b4_ultimate_epoch_45.pth",
                    "efficientnet_b4_ultimate_epoch_42.pth",
                    "efficientnet_b4_ultimate_epoch_40.pth"
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
            """Load Ultimate model with proper architecture"""
            try:
                path = model_path or self.model_path
                # Use weights_only=False for Ultimate models that contain custom classes
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                
                # Get feature dimension from checkpoint if available
                feature_dim = checkpoint.get('feature_dim', 4096)
                
                # Determine backbone from checkpoint config if available
                backbone = 'efficientnet_b4'  # Default
                if 'config' in checkpoint and hasattr(checkpoint['config'], 'backbone'):
                    backbone = checkpoint['config'].backbone
                elif 'b5' in str(path).lower():
                    backbone = 'efficientnet_b5'
                
                print(f"Creating Ultimate model with backbone: {backbone}, feature_dim: {feature_dim}")
                
                # Create Ultimate model with correct architecture
                self.model = UltimateEarFeatureExtractor(
                    feature_dim=feature_dim,
                    backbone=backbone
                ).to(self.device)
                
                # Load trained weights
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                # Update feature dimension
                self.feature_dim = feature_dim
                
                epoch = checkpoint.get('epoch', 'unknown')
                print(f"✅ Ultimate model loaded: {path} (epoch {epoch}, {feature_dim}D)")
                return True
                
            except Exception as e:
                print(f"Error loading Ultimate model: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        def extract_features(self, image, skip_normalization=False):
            """Extract features from ear crop using Ultimate model"""
            if self.model is None:
                print("❌ Ultimate model not loaded")
                return None
            
            try:
                # Ensure model is in evaluation mode
                self.model.eval()
                
                # Handle different input types
                if isinstance(image, Image.Image):
                    image = np.array(image)
                
                # Ensure correct color format (RGB)
                if isinstance(image, np.ndarray):
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Apply transforms
                input_tensor = self.transform(image).unsqueeze(0).to(self.device)
                
                # Extract features
                with torch.no_grad():
                    features = self.model(input_tensor)
                
                final_features = features.cpu().numpy().flatten()
                
                # Ultimate models may not have built-in normalization
                if not skip_normalization:
                    norm = np.linalg.norm(final_features)
                    final_features = final_features / (norm + 1e-8)
                
                return final_features
                
            except Exception as e:
                print(f"Ultimate feature extraction error: {e}")
                import traceback
                traceback.print_exc()
                return None

    # Import timm for EfficientNet models
    try:
        import timm
        TIMM_AVAILABLE = True
    except ImportError:
        TIMM_AVAILABLE = False
        print("Warning: timm not available - some models may not load")

    # Excellent model architecture components
    class ChannelAttention(nn.Module):
        """Channel attention mechanism"""
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
            avg_out = self.fc(self.avg_pool(x))
            max_out = self.fc(self.max_pool(x))
            out = avg_out + max_out
            return self.sigmoid(out)

    class SpatialAttention(nn.Module):
        """Spatial attention mechanism"""
        def __init__(self, kernel_size=7):
            super().__init__()
            self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            x = torch.cat([avg_out, max_out], dim=1)
            x = self.conv1(x)
            return self.sigmoid(x)

    class CBAM(nn.Module):
        """Convolutional Block Attention Module"""
        def __init__(self, in_planes, ratio=16, kernel_size=7):
            super().__init__()
            self.ca = ChannelAttention(in_planes, ratio)
            self.sa = SpatialAttention(kernel_size)

        def forward(self, x):
            x = x * self.ca(x)
            x = x * self.sa(x)
            return x

    class ExcellentEarFeatureExtractor(nn.Module):
        """EXCELLENT discriminative ear feature extractor - EXACT match to training script"""
        
        def __init__(self, feature_dim=2048, dropout_rate=0.2):
            super().__init__()
            
            if not TIMM_AVAILABLE:
                raise ImportError("timm is required for ExcellentEarFeatureExtractor")
            
            # Use EfficientNet-B4 with single scale (simplify to avoid errors) - EXACT COPY
            self.backbone = timm.create_model('efficientnet_b4', pretrained=True, 
                                            features_only=True, out_indices=[4])  # Use only final features
            
            # Get feature dimensions - but only for the indices we're using - EXACT COPY
            feature_info = self.backbone.feature_info
            all_feature_dims = [info['num_chs'] for info in feature_info]
            # Since we use out_indices=[4], we only get the 5th feature map (index 4)
            self.feature_dims = [all_feature_dims[4]]  # Only the 448-channel feature
            
            # Attention module for the single scale we're using
            self.attentions = nn.ModuleList([
                CBAM(self.feature_dims[0])  # 448 channels
            ])
            
            # Single-scale feature processing - EXACT COPY
            total_features = self.feature_dims[0]  # Only one feature dimension now
            
            # Advanced feature processing pipeline - EXACT COPY
            self.feature_processor = nn.Sequential(
                # Initial compression
                nn.Conv2d(total_features, feature_dim // 2, 1),
                nn.BatchNorm2d(feature_dim // 2),
                nn.ReLU(inplace=True),
                
                # Global average pooling
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                
                # Advanced MLP head
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim // 2, feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(inplace=True),
                
                nn.Dropout(dropout_rate / 2),
                nn.Linear(feature_dim, feature_dim),
                nn.BatchNorm1d(feature_dim),
            )
            
            self.feature_dim = feature_dim
            
            # Initialize weights - EXACT COPY
            self._initialize_weights()
            
        def _initialize_weights(self):
            """Initialize weights for better convergence - EXACT COPY"""
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            
        def forward(self, x):
            # Multi-scale feature extraction - EXACT COPY
            features = self.backbone(x)
            
            # Apply attention to single feature map
            attended_feat = self.attentions[0](features[0])
            
            # No concatenation needed - single feature map
            fused_features = attended_feat
            
            # Process through advanced head
            final_features = self.feature_processor(fused_features)
            
            # L2 normalize for cosine similarity
            final_features = torch.nn.functional.normalize(final_features, p=2, dim=1)
            
            return final_features

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
                
                # Look for default trained model
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
                
                # Check if this is an Excellent model (has attention layers or timm backbone)
                state_dict_keys = list(checkpoint['model_state_dict'].keys())
                is_excellent_model = (any('attentions' in key for key in state_dict_keys) or 
                                    any('backbone.blocks' in key for key in state_dict_keys) or
                                    any('feature_processor' in key for key in state_dict_keys))
                
                if is_excellent_model:
                    print(f"Loading Excellent model architecture with {self.feature_dim}D features...")
                    # Use the ExcellentEarFeatureExtractor architecture
                    self.model = ExcellentEarFeatureExtractor(feature_dim=self.feature_dim)
                else:
                    print(f"Loading standard EfficientNet architecture with {self.feature_dim}D features...")
                    # Create the standard EfficientNet model architecture
                    class EarFeatureExtractor(nn.Module):
                        def __init__(self, feature_dim=512, pretrained=True):
                            super(EarFeatureExtractor, self).__init__()
                            
                            # Load EfficientNet-B4
                            if pretrained:
                                from torchvision.models import EfficientNet_B4_Weights
                                self.backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
                            else:
                                self.backbone = efficientnet_b4(weights=None)
                            
                            # Get the number of features from the classifier
                            if hasattr(self.backbone.classifier, 'in_features'):
                                in_features = self.backbone.classifier.in_features
                            else:
                                in_features = 1792  # EfficientNet-B4 default feature size
                            
                            # Replace classifier with feature projection
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
                            # L2 normalize features
                            features = torch.nn.functional.normalize(features, p=2, dim=1)
                            return features
                    
                    # Create model with standard architecture
                    self.model = EarFeatureExtractor(feature_dim=self.feature_dim, pretrained=True)
                
                # Move model to device first
                self.model.to(self.device)
                
                # Load trained weights
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # Set to evaluation mode
                self.model.eval()
                
                model_type = "Excellent" if is_excellent_model else "Standard"
                print(f"✓ Trained {model_type} ear model loaded: {path}")
                print(f"  Feature dimension: {self.feature_dim}")
                return True
                
            except Exception as e:
                print(f"Error loading trained model: {e}")
                print(f"Model path: {path}")
                print(f"Expected feature dim: {self.feature_dim}")
                print(f"Is excellent model: {is_excellent_model if 'is_excellent_model' in locals() else 'Unknown'}")
                if 'state_dict_keys' in locals():
                    print(f"First 10 keys in state_dict: {state_dict_keys[:10]}")
                import traceback
                traceback.print_exc()
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
            """Extract features from ear crop"""
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
            
            # Create features table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT,
                    feature_vector BLOB,
                    confidence REAL,
                    image_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    feature_dim INTEGER,
                    model_type TEXT,
                    FOREIGN KEY (person_id) REFERENCES persons (id)
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
            
            # Create current model state table
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
            
            # Migrate existing database schema if needed
            self.migrate_database_schema(cursor)
            
            conn.commit()
            conn.close()
        
        def migrate_database_schema(self, cursor):
            """Migrate database schema to add missing columns"""
            try:
                # Check if features table has all required columns
                cursor.execute("PRAGMA table_info(features)")
                columns = [column[1] for column in cursor.fetchall()]
                
                # Add missing columns if they don't exist
                required_columns = {
                    'image_path': 'TEXT',
                    'feature_dim': 'INTEGER',
                    'model_type': 'TEXT',
                    'confidence': 'REAL'
                }
                
                for column_name, column_type in required_columns.items():
                    if column_name not in columns:
                        print(f"🔧 Adding missing column: {column_name} ({column_type})")
                        cursor.execute(f"ALTER TABLE features ADD COLUMN {column_name} {column_type}")
                        
                        # Set default values for existing records
                        if column_name == 'confidence':
                            cursor.execute("UPDATE features SET confidence = 0.8 WHERE confidence IS NULL")
                        elif column_name == 'feature_dim':
                            cursor.execute("UPDATE features SET feature_dim = 512 WHERE feature_dim IS NULL")
                        elif column_name == 'model_type':
                            cursor.execute("UPDATE features SET model_type = 'Legacy' WHERE model_type IS NULL")
                        # image_path can remain NULL for existing records
                
                print("✅ Database schema migration completed")
                
            except Exception as e:
                print(f"⚠️ Database migration error: {e}")
                # Don't fail the entire setup if migration fails
        
        def add_person(self, person_id, name, feature_vectors, confidences=None, model_type=None, feature_dim=None, image_paths=None, face_photo=None):
            """Add person with feature vectors, model metadata, and face photo"""
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Insert person
                cursor.execute(
                    "INSERT OR REPLACE INTO persons (id, name, num_samples) VALUES (?, ?, ?)",
                    (person_id, name, len(feature_vectors))
                )
                
                # Insert features with model metadata
                for i, features in enumerate(feature_vectors):
                    confidence = confidences[i] if confidences else 0.8
                    feature_blob = pickle.dumps(features)
                    image_path = image_paths[i] if image_paths and i < len(image_paths) else None
                    
                    cursor.execute(
                        "INSERT INTO features (person_id, feature_vector, confidence, feature_dim, model_type, image_path) VALUES (?, ?, ?, ?, ?, ?)",
                        (person_id, feature_blob, confidence, feature_dim, model_type, image_path)
                    )
                
                # Insert face photo if provided
                if face_photo is not None:
                    face_photo_blob = pickle.dumps(face_photo)
                    cursor.execute(
                        "INSERT OR REPLACE INTO face_photos (person_id, face_photo) VALUES (?, ?)",
                        (person_id, face_photo_blob)
                    )
                
                conn.commit()
                conn.close()
                return True
                
            except Exception as e:
                print(f"Database error: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        def get_all_features(self, model_type=None, feature_dim=None, current_model_dim=None):
            """Get all features for matching with improved compatibility checking"""
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # First, try exact match
                if model_type and feature_dim:
                    print(f"📊 Looking for exact match: {model_type} ({feature_dim}D)")
                    cursor.execute(
                        "SELECT person_id, feature_vector, feature_dim, model_type, confidence, image_path FROM features WHERE model_type = ? AND feature_dim = ?",
                        (model_type, feature_dim)
                    )
                    results = cursor.fetchall()
                    
                    if not results:
                        print(f"📊 No exact match for {model_type} ({feature_dim}D), searching for compatible features...")
                        # Try to find any features from the same model type
                        cursor.execute(
                            "SELECT DISTINCT model_type, feature_dim FROM features WHERE model_type = ?",
                            (model_type,)
                        )
                        available = cursor.fetchall()
                        
                        if available:
                            print(f"📊 Available {model_type} features: {available}")
                            # Suggest compatible model if exists
                            for avail_type, avail_dim in available:
                                if avail_dim != feature_dim:
                                    print(f"💡 Suggestion: Switch to a model with {avail_dim}D features or clear database")
                        
                        # Try legacy features (no model_type specified)
                        print("📊 Trying legacy features...")
                        cursor.execute("SELECT person_id, feature_vector, feature_dim, model_type, confidence, image_path FROM features WHERE model_type IS NULL OR model_type = ''")
                        results = cursor.fetchall()
                        
                        if not results:
                            print("📊 Legacy query returned 0 results")
                            # Try ANY available features as last resort
                            print("📊 No compatible features found, trying ANY available features...")
                            cursor.execute("SELECT COUNT(*) FROM features")
                            total_features = cursor.fetchone()[0]
                            print(f"📊 Found {total_features} total features in database")
                            
                            if total_features == 0:
                                print("⚠️ No features in database at all")
                                conn.close()
                                return [], []
                            
                            print("⚠️ No compatible features in database")
                            conn.close()
                            return [], []
                else:
                    cursor.execute("SELECT person_id, feature_vector, feature_dim, model_type, confidence, image_path FROM features")
                    results = cursor.fetchall()
                
                if not results:
                    conn.close()
                    return [], []
                
                # Group features by person_id and validate dimensions
                person_features = {}
                valid_features = []
                invalid_count = 0
                
                for person_id, feature_blob, stored_dim, stored_model_type, confidence, image_path in results:
                    try:
                        feature_vector = pickle.loads(feature_blob)
                        
                        # Validate feature dimension
                        if feature_dim and len(feature_vector) != feature_dim:
                            invalid_count += 1
                            continue
                            
                        if person_id not in person_features:
                            person_features[person_id] = []
                        person_features[person_id].append({
                            'features': feature_vector,
                            'confidence': confidence,
                            'image_path': image_path,
                            'model_type': stored_model_type,
                            'feature_dim': stored_dim
                        })
                        
                        # Debug feature fingerprint
                        feature_hash = hash(tuple(feature_vector[:10]))  # First 10 elements
                        print(f"📊 Person {person_id}: using {len(person_features[person_id])} individual features")
                        print(f"📊 Individual fingerprints: {[f'{hash(tuple(f["features"][:10])):08x}' for f in person_features[person_id][-5:]]}")  # Last 5
                        print(f"📊 Features already normalized by model (no double normalization)")
                        
                    except Exception as e:
                        print(f"Error loading feature: {e}")
                        invalid_count += 1
                        continue
                
                if invalid_count > 0:
                    print(f"⚠️ Skipped {invalid_count} incompatible features")
                
                # Use individual features instead of averaging
                person_ids = []
                features = []
                
                for person_id, feature_list in person_features.items():
                    for feature_data in feature_list:
                        person_ids.append(person_id)
                        features.append(feature_data['features'])
                
                conn.close()
                
                if features:
                    features_array = np.array(features)
                    return person_ids, features_array
                
                return person_ids, features
                
            except Exception as e:
                print(f"Database query error: {e}")
                import traceback
                traceback.print_exc()
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
                
                # Delete features first
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
                return True
                
            except Exception as e:
                print(f"Error saving model state: {e}")
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
                    return model_type, feature_dim, model_path, model_name
                else:
                    print("📖 No saved model state found")
                    return None, None, None, None
                
            except Exception as e:
                print(f"Error loading model state: {e}")
                return None, None, None, None


class ModernTheme:
    """Ultra-modern theme with sophisticated design language"""
    
    # Sophisticated color palette with depth and elegance
    COLORS = {
        # Background system with depth
        'bg_primary': '#0A0A0B',           # Deep dark background
        'bg_secondary': '#1A1A1D',         # Elevated surface
        'bg_tertiary': '#2A2A2F',          # Card background
        'bg_card': '#1E1E23',              # Premium card surface
        'bg_glass': '#1A1A1D',             # Glass morphism base
        
        # Accent colors with modern vibrancy
        'accent_primary': '#6366F1',        # Modern indigo
        'accent_secondary': '#8B5CF6',      # Vibrant purple
        'accent_tertiary': '#10B981',       # Emerald green
        'accent_danger': '#EF4444',         # Modern red
        'accent_warning': '#F59E0B',        # Amber
        'accent_success': '#10B981',        # Success green
        'accent_info': '#06B6D4',           # Cyan
        
        # Text hierarchy with perfect contrast
        'text_primary': '#F8FAFC',          # Pure white text
        'text_secondary': '#94A3B8',        # Muted text
        'text_tertiary': '#64748B',         # Disabled text
        'text_light': '#FFFFFF',            # Bright white
        'text_accent': '#6366F1',           # Accent text
        
        # Sophisticated borders and shadows
        'border_light': '#374151',          # Subtle border
        'border_medium': '#4B5563',         # Medium border
        'border_accent': '#6366F1',         # Accent border
        'shadow_light': '#000000',          # Deep shadow
        'shadow_medium': '#000000',         # Medium shadow
        'shadow_heavy': '#000000',          # Heavy shadow
        
        # Gradient definitions
        'gradient_primary': ['#6366F1', '#8B5CF6'],    # Indigo to purple
        'gradient_success': ['#10B981', '#059669'],    # Green gradient
        'gradient_warning': ['#F59E0B', '#D97706'],    # Amber gradient
        'gradient_danger': ['#EF4444', '#DC2626'],     # Red gradient
        'gradient_dark': ['#1A1A1D', '#0A0A0B'],       # Dark gradient
        
        # Legacy compatibility
        'success': '#10B981',
        'warning': '#F59E0B',
        'error': '#EF4444',
        'border': '#374151',
        'shadow': '#000000',
        'toggle_bg': '#374151',
        'toggle_active': '#6366F1'
    }
    
    # Animation settings
    ANIMATION = {
        'duration_fast': 150,      # Fast animations (ms)
        'duration_normal': 300,    # Normal animations (ms)
        'duration_slow': 500,      # Slow animations (ms)
        'easing': 'ease-out',      # Animation easing
        'spring_tension': 0.3,     # Spring animation tension
        'spring_friction': 0.8,    # Spring animation friction
    }
    
    # Typography scale
    TYPOGRAPHY = {
        'font_family': 'Inter',  # Modern, clean font
        'font_family_fallback': 'Roboto',  # Google's modern font
        'size_xs': 10,
        'size_sm': 12,
        'size_base': 14,
        'size_lg': 16,
        'size_xl': 18,
        'size_2xl': 20,
        'size_3xl': 24,
        'size_4xl': 28,
        'weight_light': 'normal',
        'weight_normal': 'normal',
        'weight_medium': 'bold',
        'weight_bold': 'bold',
        'weight_heavy': 'bold',
    }
    
    # Spacing scale
    SPACING = {
        'xs': 4,
        'sm': 8,
        'md': 12,
        'lg': 16,
        'xl': 20,
        '2xl': 24,
        '3xl': 32,
        '4xl': 40,
        '5xl': 48,
    }
    
    # Border radius scale
    RADIUS = {
        'sm': 8,
        'md': 12,
        'lg': 18,
        'xl': 24,
        '2xl': 28,
        'full': 50,
    }
    
    @staticmethod
    def create_gradient_image(width, height, color1, color2, direction='vertical'):
        """Create a gradient image"""
        image = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(image)
        
        if direction == 'vertical':
            for y in range(height):
                ratio = y / height
                r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
                g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
                b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
                draw.line([(0, y), (width, y)], fill=(r, g, b))
        else:  # horizontal
            for x in range(width):
                ratio = x / width
                r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
                g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
                b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
                draw.line([(x, 0), (x, height)], fill=(r, g, b))
        
        return image
    
    @staticmethod
    def apply_glass_effect(image, blur_radius=2, opacity=0.8):
        """Apply glass effect to image"""
        # Apply gaussian blur
        blurred = image.filter(ImageFilter.GaussianBlur(blur_radius))
        
        # Enhance brightness slightly
        enhancer = ImageEnhance.Brightness(blurred)
        brightened = enhancer.enhance(1.1)
        
        # Create overlay
        overlay = Image.new('RGBA', image.size, (255, 255, 255, int(255 * (1 - opacity))))
        
        # Composite
        result = Image.alpha_composite(brightened.convert('RGBA'), overlay)
        return result


class AnimatedToggle:
    """Modern animated toggle switch"""
    def __init__(self, parent, left_text="Option 1", right_text="Option 2", 
                 width=200, height=40, command=None):
        self.parent = parent
        self.left_text = left_text
        self.right_text = right_text
        self.width = width
        self.height = height
        self.command = command
        self.active_side = "left"  # "left" or "right"
        self.animation_step = 0
        self.animating = False
        
        # Create canvas
        self.canvas = tk.Canvas(parent, width=width, height=height, 
                               highlightthickness=0, 
                               bg=ModernTheme.COLORS['bg_secondary'])
        self.canvas.pack()
        
        # Bind click events
        self.canvas.bind("<Button-1>", self.on_click)
        
        # Draw initial state
        self.draw_toggle()
    
    def draw_toggle(self):
        """Draw the toggle switch"""
        self.canvas.delete("all")
        
        # Toggle background (rounded rectangle)
        bg_color = ModernTheme.COLORS['toggle_bg']
        self.draw_rounded_rect(2, 2, self.width-2, self.height-2, 
                              radius=self.height//2, fill=bg_color, outline="")
        
        # Calculate slider position
        slider_width = (self.width - 8) // 2
        slider_height = self.height - 8
        
        if self.active_side == "left":
            slider_x = 4
        else:
            slider_x = self.width - slider_width - 4
        
        # Animate slider position if animating
        if self.animating:
            start_x = 4 if self.active_side == "right" else self.width - slider_width - 4
            end_x = 4 if self.active_side == "left" else self.width - slider_width - 4
            progress = self.animation_step / 10.0
            slider_x = start_x + (end_x - start_x) * progress
        
        # Draw active slider
        active_color = ModernTheme.COLORS['toggle_active']
        self.draw_rounded_rect(slider_x, 4, slider_x + slider_width, self.height - 4,
                              radius=(self.height-8)//2, fill=active_color, outline="")
        
        # Draw text labels
        left_color = ModernTheme.COLORS['text_light'] if self.active_side == "left" else ModernTheme.COLORS['text_secondary']
        right_color = ModernTheme.COLORS['text_light'] if self.active_side == "right" else ModernTheme.COLORS['text_secondary']
        
        # Left text
        self.canvas.create_text(slider_width//2 + 4, self.height//2, 
                               text=self.left_text, fill=left_color, 
                               font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 10, 'bold'))
        
        # Right text  
        self.canvas.create_text(self.width - slider_width//2 - 4, self.height//2,
                               text=self.right_text, fill=right_color,
                               font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 10, 'bold'))
    
    def draw_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        """Draw a rounded rectangle"""
        points = []
        for x, y in [(x1, y1 + radius), (x1, y1), (x1 + radius, y1),
                     (x2 - radius, y1), (x2, y1), (x2, y1 + radius),
                     (x2, y2 - radius), (x2, y2), (x2 - radius, y2),
                     (x1 + radius, y2), (x1, y2), (x1, y2 - radius)]:
            points.extend([x, y])
        
        return self.canvas.create_polygon(points, smooth=True, **kwargs)
    
    def on_click(self, event):
        """Handle click events"""
        if self.animating:
            return
            
        # Determine which side was clicked
        if event.x < self.width // 2:
            new_side = "left"
        else:
            new_side = "right"
        
        if new_side != self.active_side:
            self.active_side = new_side
            self.animate_toggle()
            if self.command:
                self.command(self.active_side)
    
    def animate_toggle(self):
        """Animate the toggle transition"""
        self.animating = True
        self.animation_step = 0
        self._animate_step()
    
    def _animate_step(self):
        """Single animation step"""
        if self.animation_step < 10:
            self.animation_step += 1
            self.draw_toggle()
            self.canvas.after(20, self._animate_step)
        else:
            self.animating = False
            self.draw_toggle()
    
    def set_active(self, side):
        """Programmatically set active side"""
        if side in ["left", "right"] and side != self.active_side:
            self.active_side = side
            if not self.animating:
                self.animate_toggle()

class ModernButton(tk.Canvas):
    """Custom modern button with glass effect"""
    
    def __init__(self, parent, text="", command=None, width=120, height=40, 
                 bg_color=ModernTheme.COLORS['accent_primary'], 
                 text_color=ModernTheme.COLORS['text_light'], 
                 style="primary", **kwargs):
        super().__init__(parent, width=width, height=height, 
                        highlightthickness=0, 
                        bg=ModernTheme.COLORS['bg_primary'],
                        **kwargs)
        
        self.text = text
        self.command = command
        self.width = width
        self.height = height
        self.bg_color = bg_color
        self.text_color = text_color
        self.style = style
        self.is_hovered = False
        self.is_pressed = False
        
        # Animation properties
        self.animation_id = None
        self.scale_factor = 1.0
        self.shadow_offset = 0
        self.target_scale = 1.0
        self.current_scale = 1.0
        self.target_shadow = 0
        self.current_shadow = 0
        
        self.draw_button()
        self.bind_events()
    
    def draw_button(self):
        """Draw ultra-modern button with sophisticated gradients and effects"""
        self.delete("all")
        
        # Calculate dimensions with scaling
        w = int(self.width * self.current_scale)
        h = int(self.height * self.current_scale)
        x_offset = (self.width - w) // 2
        y_offset = (self.height - h) // 2
        
        # Draw sophisticated shadow with multiple layers
        if self.current_shadow > 0:
            shadow_intensity = self.current_shadow
            # Outer glow shadow
            glow_color = self.hex_to_rgba(ModernTheme.COLORS['accent_primary'], 0.2 * shadow_intensity)
            self.create_rounded_rect(
                x_offset - 2, y_offset - 2 + self.shadow_offset,
                x_offset + w + 2, y_offset + h + 2 + self.shadow_offset,
                radius=ModernTheme.RADIUS['xl'], 
                fill=glow_color, outline=""
            )
            # Main shadow
            shadow_color = self.hex_to_rgba(ModernTheme.COLORS['shadow_light'], 0.3 * shadow_intensity)
            self.create_rounded_rect(
                x_offset + 3, y_offset + 3 + self.shadow_offset,
                x_offset + w - 3, y_offset + h - 3 + self.shadow_offset,
                radius=ModernTheme.RADIUS['lg'], 
                fill=shadow_color, outline=""
            )
        
        # Determine gradient colors based on style
        if self.style == "primary":
            colors = ModernTheme.COLORS['gradient_primary']
        elif self.style == "success":
            colors = ModernTheme.COLORS['gradient_success']
        elif self.style == "warning":
            colors = ModernTheme.COLORS['gradient_warning']
        elif self.style == "danger":
            colors = ModernTheme.COLORS['gradient_danger']
        else:
            colors = [self.bg_color, self.bg_color]
        
        # Apply state-based modifications
        if self.is_pressed:
            colors = [self.darken_color(colors[0], 0.2), self.darken_color(colors[1], 0.2)]
        elif self.is_hovered:
            colors = [self.lighten_color(colors[0], 0.1), self.lighten_color(colors[1], 0.1)]
        
        # Create gradient effect by drawing multiple rectangles
        gradient_steps = 20
        for i in range(gradient_steps):
            ratio = i / (gradient_steps - 1)
            # Interpolate between gradient colors
            r1, g1, b1 = self.hex_to_rgb(colors[0])
            r2, g2, b2 = self.hex_to_rgb(colors[1])
            r = int(r1 + (r2 - r1) * ratio)
            g = int(g1 + (g2 - g1) * ratio)
            b = int(b1 + (b2 - b1) * ratio)
            gradient_color = f"#{r:02x}{g:02x}{b:02x}"
            
            step_height = h // gradient_steps
            self.create_rounded_rect(
                x_offset, y_offset + i * step_height,
                x_offset + w, y_offset + (i + 1) * step_height,
                radius=0,  # No radius for gradient steps
                fill=gradient_color, outline=""
            )
        
        # Add sophisticated border with accent
        border_color = ModernTheme.COLORS['border_accent'] if self.is_hovered else ModernTheme.COLORS['border_light']
        self.create_rounded_rect(
            x_offset, y_offset,
            x_offset + w, y_offset + h,
            radius=ModernTheme.RADIUS['lg'], 
            fill="", outline=border_color, width=2
        )
        
        # Add premium text with glow effect
        font_size = int(ModernTheme.TYPOGRAPHY['size_base'] * self.current_scale)
        text_color = ModernTheme.COLORS['text_light']
        
        # Text shadow for depth
        self.create_text(
            self.width//2 + 1, self.height//2 + 1,
            text=self.text, fill=ModernTheme.COLORS['shadow_light'],
            font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], font_size, 'bold')
        )
        
        # Main text
        self.create_text(
            self.width//2, self.height//2,
            text=self.text, fill=text_color,
            font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], font_size, 'bold')
        )
    
    def create_rounded_rect(self, x1, y1, x2, y2, radius=5, **kwargs):
        """Create a rounded rectangle"""
        points = []
        for x, y in [(x1, y1 + radius), (x1, y1), (x1 + radius, y1),
                     (x2 - radius, y1), (x2, y1), (x2, y1 + radius),
                     (x2, y2 - radius), (x2, y2), (x2 - radius, y2),
                     (x1 + radius, y2), (x1, y2), (x1, y2 - radius)]:
            points.extend([x, y])
        return self.create_polygon(points, smooth=True, **kwargs)
    
    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def rgb_to_hex(self, rgb):
        """Convert RGB to hex"""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    def lighten_color(self, hex_color, factor):
        """Lighten a hex color by factor"""
        rgb = self.hex_to_rgb(hex_color)
        rgb = tuple(min(255, int(c + (255-c) * factor)) for c in rgb)
        return self.rgb_to_hex(rgb)
    
    def darken_color(self, hex_color, factor):
        """Darken a hex color by factor"""
        rgb = self.hex_to_rgb(hex_color)
        rgb = tuple(max(0, int(c * (1 - factor))) for c in rgb)
        return self.rgb_to_hex(rgb)
    
    def hex_to_rgba(self, hex_color, alpha=1.0):
        """Convert hex to a lighter/darker version for tkinter (simulating alpha)"""
        rgb = self.hex_to_rgb(hex_color)
        if alpha < 1.0:
            # Make color lighter to simulate transparency
            rgb = tuple(min(255, int(c + (255-c) * (1-alpha))) for c in rgb)
        return self.rgb_to_hex(rgb)
    
    def animate_to_target(self):
        """Smooth animation to target values"""
        # Scale animation
        scale_diff = self.target_scale - self.current_scale
        if abs(scale_diff) > 0.01:
            self.current_scale += scale_diff * 0.2
        else:
            self.current_scale = self.target_scale
        
        # Shadow animation
        shadow_diff = self.target_shadow - self.current_shadow
        if abs(shadow_diff) > 0.01:
            self.current_shadow += shadow_diff * 0.2
        else:
            self.current_shadow = self.target_shadow
        
        self.draw_button()
        
        # Continue animation if not at target
        if abs(scale_diff) > 0.01 or abs(shadow_diff) > 0.01:
            self.animation_id = self.after(16, self.animate_to_target)  # ~60fps
    
    def bind_events(self):
        """Bind mouse events"""
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_press)
        self.bind("<ButtonRelease-1>", self.on_release)
    
    def on_enter(self, event):
        """Handle mouse enter with smooth animation"""
        self.is_hovered = True
        self.target_scale = 1.05
        self.target_shadow = 1.0
        self.shadow_offset = 2
        self.config(cursor="hand2")
        self.animate_to_target()
    
    def on_leave(self, event):
        """Handle mouse leave with smooth animation"""
        self.is_hovered = False
        self.is_pressed = False
        self.target_scale = 1.0
        self.target_shadow = 0.0
        self.shadow_offset = 0
        self.config(cursor="")
        self.animate_to_target()
    
    def on_press(self, event):
        """Handle button press with animation"""
        self.is_pressed = True
        self.target_scale = 0.95
        self.target_shadow = 0.5
        self.shadow_offset = 1
        self.animate_to_target()
    
    def on_release(self, event):
        """Handle button release with animation"""
        self.is_pressed = False
        if self.is_hovered:
            self.target_scale = 1.05
            self.target_shadow = 1.0
            self.shadow_offset = 2
        else:
            self.target_scale = 1.0
            self.target_shadow = 0.0
            self.shadow_offset = 0
        self.animate_to_target()
        
        if self.command:
            self.command()


class GlassFrame(tk.Frame):
    """Premium glass morphism frame with blur effects"""
    
    def __init__(self, parent, glass_intensity=0.1, **kwargs):
        super().__init__(parent, 
                        bg=ModernTheme.COLORS['bg_secondary'],
                        relief='flat',
                        bd=0,
                        highlightbackground=ModernTheme.COLORS['border_light'],
                        highlightthickness=1,
                        **kwargs)
        self.glass_intensity = glass_intensity
        self.setup_glass_effect()
    
    def setup_glass_effect(self):
        """Setup glass morphism effect"""
        # Create a subtle border for glass effect
        self.configure(highlightbackground=ModernTheme.COLORS['border_light'],
                      highlightthickness=1)


class StatusIndicator(tk.Canvas):
    """Status indicator with animated glow effect"""
    
    def __init__(self, parent, status="inactive", size=12, **kwargs):
        super().__init__(parent, width=size*2, height=size*2, 
                        highlightthickness=0, 
                        bg=ModernTheme.COLORS['bg_secondary'],
                        **kwargs)
        self.size = size
        self.status = status
        self.glow_phase = 0
        self.animation_id = None
        self.draw_indicator()
    
    def draw_indicator(self):
        """Draw the status indicator with premium animations"""
        self.delete("all")
        
        colors = {
            'active': ModernTheme.COLORS['accent_success'],
            'warning': ModernTheme.COLORS['accent_warning'],
            'error': ModernTheme.COLORS['accent_danger'],
            'inactive': ModernTheme.COLORS['text_tertiary']
        }
        
        color = colors.get(self.status, colors['inactive'])
        
        # Outer glow (animated for active status)
        if self.status == 'active':
            glow_intensity = 0.4 + 0.3 * abs(np.sin(self.glow_phase))
            glow_color = self.lighten_color(color, 0.5 + glow_intensity)
            self.create_oval(2, 2, self.size*2-2, self.size*2-2, 
                           fill=glow_color, outline="")
        
        # Main indicator
        self.create_oval(4, 4, self.size*2-4, self.size*2-4, 
                        fill=color, outline="")
        
        # Inner highlight
        highlight = self.lighten_color(color, 1.5)
        self.create_oval(6, 6, self.size-2, self.size-2, 
                        fill=highlight, outline="")
    
    def set_status(self, status):
        """Set the status and update display"""
        self.status = status
        self.draw_indicator()
        
        if status == 'active' and not self.animation_id:
            self.animate()
        elif status != 'active' and self.animation_id:
            self.after_cancel(self.animation_id)
            self.animation_id = None
    
    def animate(self):
        """Animate the glow effect"""
        self.glow_phase += 0.2
        self.draw_indicator()
        self.animation_id = self.after(50, self.animate)
    
    
    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def rgb_to_hex(self, rgb):
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    def lighten_color(self, hex_color, factor):
        rgb = self.hex_to_rgb(hex_color)
        rgb = tuple(min(255, int(c * factor)) for c in rgb)
        return self.rgb_to_hex(rgb)


class IdentificationPopup:
    """In-app popup for identification results"""
    
    def __init__(self, parent, result_data, face_photo=None, timeout_callback=None, override_callback=None):
        self.parent = parent
        self.result_data = result_data
        self.face_photo = face_photo
        self.timeout_callback = timeout_callback
        self.override_callback = override_callback
        self.timeout_id = None
        self.is_destroyed = False
        
        # Create popup frame
        self.popup_frame = tk.Frame(parent, bg=ModernTheme.COLORS['bg_secondary'],
                                   relief='raised', borderwidth=3)
        self.popup_frame.place(relx=0.5, rely=0.3, anchor='center')
        
        self.create_popup_content()
        self.start_timeout()
    
    def create_popup_content(self):
        """Create the popup content"""
        # Header
        header_frame = tk.Frame(self.popup_frame, bg=ModernTheme.COLORS['accent_primary'])
        header_frame.pack(fill='x')
        
        header_label = tk.Label(header_frame, text="🔍 IDENTIFICATION RESULT",
                               bg=ModernTheme.COLORS['accent_primary'],
                               fg=ModernTheme.COLORS['text_light'],
                               font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 14, 'bold'),
                               pady=10)
        header_label.pack()
        
        # Content area
        content_frame = tk.Frame(self.popup_frame, bg=ModernTheme.COLORS['bg_secondary'],
                               padx=30, pady=20)
        content_frame.pack(fill='both', expand=True)
        
        # Result display
        person_name = self.result_data.get('person_name', 'Unknown')
        confidence = self.result_data.get('confidence', 0.0)
        
        if person_name != 'Unknown' and confidence > 0:
            # Show match
            result_color = ModernTheme.COLORS['success']
            result_icon = "✅"
            result_text = f"MATCH FOUND"
            
            # Face photo (if available)
            if self.face_photo is not None:
                try:
                    # Resize face photo for display
                    face_display = cv2.resize(self.face_photo, (100, 100))
                    face_rgb = cv2.cvtColor(face_display, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb)
                    face_tk = ImageTk.PhotoImage(face_pil)
                    
                    face_label = tk.Label(content_frame, image=face_tk,
                                         bg=ModernTheme.COLORS['bg_secondary'])
                    face_label.image = face_tk  # Keep reference
                    face_label.pack(pady=(0, 10))
                except Exception as e:
                    print(f"❌ Error displaying face photo: {e}")
            
            # Person name
            name_label = tk.Label(content_frame, text=person_name,
                                bg=ModernTheme.COLORS['bg_secondary'],
                                fg=ModernTheme.COLORS['text_primary'],
                                font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 18, 'bold'))
            name_label.pack(pady=(0, 10))
            
            # Face photo if available
            if 'face_photo' in self.result_data and self.result_data['face_photo'] is not None:
                try:
                    face_photo = self.result_data['face_photo']
                    face_rgb = cv2.cvtColor(face_photo, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb)
                    face_tk = ImageTk.PhotoImage(face_pil)
                    
                    face_label = tk.Label(content_frame, image=face_tk,
                                        bg=ModernTheme.COLORS['bg_tertiary'],
                                        relief='solid', borderwidth=2)
                    face_label.image = face_tk  # Keep reference
                    face_label.pack(pady=(0, 10))
                except Exception as e:
                    print(f"Error displaying face photo: {e}")
            
            # Confidence
            conf_label = tk.Label(content_frame, text=f"Confidence: {confidence:.1%}",
                                bg=ModernTheme.COLORS['bg_secondary'],
                                fg=result_color,
                                font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 12, 'bold'))
            conf_label.pack(pady=(0, 10))
            
        else:
            # No match
            result_color = ModernTheme.COLORS['error']
            result_icon = "❌"
            result_text = "NO MATCH FOUND"
            
            no_match_label = tk.Label(content_frame, text="Person not recognized",
                                    bg=ModernTheme.COLORS['bg_secondary'],
                                    fg=ModernTheme.COLORS['text_secondary'],
                                    font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 14))
            no_match_label.pack(pady=(0, 10))
        
        # Status indicator
        status_label = tk.Label(content_frame, text=f"{result_icon} {result_text}",
                              bg=ModernTheme.COLORS['bg_secondary'],
                              fg=result_color,
                              font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 16, 'bold'))
        status_label.pack(pady=(0, 15))
        
        # Timeout progress
        self.progress_frame = tk.Frame(content_frame, bg=ModernTheme.COLORS['bg_secondary'])
        self.progress_frame.pack(fill='x', pady=(0, 15))
        
        self.timeout_label = tk.Label(self.progress_frame, text="Auto-closing in 5s...",
                                    bg=ModernTheme.COLORS['bg_secondary'],
                                    fg=ModernTheme.COLORS['text_secondary'],
                                    font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 10))
        self.timeout_label.pack()
        
        # Buttons
        button_frame = tk.Frame(content_frame, bg=ModernTheme.COLORS['bg_secondary'])
        button_frame.pack(fill='x')
        
        # Override button
        override_btn = ModernButton(button_frame, text="🔄 Identify Again",
                                  command=self.on_override,
                                  width=130, height=35,
                                  bg_color=ModernTheme.COLORS['accent_primary'])
        override_btn.pack(side='left', padx=(0, 10))
        
        # Close button
        close_btn = ModernButton(button_frame, text="✓ OK",
                               command=self.close_popup,
                               width=80, height=35,
                               bg_color=ModernTheme.COLORS['success'])
        close_btn.pack(side='right')
    
    def start_timeout(self):
        """Start the timeout countdown"""
        self.timeout_seconds = 5
        self.update_timeout()
    
    def update_timeout(self):
        """Update timeout countdown"""
        if self.is_destroyed:
            return
            
        if self.timeout_seconds > 0:
            self.timeout_label.configure(text=f"Auto-closing in {self.timeout_seconds}s...")
            self.timeout_seconds -= 1
            self.timeout_id = self.parent.after(1000, self.update_timeout)
        else:
            self.close_popup()
    
    def on_override(self):
        """Handle override button click"""
        if self.override_callback:
            self.override_callback()
        self.close_popup()
    
    def close_popup(self):
        """Close the popup"""
        if self.is_destroyed:
            return
            
        self.is_destroyed = True
        
        if self.timeout_id:
            self.parent.after_cancel(self.timeout_id)
        
        if self.timeout_callback:
            self.timeout_callback()
        
        self.popup_frame.destroy()


class EarBiometricsGUI:
    """Main GUI application for Ear Biometrics System with Liquid Glass theme"""
    
    def __init__(self, root):
        self.root = root
        self.setup_window()
        
        # Core system components - supporting both Excellent and Ultimate models
        self.yolo_model = None
        self.feature_extractor = None  # Will be EfficientNetFeatureExtractor or UltimateFeatureExtractor
        self.database = BiometricDatabase("ear_biometrics_v3.db")
        self.current_model_type = "TIMM Models"  # Track current model type - default to TIMM
        
        # Camera and processing
        self.camera = None
        self.is_running = False
        self.current_camera_index = 0
        self.selected_camera = 0  # Default camera
        self.available_cameras = []
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Application state
        self.current_mode = "identify"  # "identify" or "enroll"
        self.enrollment_samples = []    # Store enrollment samples with metadata
        self.enrollment_images = []     # Store cropped ear images for preview
        self.sample_count = 0
        self.max_samples = 5
        
        # Quality control
        self.min_ear_size = 80
        self.target_ear_size = 150
        self.show_guidelines = True
        self.confidence_threshold = 0.90
        # Additional identification safeguards
        self.match_margin = 0.08  # required gap between best and second-best person
        self.min_votes_for_person = 2  # min neighbors among top-K belonging to best person
        
        # Streamlined identification system
        self.identification_state = "idle"  # "idle", "analyzing", "showing_result", "timeout"
        self.identification_start_time = 0
        self.identification_results = []
        self.current_popup = None
        self.analysis_duration = 3.0  # 3 seconds
        self.result_timeout = 5.0  # 5 seconds
        
        # Matching system
        self.knn_model = None
        self.person_ids = []
        self.feature_database = None
        
        # Sample storage directory
        self.samples_dir = Path("enrollment_samples")
        self.samples_dir.mkdir(exist_ok=True)
        
        # Initialize variables that might be used but not always defined
        self.debug_mode_var = tk.BooleanVar(value=False)
        self.save_samples_var = tk.BooleanVar(value=True)
        self.use_gpu_var = tk.BooleanVar(value=torch.cuda.is_available())
        self.resolution_var = tk.StringVar(value="640x480")
        self.auto_exposure_var = tk.BooleanVar(value=True)
        
        # GUI setup
        self.setup_styles()
        self.create_main_interface()
        self.find_cameras()
        self.load_all_models()
        
        # Ensure Load Models button is visible initially
        self.root.after(500, self.update_load_models_button_state)
        
        # Auto-load models after interface is ready
        self.root.after(2000, self.auto_load_models)
    
    def setup_window(self):
        """Setup main window properties for RPi5 5-inch touchscreen"""
        self.root.title("Ear Biometrics System")
        self.root.geometry("1024x600")  # RPi5 5-inch DSI touchscreen resolution
        self.root.configure(bg=ModernTheme.COLORS['bg_primary'])
        self.root.resizable(False, False)  # Fixed size for embedded display
        
        # Remove window decorations for embedded/kiosk mode
        # self.root.overrideredirect(True)  # Uncomment for fullscreen kiosk mode
        
        # Configure for touch interface
        self.root.configure(cursor="")  # Hide cursor for touch interface
        
        # Bind keyboard shortcuts
        self.setup_hotkeys()
        
        # Set window icon if available
        try:
            # You can add an icon file here
            pass
        except:
            pass
    
    def setup_hotkeys(self):
        """Setup keyboard shortcuts for accessibility"""
        # Mode switching
        self.root.bind('<F1>', lambda e: self.switch_to_identification())
        self.root.bind('<F2>', lambda e: self.switch_to_enrollment())
        self.root.bind('<F3>', lambda e: self.open_settings())
        
        # Camera controls
        self.root.bind('<space>', lambda e: self.toggle_camera())
        self.root.bind('<Return>', lambda e: self.capture_sample() if self.current_mode == "enroll" else None)
        
        # Navigation
        self.root.bind('<Escape>', lambda e: self.stop_camera())
        self.root.bind('<F11>', lambda e: self.toggle_fullscreen())
        
        # Focus on window to receive key events
        self.root.focus_set()
    
    def setup_styles(self):
        """Setup ttk styles for Liquid Glass theme"""
        style = ttk.Style()
        
        # Configure notebook (tab) style
        style.theme_use('clam')
        
        style.configure('LiquidGlass.TNotebook', 
                       background=ModernTheme.COLORS['bg_primary'],
                       borderwidth=0)
        
        style.configure('LiquidGlass.TNotebook.Tab',
                       background=ModernTheme.COLORS['bg_secondary'],
                       foreground=ModernTheme.COLORS['text_secondary'],
                       padding=[20, 10],
                       borderwidth=1,
                       relief='flat')
        
        style.map('LiquidGlass.TNotebook.Tab',
                 background=[('selected', ModernTheme.COLORS['accent_primary']),
                            ('active', ModernTheme.COLORS['bg_tertiary'])],
                 foreground=[('selected', ModernTheme.COLORS['text_primary']),
                            ('active', ModernTheme.COLORS['text_primary'])])
        
        # Configure other widgets
        style.configure('LiquidGlass.TFrame',
                       background=ModernTheme.COLORS['bg_secondary'],
                       relief='flat',
                       borderwidth=1)
        
        style.configure('LiquidGlass.TLabel',
                       background=ModernTheme.COLORS['bg_secondary'],
                       foreground=ModernTheme.COLORS['text_primary'],
                       font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 10))
        
        style.configure('Title.TLabel',
                       background=ModernTheme.COLORS['bg_secondary'],
                       foreground=ModernTheme.COLORS['accent_primary'],
                       font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 14, 'bold'))
        
        style.configure('LiquidGlass.TEntry',
                       fieldbackground=ModernTheme.COLORS['bg_tertiary'],
                       foreground=ModernTheme.COLORS['text_primary'],
                       borderwidth=1,
                       relief='flat')
        
        style.configure('LiquidGlass.TCombobox',
                       fieldbackground=ModernTheme.COLORS['bg_tertiary'],
                       foreground=ModernTheme.COLORS['text_primary'],
                       borderwidth=1,
                       relief='flat')
    
    def create_main_interface(self):
        """Create modern touch-friendly interface for RPi5"""
        # Main container
        self.main_container = tk.Frame(self.root, bg=ModernTheme.COLORS['bg_primary'])
        self.main_container.pack(fill='both', expand=True)
        
        # Create header with mode toggle and settings
        self.create_header()
        
        # Create main content area
        self.create_main_content()
        
        # Create footer with status and controls
        self.create_footer()
        
        # Initialize in identification mode
        self.switch_to_identification()
    
    def create_header(self):
        """Create ultra-modern header with sophisticated dark design"""
        header = tk.Frame(self.main_container, bg=ModernTheme.COLORS['bg_primary'], height=100)
        header.pack(fill='x', padx=0, pady=0)
        header.pack_propagate(False)
        
        # Add sophisticated gradient-like effect with multiple layers
        gradient_frame = tk.Frame(header, bg=ModernTheme.COLORS['bg_secondary'], height=2)
        gradient_frame.pack(fill='x', side='bottom')
        
        accent_frame = tk.Frame(header, bg=ModernTheme.COLORS['accent_primary'], height=1)
        accent_frame.pack(fill='x', side='bottom')
        
        # Left side - App title and status
        left_frame = tk.Frame(header, bg=ModernTheme.COLORS['bg_secondary'])
        left_frame.pack(side='left', fill='y', padx=20, pady=10)
        
        # Ultra-modern app title with sophisticated styling
        title_frame = tk.Frame(left_frame, bg=ModernTheme.COLORS['bg_secondary'])
        title_frame.pack(anchor='w')
        
        # Main title with accent color
        title_label = tk.Label(title_frame, text="👂 EAR BIOMETRICS", 
                              bg=ModernTheme.COLORS['bg_secondary'],
                              fg=ModernTheme.COLORS['text_primary'],
                              font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 
                                   ModernTheme.TYPOGRAPHY['size_2xl'], 
                                   ModernTheme.TYPOGRAPHY['weight_bold']))
        title_label.pack(side='left')
        
        # Accent line
        accent_line = tk.Frame(title_frame, bg=ModernTheme.COLORS['accent_primary'], 
                              width=60, height=3)
        accent_line.pack(side='left', padx=(10, 0), pady=(5, 0))
        
        # Status indicators
        status_frame = tk.Frame(left_frame, bg=ModernTheme.COLORS['bg_secondary'])
        status_frame.pack(anchor='w', fill='x', pady=(5, 0))
        
        # Modern status indicators with sophisticated styling
        self.model_status_header = tk.Label(status_frame, text="● MODEL: LOADING...", 
                                          bg=ModernTheme.COLORS['bg_secondary'],
                                          fg=ModernTheme.COLORS['accent_warning'],
                                          font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 
                                               ModernTheme.TYPOGRAPHY['size_xs'], 
                                               ModernTheme.TYPOGRAPHY['weight_bold']))
        self.model_status_header.pack(side='left', padx=(0, 20))
        
        self.camera_status_header = tk.Label(status_frame, text="● CAMERA: READY", 
                                           bg=ModernTheme.COLORS['bg_secondary'],
                                           fg=ModernTheme.COLORS['accent_success'],
                                           font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 
                                                ModernTheme.TYPOGRAPHY['size_xs'], 
                                                ModernTheme.TYPOGRAPHY['weight_bold']))
        self.camera_status_header.pack(side='left')
        
        # Center - Mode toggle
        center_frame = tk.Frame(header, bg=ModernTheme.COLORS['bg_secondary'])
        center_frame.pack(side='left', fill='both', expand=True)
        
        # Animated toggle switch
        self.mode_toggle = AnimatedToggle(center_frame, 
                                        left_text="🔍 Identify", 
                                        right_text="📝 Enroll",
                                        width=240, height=45,
                                        command=self.on_mode_toggle)
        self.mode_toggle.canvas.pack(expand=True, pady=15)
        
        # Right side - Settings and controls
        right_frame = tk.Frame(header, bg=ModernTheme.COLORS['bg_secondary'])
        right_frame.pack(side='right', fill='y', padx=20, pady=10)
        
        # Load Models button - PROMINENTLY VISIBLE IN HEADER
        self.header_load_models_button = ModernButton(right_frame, text="🤖 Load Models", 
                                                     command=self.load_models,
                                                     width=100, height=35,
                                                     style="primary")
        self.header_load_models_button.pack(side='right', padx=(10, 0))
        
        # Ultra-modern face image display with sophisticated styling
        self.face_image_frame = tk.Frame(right_frame, bg=ModernTheme.COLORS['bg_tertiary'], 
                                        width=70, height=70, relief='flat', borderwidth=0)
        self.face_image_frame.pack(side='right', padx=(10, 0))
        self.face_image_frame.pack_propagate(False)
        
        # Add sophisticated border with accent
        self.face_image_frame.configure(highlightbackground=ModernTheme.COLORS['accent_primary'],
                                       highlightthickness=2)
        
        # Inner frame for depth
        inner_frame = tk.Frame(self.face_image_frame, bg=ModernTheme.COLORS['bg_card'], 
                              width=66, height=66)
        inner_frame.pack(expand=True, padx=2, pady=2)
        inner_frame.pack_propagate(False)
        
        self.face_image_label = tk.Label(inner_frame, text="👤", 
                                        bg=ModernTheme.COLORS['bg_card'],
                                        fg=ModernTheme.COLORS['accent_primary'],
                                        font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 24))
        self.face_image_label.pack(expand=True)
        
        # Settings button
        self.settings_button = self.create_icon_button(right_frame, "⚙️", self.open_settings, 50)
        self.settings_button.pack(side='right', padx=(10, 0))
        
        # Camera toggle button
        self.camera_toggle_button = self.create_icon_button(right_frame, "📹", self.toggle_camera, 50)
        self.camera_toggle_button.pack(side='right', padx=(10, 0))
    
    def create_main_content(self):
        """Create ultra-modern main content area with sophisticated dark design"""
        # Main content container with sophisticated spacing
        content = tk.Frame(self.main_container, bg=ModernTheme.COLORS['bg_primary'])
        content.pack(fill='both', expand=True, padx=ModernTheme.SPACING['xl'], pady=(ModernTheme.SPACING['lg'], ModernTheme.SPACING['xl']))
        
        # Left panel - Video feed with ultra-modern styling
        self.video_panel = tk.Frame(content, bg=ModernTheme.COLORS['bg_secondary'], 
                                   relief='flat', borderwidth=0)
        self.video_panel.pack(side='left', fill='both', expand=True, padx=(0, ModernTheme.SPACING['lg']))
        
        # Add sophisticated border to video panel
        self.video_panel.configure(highlightbackground=ModernTheme.COLORS['border_light'],
                                  highlightthickness=1)
        
        # Video container with premium dark styling
        video_container = tk.Frame(self.video_panel, bg=ModernTheme.COLORS['bg_tertiary'])
        video_container.pack(fill='both', expand=True, padx=ModernTheme.SPACING['lg'], pady=ModernTheme.SPACING['lg'])
        
        # Video display with sophisticated typography
        self.video_label = tk.Label(video_container, 
                                   text="LOAD MODELS AND START CAMERA TO BEGIN",
                                   bg=ModernTheme.COLORS['bg_tertiary'],
                                   fg=ModernTheme.COLORS['text_secondary'],
                                   font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 
                                        ModernTheme.TYPOGRAPHY['size_lg'], 
                                        ModernTheme.TYPOGRAPHY['weight_bold']),
                                   width=50, height=20)
        self.video_label.pack(expand=True, fill='both', padx=ModernTheme.SPACING['lg'], pady=ModernTheme.SPACING['lg'])
        
        # Right panel - Information and controls with ultra-modern styling
        self.info_panel = tk.Frame(content, bg=ModernTheme.COLORS['bg_secondary'], 
                                  relief='flat', borderwidth=0)
        self.info_panel.pack(side='right', fill='y', padx=(ModernTheme.SPACING['lg'], 0))
        self.info_panel.configure(width=320)  # Wider for better content
        
        # Add sophisticated border to info panel
        self.info_panel.configure(highlightbackground=ModernTheme.COLORS['border_light'],
                                 highlightthickness=1)
        
        # Mode-specific content will be added here
        self.create_info_content()
    
    def create_footer(self):
        """Create ultra-modern footer with sophisticated dark design"""
        footer = tk.Frame(self.main_container, bg=ModernTheme.COLORS['bg_primary'], height=80)
        footer.pack(fill='x', padx=0, pady=0)
        footer.pack_propagate(False)
        
        # Add sophisticated gradient-like effect
        accent_frame = tk.Frame(footer, bg=ModernTheme.COLORS['accent_primary'], height=1)
        accent_frame.pack(fill='x', side='top')
        
        border_frame = tk.Frame(footer, bg=ModernTheme.COLORS['bg_secondary'], height=2)
        border_frame.pack(fill='x', side='top')
        
        # Left - Status messages
        status_frame = tk.Frame(footer, bg=ModernTheme.COLORS['bg_secondary'])
        status_frame.pack(side='left', fill='y', padx=20, pady=10)
        
        self.status_label = tk.Label(status_frame, text="READY", 
                                   bg=ModernTheme.COLORS['bg_secondary'],
                                   fg=ModernTheme.COLORS['accent_success'],
                                   font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 
                                        ModernTheme.TYPOGRAPHY['size_lg'], 
                                        ModernTheme.TYPOGRAPHY['weight_bold']))
        self.status_label.pack(anchor='w')
        
        self.fps_label_footer = tk.Label(status_frame, text="FPS: 0", 
                                       bg=ModernTheme.COLORS['bg_secondary'],
                                       fg=ModernTheme.COLORS['text_tertiary'],
                                       font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 
                                            ModernTheme.TYPOGRAPHY['size_xs'], 
                                            ModernTheme.TYPOGRAPHY['weight_bold']))
        self.fps_label_footer.pack(anchor='w')
        
        # Right - Quick actions
        actions_frame = tk.Frame(footer, bg=ModernTheme.COLORS['bg_secondary'])
        actions_frame.pack(side='right', fill='y', padx=20, pady=10)
        
        # Mode-specific action button (will be updated based on mode)
        self.main_action_button = ModernButton(actions_frame, text="Start Camera", 
                                             command=self.main_action,
                                             width=120, height=40,
                                             style="success")
        self.main_action_button.pack(side='right')
        
        # Load Models button - pack after main action button so it appears to the left
        self.load_models_button = ModernButton(actions_frame, text="🤖 Load Models", 
                                             command=self.load_models,
                                             width=110, height=40,
                                             style="primary")
        self.load_models_button.pack(side='right', padx=(0, 10))
        
        # Initially show Load Models button prominently
        self.update_load_models_button_state()
    
    def create_toggle_button(self, parent, text, active=False):
        """Create modern toggle button"""
        bg_color = ModernTheme.COLORS['accent_primary'] if active else ModernTheme.COLORS['bg_primary']
        
        button = ModernButton(parent, text=text, 
                            command=lambda: self.toggle_mode(text),
                            width=120, height=35,
                            bg_color=bg_color)
        return button
    
    def create_icon_button(self, parent, icon, command, size=40):
        """Create modern icon button"""
        button = ModernButton(parent, text=icon, 
                            command=command,
                            width=size, height=size,
                            bg_color=ModernTheme.COLORS['bg_tertiary'])
        return button
    
    def update_load_models_button_state(self):
        """Update Load Models button appearance based on model status"""
        # Update footer button if it exists
        if hasattr(self, 'load_models_button'):
            if self.yolo_model and self.feature_extractor:
                # Models are loaded - make button less prominent
                self.load_models_button.text = "🔄 Reload Models"
                self.load_models_button.bg_color = ModernTheme.COLORS['accent_tertiary']
            else:
                # Models need to be loaded - make button prominent
                self.load_models_button.text = "🤖 Load Models"
                self.load_models_button.bg_color = ModernTheme.COLORS['accent_primary']
            self.load_models_button.draw_button()
        
        # Update header button if it exists
        if hasattr(self, 'header_load_models_button'):
            if self.yolo_model and self.feature_extractor:
                # Models are loaded - make button less prominent
                self.header_load_models_button.text = "🔄 Reload"
                self.header_load_models_button.bg_color = ModernTheme.COLORS['accent_tertiary']
            else:
                # Models need to be loaded - make button prominent
                self.header_load_models_button.text = "🤖 Load Models"
                self.header_load_models_button.bg_color = ModernTheme.COLORS['accent_primary']
            self.header_load_models_button.draw_button()
    
    def create_info_content(self):
        """Create information panel content"""
        # Clear existing content
        for widget in self.info_panel.winfo_children():
            widget.destroy()
        
        # Create scrollable info container
        self.info_canvas = tk.Canvas(self.info_panel, bg=ModernTheme.COLORS['bg_secondary'], highlightthickness=0)
        self.info_scrollbar = tk.Scrollbar(self.info_panel, orient="vertical", command=self.info_canvas.yview)
        self.scrollable_info_frame = tk.Frame(self.info_canvas, bg=ModernTheme.COLORS['bg_secondary'])
        
        self.scrollable_info_frame.bind(
            "<Configure>",
            lambda e: self.info_canvas.configure(scrollregion=self.info_canvas.bbox("all"))
        )
        
        self.info_canvas.create_window((0, 0), window=self.scrollable_info_frame, anchor="nw")
        self.info_canvas.configure(yscrollcommand=self.info_scrollbar.set)
        
        # Pack canvas and scrollbar
        self.info_canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        self.info_scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to info canvas
        def _on_info_mousewheel(event):
            self.info_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.info_canvas.bind_all("<MouseWheel>", _on_info_mousewheel)
        
        info_container = self.scrollable_info_frame
        
        # Title
        title = tk.Label(info_container, 
                        text="Identification Results" if self.current_mode == "identify" else "Enrollment Progress",
                        bg=ModernTheme.COLORS['bg_secondary'],
                        fg=ModernTheme.COLORS['text_primary'],
                        font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 14, 'bold'))
        title.pack(anchor='w', pady=(0, 10))
        
        # Results area
        self.results_text = tk.Text(info_container, height=15, width=30,
                                   bg=ModernTheme.COLORS['bg_tertiary'],
                                   fg=ModernTheme.COLORS['text_primary'],
                                   font=('Consolas', 10),
                                   relief='flat', borderwidth=1,
                                   wrap=tk.WORD)
        self.results_text.pack(fill='both', expand=True)
        
        # Mode-specific controls
        if self.current_mode == "enroll":
            self.create_enrollment_controls_modern()
        else:
            self.create_identification_controls_modern()
    
    def create_enrollment_controls_modern(self):
        """Create modern enrollment controls"""
        controls_container = tk.Frame(self.scrollable_info_frame, bg=ModernTheme.COLORS['bg_secondary'])
        controls_container.pack(fill='x', padx=10, pady=(0, 10))
        
        # Person name input
        name_frame = tk.Frame(controls_container, bg=ModernTheme.COLORS['bg_secondary'])
        name_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(name_frame, text="Person Name:", 
                bg=ModernTheme.COLORS['bg_secondary'],
                fg=ModernTheme.COLORS['text_primary'],
                font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 12, 'bold')).pack(anchor='w')
        
        self.person_name_var = tk.StringVar()
        self.person_name_entry = tk.Entry(name_frame, textvariable=self.person_name_var,
                                         bg=ModernTheme.COLORS['bg_tertiary'],
                                         fg=ModernTheme.COLORS['text_primary'],
                                         font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 12),
                                         relief='flat', bd=10)
        self.person_name_entry.pack(fill='x', pady=(5, 0), ipady=8)
        
        # Sample counter
        self.sample_counter = tk.Label(controls_container, text="Samples: 0/5", 
                                     bg=ModernTheme.COLORS['bg_secondary'],
                                     fg=ModernTheme.COLORS['accent_primary'],
                                     font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 12, 'bold'))
        self.sample_counter.pack(anchor='w', pady=(10, 5))
        
        # Sample preview area
        preview_frame_container = tk.Frame(controls_container, bg=ModernTheme.COLORS['bg_secondary'])
        preview_frame_container.pack(fill='both', expand=True, pady=(10, 10))
        
        tk.Label(preview_frame_container, text="Sample Preview:", 
                bg=ModernTheme.COLORS['bg_secondary'],
                fg=ModernTheme.COLORS['text_primary'],
                font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 10, 'bold')).pack(anchor='w', pady=(0, 5))
        
        # Preview grid frame
        self.preview_frame = tk.Frame(preview_frame_container, bg=ModernTheme.COLORS['bg_tertiary'], 
                                     relief='raised', borderwidth=2, height=200)
        self.preview_frame.pack(fill='both', expand=True, pady=(0, 10))
        self.preview_frame.pack_propagate(False)  # Maintain fixed height
        
        # Add a placeholder label to make the frame visible
        self.preview_placeholder = tk.Label(self.preview_frame, 
                                           text="Samples will appear here\nas they are captured",
                                           bg=ModernTheme.COLORS['bg_tertiary'],
                                           fg=ModernTheme.COLORS['text_secondary'],
                                           font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 10))
        self.preview_placeholder.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Action buttons
        button_frame = tk.Frame(controls_container, bg=ModernTheme.COLORS['bg_secondary'])
        button_frame.pack(fill='x', pady=(10, 0))
        
        self.capture_button = ModernButton(button_frame, text="📷 Capture", 
                                         command=self.capture_sample,
                                         width=120, height=40,
                                         bg_color=ModernTheme.COLORS['accent_primary'])
        self.capture_button.pack(side='left', padx=(0, 5))
        
        self.save_button = ModernButton(button_frame, text="💾 Save", 
                                      command=self.accept_samples,
                                      width=80, height=40,
                                      bg_color=ModernTheme.COLORS['success'])
        self.save_button.pack(side='left', padx=5)
        
        self.clear_button = ModernButton(button_frame, text="🗑️", 
                                       command=self.clear_enrollment_samples,
                                       width=40, height=40,
                                       bg_color=ModernTheme.COLORS['error'])
        self.clear_button.pack(side='right')
        
        # Face photo section
        face_frame = tk.Frame(controls_container, bg=ModernTheme.COLORS['bg_secondary'])
        face_frame.pack(fill='x', pady=(10, 0))
        
        tk.Label(face_frame, text="Face Photo:", 
                bg=ModernTheme.COLORS['bg_secondary'],
                fg=ModernTheme.COLORS['text_primary'],
                font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 12, 'bold')).pack(anchor='w')
        
        # Face photo preview
        self.face_photo_frame = tk.Frame(face_frame, bg=ModernTheme.COLORS['bg_tertiary'],
                                       width=120, height=120, relief='solid', borderwidth=1)
        self.face_photo_frame.pack(anchor='w', pady=(5, 5))
        self.face_photo_frame.pack_propagate(False)
        
        self.face_photo_label = tk.Label(self.face_photo_frame, text="No Photo",
                                       bg=ModernTheme.COLORS['bg_tertiary'],
                                       fg=ModernTheme.COLORS['text_secondary'],
                                       font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 10))
        self.face_photo_label.pack(expand=True)
        
        # Face photo buttons
        face_button_frame = tk.Frame(face_frame, bg=ModernTheme.COLORS['bg_secondary'])
        face_button_frame.pack(fill='x', pady=(0, 5))
        
        self.capture_face_button = ModernButton(face_button_frame, text="📸 Capture Face", 
                                              command=self.capture_face_photo,
                                              width=130, height=30,
                                              bg_color=ModernTheme.COLORS['accent_secondary'])
        self.capture_face_button.pack(side='left', padx=(0, 5))
        
        self.select_face_button = ModernButton(face_button_frame, text="📁 Select", 
                                             command=self.select_face_photo,
                                             width=80, height=30,
                                             bg_color=ModernTheme.COLORS['accent_tertiary'])
        self.select_face_button.pack(side='left')
        
        # Initialize face photo storage
        self.current_face_photo = None
    
    def create_identification_controls_modern(self):
        """Create modern identification controls"""
        controls_container = tk.Frame(self.scrollable_info_frame, bg=ModernTheme.COLORS['bg_secondary'])
        controls_container.pack(fill='x', padx=10, pady=(0, 10))
        
        # Database info
        db_frame = tk.Frame(controls_container, bg=ModernTheme.COLORS['bg_secondary'])
        db_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(db_frame, text="Database:", 
                bg=ModernTheme.COLORS['bg_secondary'],
                fg=ModernTheme.COLORS['text_primary'],
                font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 12, 'bold')).pack(anchor='w')
        
        self.db_info_label = tk.Label(db_frame, text="0 persons enrolled", 
                                    bg=ModernTheme.COLORS['bg_secondary'],
                                    fg=ModernTheme.COLORS['text_secondary'],
                                    font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 10))
        self.db_info_label.pack(anchor='w')
        
        # Update database info immediately
        self.update_database()
        
        # Quick actions
        button_frame = tk.Frame(controls_container, bg=ModernTheme.COLORS['bg_secondary'])
        button_frame.pack(fill='x', pady=(10, 0))
        
        self.database_button = ModernButton(button_frame, text="📊 Database", 
                                          command=self.open_database,
                                          width=120, height=40,
                                          bg_color=ModernTheme.COLORS['accent_secondary'])
        self.database_button.pack(side='left')
        
        self.refresh_button = ModernButton(button_frame, text="🔄", 
                                         command=self.update_database,
                                         width=40, height=40,
                                         bg_color=ModernTheme.COLORS['accent_tertiary'])
        self.refresh_button.pack(side='right')
    
    # Mode switching methods
    def on_mode_toggle(self, side):
        """Handle animated toggle switch"""
        if side == "left":
            self.switch_to_identification()
        else:
            self.switch_to_enrollment()
    
    def switch_to_identification(self):
        """Switch to identification mode"""
        self.current_mode = "identify"
        
        # Update toggle switch
        if hasattr(self, 'mode_toggle'):
            self.mode_toggle.set_active("left")
        
        # Update main action button
        self.main_action_button.text = "🔍 Start Identify"
        self.main_action_button.bg_color = ModernTheme.COLORS['success']
        self.main_action_button.draw_button()
        
        # Update status
        self.status_label.configure(text="Identification Mode", 
                                  fg=ModernTheme.COLORS['accent_primary'])
        
        # Recreate info content
        self.create_info_content()
    
    def switch_to_enrollment(self):
        """Switch to enrollment mode"""
        self.current_mode = "enroll"
        
        # Update toggle switch
        if hasattr(self, 'mode_toggle'):
            self.mode_toggle.set_active("right")
        
        # Update main action button
        self.main_action_button.text = "📝 Start Enroll"
        self.main_action_button.bg_color = ModernTheme.COLORS['warning']
        self.main_action_button.draw_button()
        
        # Update status
        self.status_label.configure(text="Enrollment Mode", 
                                  fg=ModernTheme.COLORS['warning'])
        
        # Initialize enrollment
        self.sample_count = 0
        self.enrollment_samples = []
        self.enrollment_images = []
        
        # Recreate info content
        self.create_info_content()
    
    # Action methods
    def main_action(self):
        """Main action button handler"""
        if self.current_mode == "identify":
            self.start_identification()
        else:
            self.start_enrollment()
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.is_running:
            self.stop_camera()
            self.camera_toggle_button.text = "📹"
            self.camera_status_header.configure(text="⚪ Camera: Ready")
        else:
            self.start_camera()
            self.camera_toggle_button.text = "⏹️"
            self.camera_status_header.configure(text="🟢 Camera: Active")
        self.camera_toggle_button.draw_button()
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        current_state = self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', not current_state)
    
    def capture_sample(self):
        """Manually capture a sample (for enrollment)"""
        if self.current_mode == "enroll" and self.is_running:
            # Force capture a sample if we have a detection
            if hasattr(self, 'last_detection') and self.last_detection:
                try:
                    # Get the last detection
                    ear_crop = self.last_detection['ear_crop']
                    confidence = self.last_detection['confidence']
                    ear_size = self.last_detection['ear_size']
                    
                    # Process as enrollment detection
                    self.process_enrollment_detection(ear_crop, confidence, ear_size)
                    
                    # Update status
                    if hasattr(self, 'status_label'):
                        self.status_label.configure(text=f"Captured sample {self.sample_count}", 
                                                  fg=ModernTheme.COLORS['success'])
                    
                except Exception as e:
                    print(f"❌ Error capturing sample: {e}")
                    if hasattr(self, 'status_label'):
                        self.status_label.configure(text="Error capturing sample", 
                                                  fg=ModernTheme.COLORS['error'])
            else:
                if hasattr(self, 'status_label'):
                    self.status_label.configure(text="No ear detected - position ear in view", 
                                              fg=ModernTheme.COLORS['warning'])
        else:
            if hasattr(self, 'status_label'):
                self.status_label.configure(text="Start enrollment first", 
                                          fg=ModernTheme.COLORS['error'])
    
    def open_settings(self):
        """Open streamlined settings dialog"""
        self.create_settings_dialog()
    
    def open_database(self):
        """Open database management dialog"""
        self.create_database_dialog()
    
    def start_identification(self):
        """Start identification process"""
        if not self.is_running:
            self.start_camera()
        self.main_action_button.text = "🔍 Identifying..."
        self.main_action_button.bg_color = ModernTheme.COLORS['warning']
        self.main_action_button.draw_button()
    
    def start_enrollment(self):
        """Start enrollment process"""
        if not self.person_name_var.get().strip():
            self.status_label.configure(text="Please enter person name", 
                                      fg=ModernTheme.COLORS['error'])
            return
        
        if not self.is_running:
            self.start_camera()
        self.main_action_button.text = "📝 Enrolling..."
        self.main_action_button.bg_color = ModernTheme.COLORS['warning']
        self.main_action_button.draw_button()
    
    def clear_enrollment_samples(self):
        """Clear enrollment samples"""
        self.enrollment_samples = []
        self.enrollment_images = []
        self.sample_count = 0
        self.update_sample_counter()
        self.status_label.configure(text="Samples cleared", 
                                  fg=ModernTheme.COLORS['warning'])
    
    def update_database(self):
        """Update database info display"""
        if hasattr(self, 'db_info_label'):
            persons = self.database.get_persons()
            total_samples = sum(num_samples for _, _, num_samples, _ in persons) if persons else 0
            print(f"DEBUG: update_database() - Found {len(persons)} persons, {total_samples} samples")
            self.db_info_label.configure(text=f"{len(persons)} persons, {total_samples} samples")
    
    def accept_samples(self):
        """Accept and save enrollment samples"""
        if not self.enrollment_samples:
            self.status_label.configure(text="No samples to save", 
                                      fg=ModernTheme.COLORS['error'])
            return
        
        person_name = self.person_name_var.get().strip()
        if not person_name:
            self.status_label.configure(text="Please enter person name", 
                                      fg=ModernTheme.COLORS['error'])
            return
        
        try:
            # Add person to database
            self.database.add_person(person_name, self.enrollment_samples)
            
            # Clear samples
            self.clear_enrollment_samples()
            
            # Update UI
            self.status_label.configure(text=f"✅ {person_name} enrolled successfully", 
                                      fg=ModernTheme.COLORS['success'])
            self.person_name_var.set("")
            
            # Update database info
            self.update_database()
            
            # Switch back to identification mode
            self.root.after(2000, self.switch_to_identification)
            
        except Exception as e:
            self.status_label.configure(text=f"Enrollment failed: {str(e)}", 
                                      fg=ModernTheme.COLORS['error'])
    
    def create_settings_dialog(self):
        """Create streamlined settings dialog for touch interface"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("700x500")
        settings_window.configure(bg=ModernTheme.COLORS['bg_primary'])
        settings_window.resizable(True, True)
        
        # Center the window
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # Header
        header = tk.Frame(settings_window, bg=ModernTheme.COLORS['bg_secondary'], height=60)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(header, text="⚙️ Settings", 
                bg=ModernTheme.COLORS['bg_secondary'],
                fg=ModernTheme.COLORS['text_primary'],
                font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 16, 'bold')).pack(pady=15)
        
        # Content
        content = tk.Frame(settings_window, bg=ModernTheme.COLORS['bg_primary'])
        content.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Model Selection
        model_frame = GlassFrame(content)
        model_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(model_frame, text="🤖 Model Configuration", 
                bg=ModernTheme.COLORS['bg_secondary'],
                fg=ModernTheme.COLORS['text_primary'],
                font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 12, 'bold')).pack(anchor='w', padx=15, pady=(15, 5))
        
        # Model type
        model_type_frame = tk.Frame(model_frame, bg=ModernTheme.COLORS['bg_secondary'])
        model_type_frame.pack(fill='x', padx=15, pady=(0, 10))
        
        tk.Label(model_type_frame, text="Model Type:", 
                bg=ModernTheme.COLORS['bg_secondary'],
                fg=ModernTheme.COLORS['text_secondary'],
                font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 10)).pack(anchor='w')
        
        self.settings_model_type = ttk.Combobox(model_type_frame, values=["Excellent", "Ultimate", "TIMM Models"],
                                              state="readonly", font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 10))
        self.settings_model_type.set(self.current_model_type)
        self.settings_model_type.pack(fill='x', pady=(2, 0), ipady=5)
        self.settings_model_type.bind('<<ComboboxSelected>>', self.on_settings_model_type_changed)
        
        # Specific model selection
        specific_model_frame = tk.Frame(model_frame, bg=ModernTheme.COLORS['bg_secondary'])
        specific_model_frame.pack(fill='x', padx=15, pady=(0, 10))
        
        tk.Label(specific_model_frame, text="Specific Model:", 
                bg=ModernTheme.COLORS['bg_secondary'],
                fg=ModernTheme.COLORS['text_secondary'],
                font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 10)).pack(anchor='w')
        
        self.settings_specific_model = ttk.Combobox(specific_model_frame, 
                                                  state="readonly", font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 10))
        self.settings_specific_model.pack(fill='x', pady=(2, 0), ipady=5)
        
        # Populate specific models based on current type
        self.update_settings_model_list()
        
        # Set default selection to retrained model if TIMM Models is selected
        if self.current_model_type == "TIMM Models" and hasattr(self, 'settings_specific_model'):
            model_values = self.settings_specific_model['values']
            for i, model_text in enumerate(model_values):
                if "efficientnet_retrained_final" in model_text:
                    self.settings_specific_model.current(i)
                    break
        
        # Quality Settings
        quality_frame = GlassFrame(content)
        quality_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(quality_frame, text="🎯 Detection Settings", 
                bg=ModernTheme.COLORS['bg_secondary'],
                fg=ModernTheme.COLORS['text_primary'],
                font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 12, 'bold')).pack(anchor='w', padx=15, pady=(15, 5))
        
        # Confidence threshold
        conf_frame = tk.Frame(quality_frame, bg=ModernTheme.COLORS['bg_secondary'])
        conf_frame.pack(fill='x', padx=15, pady=(0, 10))
        
        tk.Label(conf_frame, text=f"Confidence Threshold: {self.confidence_threshold:.2f}", 
                bg=ModernTheme.COLORS['bg_secondary'],
                fg=ModernTheme.COLORS['text_secondary'],
                font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 10)).pack(anchor='w')
        
        self.settings_confidence = tk.Scale(conf_frame, from_=0.5, to=0.95, resolution=0.05,
                                          orient='horizontal', length=300,
                                          bg=ModernTheme.COLORS['bg_secondary'],
                                          fg=ModernTheme.COLORS['text_primary'],
                                          highlightthickness=0)
        self.settings_confidence.set(self.confidence_threshold)
        self.settings_confidence.pack(fill='x', pady=(2, 0))
        
        # Buttons - Add separator and make more prominent
        separator = tk.Frame(settings_window, height=2, bg=ModernTheme.COLORS['border'])
        separator.pack(fill='x', padx=20, pady=10)
        
        button_frame = tk.Frame(settings_window, bg=ModernTheme.COLORS['bg_primary'])
        button_frame.pack(fill='x', padx=20, pady=(10, 20))
        
        # Make buttons larger and more visible
        ModernButton(button_frame, text="💾 Save Settings", 
                   command=lambda: self.save_settings(settings_window),
                   width=140, height=45,
                   bg_color=ModernTheme.COLORS['success']).pack(side='right', padx=(10, 0))
        
        ModernButton(button_frame, text="❌ Cancel", 
                   command=settings_window.destroy,
                   width=100, height=45,
                   bg_color=ModernTheme.COLORS['error']).pack(side='right')
    
    def create_database_dialog(self):
        """Create streamlined database management dialog"""
        db_window = tk.Toplevel(self.root)
        db_window.title("Database")
        db_window.geometry("700x500")
        db_window.configure(bg=ModernTheme.COLORS['bg_primary'])
        db_window.resizable(False, False)
        
        # Center the window
        db_window.transient(self.root)
        db_window.grab_set()
        
        # Header
        header = tk.Frame(db_window, bg=ModernTheme.COLORS['bg_secondary'], height=60)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(header, text="📊 Database Management", 
                bg=ModernTheme.COLORS['bg_secondary'],
                fg=ModernTheme.COLORS['text_primary'],
                font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 16, 'bold')).pack(pady=15)
        
        # Content
        content = tk.Frame(db_window, bg=ModernTheme.COLORS['bg_primary'])
        content.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Stats
        stats_frame = GlassFrame(content)
        stats_frame.pack(fill='x', pady=(0, 15))
        
        persons = self.database.get_persons()
        total_samples = sum(num_samples for _, _, num_samples, _ in persons) if persons else 0
        
        stats_text = f"👥 {len(persons)} Persons  •  📸 {total_samples} Samples  •  🤖 {self.current_model_type}"
        
        tk.Label(stats_frame, text=stats_text, 
                bg=ModernTheme.COLORS['bg_secondary'],
                fg=ModernTheme.COLORS['text_primary'],
                font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 12)).pack(pady=15)
        
        # Persons list
        list_frame = GlassFrame(content)
        list_frame.pack(fill='both', expand=True, pady=(0, 15))
        
        tk.Label(list_frame, text="Enrolled Persons", 
                bg=ModernTheme.COLORS['bg_secondary'],
                fg=ModernTheme.COLORS['text_primary'],
                font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 12, 'bold')).pack(anchor='w', padx=15, pady=(15, 5))
        
        # Listbox with scrollbar
        list_container = tk.Frame(list_frame, bg=ModernTheme.COLORS['bg_secondary'])
        list_container.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        
        self.db_listbox = tk.Listbox(list_container, 
                                   bg=ModernTheme.COLORS['bg_tertiary'],
                                   fg=ModernTheme.COLORS['text_primary'],
                                   font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 10),
                                   relief='flat', borderwidth=0,
                                   selectbackground=ModernTheme.COLORS['accent_primary'],
                                   selectmode=tk.SINGLE,
                                   exportselection=False)
        
        scrollbar = tk.Scrollbar(list_container, orient='vertical', command=self.db_listbox.yview)
        self.db_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.db_listbox.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Populate list
        print(f"DEBUG: Found {len(persons)} persons in database")
        for person_id, name, num_samples, created_at in persons:
            print(f"DEBUG: Person - ID: {person_id}, Name: {name}, Samples: {num_samples}")
            self.db_listbox.insert(tk.END, f"{name} ({num_samples} samples)")
        
        # Action buttons
        button_frame = tk.Frame(db_window, bg=ModernTheme.COLORS['bg_primary'])
        button_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        ModernButton(button_frame, text="✏️ Edit Name", 
                   command=lambda: self.edit_person_name_from_dialog(db_window),
                   width=120, height=40,
                   bg_color=ModernTheme.COLORS['accent_secondary']).pack(side='left')
        
        ModernButton(button_frame, text="📷 Edit Photo", 
                   command=lambda: self.edit_person_photo_from_dialog(db_window),
                   width=120, height=40,
                   bg_color=ModernTheme.COLORS['accent_tertiary']).pack(side='left', padx=(5, 0))
        
        ModernButton(button_frame, text="🗑️ Delete Selected", 
                   command=lambda: self.delete_selected_from_dialog(db_window),
                   width=140, height=40,
                   bg_color=ModernTheme.COLORS['error']).pack(side='left', padx=(5, 0))
        
        ModernButton(button_frame, text="🔄 Refresh", 
                   command=lambda: self.refresh_database_dialog(db_window),
                   width=100, height=40,
                   bg_color=ModernTheme.COLORS['accent_tertiary']).pack(side='left', padx=(5, 0))
        
        ModernButton(button_frame, text="✅ Close", 
                   command=db_window.destroy,
                   width=100, height=40,
                   bg_color=ModernTheme.COLORS['success']).pack(side='right')
    
    def on_settings_model_type_changed(self, event=None):
        """Handle model type change in settings"""
        self.update_settings_model_list()
    
    def update_settings_model_list(self):
        """Update the specific model list based on selected type"""
        model_type = self.settings_model_type.get()
        
        if model_type == "Excellent":
            models = self.excellent_models
            model_names = [model.name for model in models]
        elif model_type == "Ultimate":
            models = self.ultimate_models
            model_names = [model.name for model in models]
        elif model_type == "TIMM Models":
            models = self.timm_models
            model_names = [f"{model.get('display_name', model['name'])} - {model['performance']}" for model in models]
        else:
            model_names = []
        
        self.settings_specific_model['values'] = model_names
        if model_names:
            # Try to select the current model if it exists, otherwise select the first (default)
            current_selection = 0
            if hasattr(self, 'feature_extractor') and self.feature_extractor:
                if hasattr(self.feature_extractor, 'model_name'):
                    current_model = self.feature_extractor.model_name
                    # Look for the current model in the list
                    for i, model_name in enumerate(model_names):
                        if current_model in model_name or (hasattr(self, 'model_paths') and current_model in self.model_paths[i] if i < len(self.model_paths) else False):
                            current_selection = i
                            break
            self.settings_specific_model.current(current_selection)
    
    def save_settings(self, window):
        """Save settings and close dialog"""
        # Update confidence threshold
        self.confidence_threshold = self.settings_confidence.get()
        
        # Update model type and specific model if changed
        new_model_type = self.settings_model_type.get()
        selected_specific_model = self.settings_specific_model.get()
        
        model_changed = False
        
        # Check if model type changed
        if new_model_type != self.current_model_type:
            self.current_model_type = new_model_type
            model_changed = True
            print(f"🔄 Model type changed to: {new_model_type}")
        
        # Check if specific model changed
        if selected_specific_model:
            # Extract the actual model name from the display text
            if new_model_type == "TIMM Models":
                # For TIMM models, extract the model name before the " - " separator
                if " - " in selected_specific_model:
                    model_name = selected_specific_model.split(" - ")[0]
                else:
                    model_name = selected_specific_model
                
                # Find the actual TIMM model name
                for timm_model in self.timm_models:
                    display_name = timm_model.get('display_name', timm_model['name'])
                    if display_name == model_name or timm_model['name'] == model_name:
                        # Update the model paths to prioritize this model
                        self.prioritize_model(timm_model['name'])
                        model_changed = True
                        print(f"🔄 Selected specific TIMM model: {display_name} ({timm_model['name']})")
                        break
            else:
                # For Excellent/Ultimate models, the selection contains the filename
                model_changed = True
                print(f"🔄 Selected specific {new_model_type} model: {selected_specific_model}")
        
        # Reload models if anything changed
        if model_changed:
            print("🔄 Reloading models with new selection...")
            success = self.load_models()
            if success:
                window.destroy()
                self.status_label.configure(text="Settings saved - Models reloaded", fg=ModernTheme.COLORS['success'])
            else:
                self.status_label.configure(text="Settings saved - Model reload failed", fg=ModernTheme.COLORS['warning'])
                window.destroy()
        else:
            window.destroy()
            self.status_label.configure(text="Settings saved", fg=ModernTheme.COLORS['success'])
    
    def delete_selected_from_dialog(self, window):
        """Delete selected person from database dialog"""
        selection = self.db_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a person to delete")
            return
        
        # Get the selected person info
        selected_index = selection[0]
        persons = self.database.get_persons()
        if selected_index >= len(persons):
            messagebox.showerror("Error", "Invalid selection")
            return
        
        person_id, person_name, num_samples, created_at = persons[selected_index]
        
        # Confirm deletion
        result = messagebox.askyesno("Confirm Delete", 
                                   f"Are you sure you want to delete '{person_name}' with {num_samples} samples?\n\nThis action cannot be undone.")
        if result:
            try:
                success = self.database.delete_person(person_id)
                if success:
                    messagebox.showinfo("Success", f"Successfully deleted '{person_name}'")
                    # Refresh the dialog
                    self.refresh_database_dialog(window)
                    # Update the main interface
                    self.update_database()
                else:
                    messagebox.showerror("Error", f"Failed to delete '{person_name}'")
            except Exception as e:
                messagebox.showerror("Error", f"Error deleting person: {str(e)}")
    
    def edit_person_name_from_dialog(self, window):
        """Edit selected person's name from database dialog"""
        selection = self.db_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a person to edit")
            return
        
        # Get the selected person info
        selected_index = selection[0]
        persons = self.database.get_persons()
        if selected_index >= len(persons):
            messagebox.showerror("Error", "Invalid selection")
            return
        
        person_id, current_name, num_samples, created_at = persons[selected_index]
        
        # Create edit dialog
        edit_window = tk.Toplevel(window)
        edit_window.title("Edit Person Name")
        edit_window.geometry("400x200")
        edit_window.configure(bg=ModernTheme.COLORS['bg_primary'])
        edit_window.transient(window)
        edit_window.grab_set()
        
        # Center the window
        edit_window.geometry("+%d+%d" % (window.winfo_rootx() + 50, window.winfo_rooty() + 50))
        
        # Content
        content = tk.Frame(edit_window, bg=ModernTheme.COLORS['bg_primary'])
        content.pack(fill='both', expand=True, padx=20, pady=20)
        
        tk.Label(content, text="Edit Person Name", 
                bg=ModernTheme.COLORS['bg_primary'],
                fg=ModernTheme.COLORS['text_primary'],
                font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 14, 'bold')).pack(pady=(0, 20))
        
        tk.Label(content, text="Current Name:", 
                bg=ModernTheme.COLORS['bg_primary'],
                fg=ModernTheme.COLORS['text_primary'],
                font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 10)).pack(anchor='w')
        
        tk.Label(content, text=current_name, 
                bg=ModernTheme.COLORS['bg_primary'],
                fg=ModernTheme.COLORS['text_secondary'],
                font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 10, 'italic')).pack(anchor='w', pady=(0, 10))
        
        tk.Label(content, text="New Name:", 
                bg=ModernTheme.COLORS['bg_primary'],
                fg=ModernTheme.COLORS['text_primary'],
                font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 10)).pack(anchor='w')
        
        name_var = tk.StringVar(value=current_name)
        name_entry = tk.Entry(content, textvariable=name_var,
                            bg=ModernTheme.COLORS['bg_tertiary'],
                            fg=ModernTheme.COLORS['text_primary'],
                            font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 10),
                            relief='flat', borderwidth=1)
        name_entry.pack(fill='x', pady=(5, 20))
        name_entry.focus()
        name_entry.select_range(0, tk.END)
        
        # Buttons
        button_frame = tk.Frame(content, bg=ModernTheme.COLORS['bg_primary'])
        button_frame.pack(fill='x')
        
        def save_name():
            new_name = name_var.get().strip()
            if not new_name:
                messagebox.showerror("Error", "Name cannot be empty")
                return
            if new_name == current_name:
                messagebox.showinfo("Info", "Name unchanged")
                edit_window.destroy()
                return
            
            try:
                # Update name in database
                conn = sqlite3.connect(self.database.db_path)
                cursor = conn.cursor()
                cursor.execute("UPDATE persons SET name = ? WHERE id = ?", (new_name, person_id))
                conn.commit()
                conn.close()
                
                messagebox.showinfo("Success", f"Name updated from '{current_name}' to '{new_name}'")
                edit_window.destroy()
                # Refresh the dialog
                self.refresh_database_dialog(window)
                # Update the main interface
                self.update_database()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to update name: {str(e)}")
        
        ModernButton(button_frame, text="💾 Save", 
                   command=save_name,
                   width=100, height=40,
                   bg_color=ModernTheme.COLORS['success']).pack(side='left')
        
        ModernButton(button_frame, text="❌ Cancel", 
                   command=edit_window.destroy,
                   width=100, height=40,
                   bg_color=ModernTheme.COLORS['error']).pack(side='right')
        
        # Bind Enter key
        name_entry.bind('<Return>', lambda e: save_name())
    
    def edit_person_photo_from_dialog(self, window):
        """Edit selected person's face photo from database dialog"""
        selection = self.db_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a person to edit")
            return
        
        # Get the selected person info
        selected_index = selection[0]
        persons = self.database.get_persons()
        if selected_index >= len(persons):
            messagebox.showerror("Error", "Invalid selection")
            return
        
        person_id, person_name, num_samples, created_at = persons[selected_index]
        
        # Open file dialog to select new photo
        file_path = filedialog.askopenfilename(
            title="Select New Face Photo",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # Load and resize the image
            from PIL import Image
            img = Image.open(file_path)
            img = img.convert('RGB')
            img = img.resize((150, 150), Image.Resampling.LANCZOS)
            
            # Save to database
            conn = sqlite3.connect(self.database.db_path)
            cursor = conn.cursor()
            face_photo_blob = pickle.dumps(img)
            cursor.execute(
                "INSERT OR REPLACE INTO face_photos (person_id, face_photo) VALUES (?, ?)",
                (person_id, face_photo_blob)
            )
            conn.commit()
            conn.close()
            
            messagebox.showinfo("Success", f"Face photo updated for '{person_name}'")
            # Refresh the dialog
            self.refresh_database_dialog(window)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update face photo: {str(e)}")
    
    def refresh_database_dialog(self, window):
        """Refresh the database dialog"""
        window.destroy()
        self.create_database_dialog()
    
    def create_main_tab(self):
        """Create main identification tab"""
        main_frame = GlassFrame(self.notebook)
        self.notebook.add(main_frame, text="🔍 Identification")
        
        # Left panel - Video feed
        left_panel = GlassFrame(main_frame)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Video header
        video_header = tk.Frame(left_panel, bg=ModernTheme.COLORS['bg_secondary'], height=50)
        video_header.pack(fill='x', padx=10, pady=10)
        video_header.pack_propagate(False)
        
        title_label = ttk.Label(video_header, text="Live Video Feed", style='Title.TLabel')
        title_label.pack(side='left', pady=10)
        
        # Status indicators
        status_frame = tk.Frame(video_header, bg=ModernTheme.COLORS['bg_secondary'])
        status_frame.pack(side='right', pady=10)
        
        self.yolo_indicator = StatusIndicator(status_frame, status="inactive")
        self.yolo_indicator.pack(side='right', padx=(5, 0))
        ttk.Label(status_frame, text="YOLO", style='LiquidGlass.TLabel').pack(side='right')
        
        self.model_indicator = StatusIndicator(status_frame, status="inactive")
        self.model_indicator.pack(side='right', padx=(10, 5))
        self.model_label = ttk.Label(status_frame, text="Model", style='LiquidGlass.TLabel')
        self.model_label.pack(side='right')
        
        # Video display
        video_container = GlassFrame(left_panel)
        video_container.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        self.video_label = tk.Label(video_container, 
                                   text="Load models and start camera to begin",
                                   bg=ModernTheme.COLORS['bg_tertiary'],
                                   fg=ModernTheme.COLORS['text_secondary'],
                                   font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 12))
        self.video_label.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Right panel - Controls
        right_panel = GlassFrame(main_frame)
        right_panel.pack(side='right', fill='y', padx=(5, 0))
        right_panel.configure(width=350)
        
        self.create_model_controls(right_panel)
        self.create_camera_controls(right_panel)
        self.create_results_panel(right_panel)
        self.create_persons_panel(right_panel)
    
    def create_model_controls(self, parent):
        """Create model loading controls - Supporting both Excellent and Ultimate models"""
        model_frame = GlassFrame(parent)
        model_frame.pack(fill='x', padx=10, pady=10)
        
        # Header
        header = tk.Frame(model_frame, bg=ModernTheme.COLORS['bg_secondary'])
        header.pack(fill='x', padx=10, pady=(10, 5))
        
        ttk.Label(header, text="Model Configuration", style='Title.TLabel').pack(anchor='w')
        
        # Content
        content = tk.Frame(model_frame, bg=ModernTheme.COLORS['bg_secondary'])
        content.pack(fill='x', padx=10, pady=(0, 10))
        
        # Model type selector
        ttk.Label(content, text="Model Type:", style='LiquidGlass.TLabel').pack(anchor='w')
        self.model_type_selector = ttk.Combobox(content, state="readonly", width=35, style='LiquidGlass.TCombobox')
        self.model_type_selector['values'] = ["Excellent", "Ultimate", "TIMM Models"]
        self.model_type_selector.set("TIMM Models")  # Default to TIMM Models for efficientnet_retrained_final
        self.model_type_selector.pack(fill='x', pady=(2, 5))
        self.model_type_selector.bind('<<ComboboxSelected>>', self.on_model_type_changed)
        
        # Model selector
        ttk.Label(content, text="Model File:", style='LiquidGlass.TLabel').pack(anchor='w')
        self.model_selector = ttk.Combobox(content, state="readonly", width=35, style='LiquidGlass.TCombobox')
        self.model_selector.pack(fill='x', pady=(2, 5))
        
        # YOLO model selector
        ttk.Label(content, text="YOLO Detection Model:", style='LiquidGlass.TLabel').pack(anchor='w')
        self.yolo_selector = ttk.Combobox(content, state="readonly", width=35, style='LiquidGlass.TCombobox')
        self.yolo_selector.pack(fill='x', pady=(2, 10))
        
        # Load button
        self.load_button = ModernButton(content, text="Load Models", 
                                       command=self.load_models,
                                       width=150, height=35,
                                       bg_color=ModernTheme.COLORS['accent_primary'])
        self.load_button.pack(pady=5)
        
        # Status labels
        status_frame = tk.Frame(content, bg=ModernTheme.COLORS['bg_secondary'])
        status_frame.pack(fill='x', pady=(10, 0))
        
        self.yolo_status = ttk.Label(status_frame, text="YOLO: Not loaded", 
                                    style='LiquidGlass.TLabel')
        self.yolo_status.pack(anchor='w')
        
        self.model_status = ttk.Label(status_frame, text="Model: Not loaded", 
                                     style='LiquidGlass.TLabel')
        self.model_status.pack(anchor='w')
    
    def create_camera_controls(self, parent):
        """Create camera controls"""
        camera_frame = GlassFrame(parent)
        camera_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        # Header
        header = tk.Frame(camera_frame, bg=ModernTheme.COLORS['bg_secondary'])
        header.pack(fill='x', padx=10, pady=(10, 5))
        
        ttk.Label(header, text="Camera Controls", style='Title.TLabel').pack(anchor='w')
        
        # Content
        content = tk.Frame(camera_frame, bg=ModernTheme.COLORS['bg_secondary'])
        content.pack(fill='x', padx=10, pady=(0, 10))
        
        # Camera selector
        ttk.Label(content, text="Camera:", style='LiquidGlass.TLabel').pack(anchor='w')
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(content, textvariable=self.camera_var, 
                                        state="readonly", style='LiquidGlass.TCombobox')
        self.camera_combo.pack(fill='x', pady=(2, 10))
        
        # Control buttons
        button_frame = tk.Frame(content, bg=ModernTheme.COLORS['bg_secondary'])
        button_frame.pack(fill='x', pady=5)
        
        self.start_button = ModernButton(button_frame, text="Start Camera", 
                                        command=self.start_identification,
                                        width=120, height=35,
                                        bg_color=ModernTheme.COLORS['success'])
        self.start_button.pack(side='left', padx=(0, 5))
        
        self.stop_button = ModernButton(button_frame, text="Stop", 
                                       command=self.stop_camera,
                                       width=80, height=35,
                                       bg_color=ModernTheme.COLORS['error'])
        self.stop_button.pack(side='left')
        
        # FPS display
        self.fps_label = ttk.Label(content, text="FPS: 0", style='LiquidGlass.TLabel')
        self.fps_label.pack(anchor='w', pady=(10, 0))
    
    def create_results_panel(self, parent):
        """Create results display panel"""
        results_frame = GlassFrame(parent)
        results_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Header
        header = tk.Frame(results_frame, bg=ModernTheme.COLORS['bg_secondary'])
        header.pack(fill='x', padx=10, pady=(10, 5))
        
        ttk.Label(header, text="Identification Results", style='Title.TLabel').pack(anchor='w')
        
        # Results text area
        text_frame = tk.Frame(results_frame, bg=ModernTheme.COLORS['bg_secondary'])
        text_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        self.results_text = tk.Text(text_frame, height=8, width=40, 
                                   bg=ModernTheme.COLORS['bg_tertiary'],
                                   fg=ModernTheme.COLORS['text_primary'],
                                   font=('Consolas', 9),
                                   relief='flat', borderwidth=1)
        
        results_scrollbar = ttk.Scrollbar(text_frame, orient='vertical', 
                                         command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.pack(side='left', fill='both', expand=True)
        results_scrollbar.pack(side='right', fill='y')
    
    def create_persons_panel(self, parent):
        """Create enrolled persons panel"""
        persons_frame = GlassFrame(parent)
        persons_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        # Header
        header = tk.Frame(persons_frame, bg=ModernTheme.COLORS['bg_secondary'])
        header.pack(fill='x', padx=10, pady=(10, 5))
        
        title_frame = tk.Frame(header, bg=ModernTheme.COLORS['bg_secondary'])
        title_frame.pack(fill='x')
        
        ttk.Label(title_frame, text="Enrolled Persons", style='Title.TLabel').pack(side='left')
        
        # Refresh button
        refresh_btn = ModernButton(title_frame, text="↻", command=self.update_database,
                                  width=30, height=25,
                                  bg_color=ModernTheme.COLORS['accent_secondary'])
        refresh_btn.pack(side='right')
        
        # Persons list
        list_frame = tk.Frame(persons_frame, bg=ModernTheme.COLORS['bg_secondary'])
        list_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        self.persons_listbox = tk.Listbox(list_frame, height=6, 
                                         bg=ModernTheme.COLORS['bg_tertiary'],
                                         fg=ModernTheme.COLORS['text_primary'],
                                         font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 9),
                                         relief='flat', borderwidth=1,
                                         selectbackground=ModernTheme.COLORS['accent_primary'])
        
        persons_scrollbar = ttk.Scrollbar(list_frame, orient='vertical', 
                                         command=self.persons_listbox.yview)
        self.persons_listbox.configure(yscrollcommand=persons_scrollbar.set)
        
        self.persons_listbox.pack(side='left', fill='both', expand=True)
        persons_scrollbar.pack(side='right', fill='y')
    
    def create_enrollment_tab(self):
        """Create enrollment tab with sample preview"""
        enroll_frame = GlassFrame(self.notebook)
        self.notebook.add(enroll_frame, text="📝 Enrollment")
        
        # Left panel - Camera and preview
        left_panel = GlassFrame(enroll_frame)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Camera section
        camera_section = GlassFrame(left_panel)
        camera_section.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Header
        header = tk.Frame(camera_section, bg=ModernTheme.COLORS['bg_secondary'])
        header.pack(fill='x', padx=10, pady=(10, 5))
        
        ttk.Label(header, text="Enrollment Camera", style='Title.TLabel').pack(side='left')
        
        # Sample counter
        self.sample_counter = ttk.Label(header, text="Sample 0/5", style='LiquidGlass.TLabel')
        self.sample_counter.pack(side='right')
        
        # Video display for enrollment
        self.enroll_video_label = tk.Label(camera_section,
                                          text="Start enrollment to begin capturing samples",
                                          bg=ModernTheme.COLORS['bg_tertiary'],
                                          fg=ModernTheme.COLORS['text_secondary'],
                                          font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 12))
        self.enroll_video_label.pack(expand=True, fill='both', padx=10, pady=(0, 10))
        
        # Sample preview section
        preview_section = GlassFrame(left_panel)
        preview_section.pack(fill='x', padx=10, pady=(0, 10))
        
        preview_header = tk.Frame(preview_section, bg=ModernTheme.COLORS['bg_secondary'])
        preview_header.pack(fill='x', padx=10, pady=(10, 5))
        
        ttk.Label(preview_header, text="Sample Preview", style='Title.TLabel').pack(anchor='w')
        
        # Preview grid
        self.preview_frame = tk.Frame(preview_section, bg=ModernTheme.COLORS['bg_secondary'], 
                                     relief='raised', borderwidth=2)
        self.preview_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Add a placeholder label to make the frame visible
        self.preview_placeholder = tk.Label(self.preview_frame, 
                                           text="Samples will appear here as they are captured",
                                           bg=ModernTheme.COLORS['bg_tertiary'],
                                           fg=ModernTheme.COLORS['text_secondary'],
                                           font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 10))
        self.preview_placeholder.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Right panel - Controls
        right_panel = GlassFrame(enroll_frame)
        right_panel.pack(side='right', fill='y', padx=(5, 0))
        right_panel.configure(width=350)
        
        self.create_enrollment_controls(right_panel)
    
    def create_enrollment_controls(self, parent):
        """Create enrollment controls"""
        # Person info
        info_frame = GlassFrame(parent)
        info_frame.pack(fill='x', padx=10, pady=10)
        
        header = tk.Frame(info_frame, bg=ModernTheme.COLORS['bg_secondary'])
        header.pack(fill='x', padx=10, pady=(10, 5))
        
        ttk.Label(header, text="Person Information", style='Title.TLabel').pack(anchor='w')
        
        content = tk.Frame(info_frame, bg=ModernTheme.COLORS['bg_secondary'])
        content.pack(fill='x', padx=10, pady=(0, 10))
        
        ttk.Label(content, text="Full Name:", style='LiquidGlass.TLabel').pack(anchor='w')
        self.person_name_var = tk.StringVar()
        self.person_name_entry = ttk.Entry(content, textvariable=self.person_name_var,
                                          style='LiquidGlass.TEntry', font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 11))
        self.person_name_entry.pack(fill='x', pady=(2, 10))
        
        # Enrollment controls
        controls_frame = GlassFrame(parent)
        controls_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        header = tk.Frame(controls_frame, bg=ModernTheme.COLORS['bg_secondary'])
        header.pack(fill='x', padx=10, pady=(10, 5))
        
        ttk.Label(header, text="Enrollment Controls", style='Title.TLabel').pack(anchor='w')
        
        content = tk.Frame(controls_frame, bg=ModernTheme.COLORS['bg_secondary'])
        content.pack(fill='x', padx=10, pady=(0, 10))
        
        # Start enrollment button
        self.start_enroll_button = ModernButton(content, text="Start Enrollment",
                                               command=self.start_enrollment,
                                               width=150, height=35,
                                               bg_color=ModernTheme.COLORS['success'])
        self.start_enroll_button.pack(pady=5)
        
        # Stop enrollment button
        self.stop_enroll_button = ModernButton(content, text="Stop Enrollment",
                                              command=self.stop_enrollment,
                                              width=150, height=35,
                                              bg_color=ModernTheme.COLORS['error'])
        self.stop_enroll_button.pack(pady=5)
        
        # Instructions
        instructions_frame = GlassFrame(parent)
        instructions_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        header = tk.Frame(instructions_frame, bg=ModernTheme.COLORS['bg_secondary'])
        header.pack(fill='x', padx=10, pady=(10, 5))
        
        ttk.Label(header, text="Instructions", style='Title.TLabel').pack(anchor='w')
        
        self.instructions_text = tk.Text(instructions_frame, height=10, width=40,
                                        bg=ModernTheme.COLORS['bg_tertiary'],
                                        fg=ModernTheme.COLORS['text_primary'],
                                        font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 9),
                                        relief='flat', borderwidth=1,
                                        wrap=tk.WORD)
        self.instructions_text.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Default instructions
        instructions = """Welcome to Enrollment!

1. Enter your full name above
2. Click 'Start Enrollment' 
3. Position your ear in the green target box
4. Hold still when prompted for each sample
5. Move slightly between samples for variety
6. Review samples before confirming
7. Click 'Save Enrollment' to complete

Tips:
• Good lighting improves quality
• Keep ear clearly visible
• Avoid hair covering the ear
• Stay within the target area"""
        
        self.instructions_text.insert('1.0', instructions)
        self.instructions_text.configure(state='disabled')
        
        # Sample actions
        actions_frame = GlassFrame(parent)
        actions_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        header = tk.Frame(actions_frame, bg=ModernTheme.COLORS['bg_secondary'])
        header.pack(fill='x', padx=10, pady=(10, 5))
        
        ttk.Label(header, text="Sample Actions", style='Title.TLabel').pack(anchor='w')
        
        content = tk.Frame(actions_frame, bg=ModernTheme.COLORS['bg_secondary'])
        content.pack(fill='x', padx=10, pady=(0, 10))
        
        button_frame = tk.Frame(content, bg=ModernTheme.COLORS['bg_secondary'])
        button_frame.pack(fill='x', pady=5)
        
        self.accept_button = ModernButton(button_frame, text="Accept Samples",
                                         command=self.accept_samples,
                                         width=120, height=30,
                                         bg_color=ModernTheme.COLORS['success'])
        self.accept_button.pack(side='left', padx=(0, 5))
        
        self.reject_button = ModernButton(button_frame, text="Reject",
                                         command=self.reject_samples,
                                         width=80, height=30,
                                         bg_color=ModernTheme.COLORS['warning'])
        self.reject_button.pack(side='left')
        
        self.clear_samples_button = ModernButton(content, text="Clear All Samples",
                                                command=self.clear_enrollment_samples,
                                                width=150, height=30,
                                                bg_color=ModernTheme.COLORS['error'])
        self.clear_samples_button.pack(pady=(5, 0))
    
    def create_database_tab(self):
        """Create enhanced database management tab with samples, faces, and editing"""
        db_frame = GlassFrame(self.notebook)
        self.notebook.add(db_frame, text="🗃️ Database")
        
        # Main container
        main_container = tk.Frame(db_frame, bg=ModernTheme.COLORS['bg_primary'])
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Header
        header_frame = tk.Frame(main_container, bg=ModernTheme.COLORS['bg_secondary'], height=60)
        header_frame.pack(fill='x', pady=(0, 10))
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, text="🗃️ Database Management", 
                font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 16, 'bold'),
                bg=ModernTheme.COLORS['bg_secondary'],
                fg=ModernTheme.COLORS['text_primary']).pack(side='left', padx=20, pady=15)
        
        # Person count
        self.person_count_label = tk.Label(header_frame, text="0 persons", 
                                         font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 12),
                                         bg=ModernTheme.COLORS['bg_secondary'],
                                         fg=ModernTheme.COLORS['text_secondary'])
        self.person_count_label.pack(side='right', padx=20, pady=15)
        
        # Content area with scrollable person cards
        content_frame = tk.Frame(main_container, bg=ModernTheme.COLORS['bg_primary'])
        content_frame.pack(fill='both', expand=True)
        
        # Scrollable canvas for person cards
        self.db_canvas = tk.Canvas(content_frame, bg=ModernTheme.COLORS['bg_primary'], highlightthickness=0)
        self.db_scrollbar = tk.Scrollbar(content_frame, orient="vertical", command=self.db_canvas.yview)
        self.db_scrollable_frame = tk.Frame(self.db_canvas, bg=ModernTheme.COLORS['bg_primary'])
        
        self.db_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.db_canvas.configure(scrollregion=self.db_canvas.bbox("all"))
        )
        
        self.db_canvas.create_window((0, 0), window=self.db_scrollable_frame, anchor="nw")
        self.db_canvas.configure(yscrollcommand=self.db_scrollbar.set)
        
        self.db_canvas.pack(side="left", fill="both", expand=True)
        self.db_scrollbar.pack(side="right", fill="y")
        
        # Store for person cards
        self.person_cards = {}
    
    def create_person_card(self, person_id, person_name, num_samples, created_at):
        """Create a visual card for a person with face, samples, and edit options"""
        card_frame = tk.Frame(self.db_scrollable_frame, bg=ModernTheme.COLORS['bg_secondary'], 
                             relief='raised', borderwidth=1)
        card_frame.pack(fill='x', padx=10, pady=5)
        
        # Main card content
        content_frame = tk.Frame(card_frame, bg=ModernTheme.COLORS['bg_secondary'])
        content_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Top row - Face photo and basic info
        top_row = tk.Frame(content_frame, bg=ModernTheme.COLORS['bg_secondary'])
        top_row.pack(fill='x', pady=(0, 10))
        
        # Face photo
        face_frame = tk.Frame(top_row, bg=ModernTheme.COLORS['bg_secondary'])
        face_frame.pack(side='left', padx=(0, 15))
        
        # Get and display face photo
        face_photo = self.database.get_person_face_photo(person_id)
        if face_photo is not None:
            try:
                face_display = cv2.resize(face_photo, (80, 80))
                face_rgb = cv2.cvtColor(face_display, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)
                face_tk = ImageTk.PhotoImage(face_pil)
                
                face_label = tk.Label(face_frame, image=face_tk, 
                                     bg=ModernTheme.COLORS['bg_secondary'])
                face_label.image = face_tk  # Keep reference
                face_label.pack()
            except Exception as e:
                print(f"❌ Error displaying face: {e}")
                self.create_placeholder_face(face_frame)
        else:
            self.create_placeholder_face(face_frame)
        
        # Person info
        info_frame = tk.Frame(top_row, bg=ModernTheme.COLORS['bg_secondary'])
        info_frame.pack(side='left', fill='x', expand=True)
        
        # Editable name
        name_frame = tk.Frame(info_frame, bg=ModernTheme.COLORS['bg_secondary'])
        name_frame.pack(fill='x', pady=(0, 5))
        
        tk.Label(name_frame, text="Name:", font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 10, 'bold'),
                bg=ModernTheme.COLORS['bg_secondary'],
                fg=ModernTheme.COLORS['text_secondary']).pack(side='left')
        
        name_var = tk.StringVar(value=person_name)
        name_entry = tk.Entry(name_frame, textvariable=name_var,
                             font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 12), width=20,
                                            bg=ModernTheme.COLORS['bg_tertiary'],
                                            fg=ModernTheme.COLORS['text_primary'],
                             relief='flat', borderwidth=1)
        name_entry.pack(side='left', padx=(10, 5))
        
        # Save name button
        save_name_btn = ModernButton(name_frame, text="💾", 
                                   command=lambda: self.save_person_name(person_id, name_var.get()),
                                   width=30, height=25,
                                   bg_color=ModernTheme.COLORS['success'])
        save_name_btn.pack(side='left', padx=(5, 0))
        
        # Person stats
        stats_text = f"ID: {person_id}\nSamples: {num_samples}\nCreated: {created_at[:10] if created_at else 'Unknown'}"
        tk.Label(info_frame, text=stats_text, font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 9),
                bg=ModernTheme.COLORS['bg_secondary'],
                fg=ModernTheme.COLORS['text_secondary'],
                justify='left').pack(anchor='w')
        
        # Action buttons
        action_frame = tk.Frame(info_frame, bg=ModernTheme.COLORS['bg_secondary'])
        action_frame.pack(fill='x', pady=(10, 0))
        
        # View samples button
        view_btn = ModernButton(action_frame, text="👁️ View Samples", 
                               command=lambda: self.view_person_samples(person_id),
                               width=120, height=30,
                               bg_color=ModernTheme.COLORS['accent_primary'])
        view_btn.pack(side='left', padx=(0, 5))
        
        # Update face button
        face_btn = ModernButton(action_frame, text="📸 Update Face", 
                               command=lambda: self.update_person_face(person_id),
                               width=120, height=30,
                               bg_color=ModernTheme.COLORS['accent_secondary'])
        face_btn.pack(side='left', padx=(0, 5))
        
        # Delete person button
        delete_btn = ModernButton(action_frame, text="🗑️ Delete", 
                                 command=lambda: self.delete_person_confirm(person_id),
                                 width=80, height=30,
                                 bg_color=ModernTheme.COLORS['error'])
        delete_btn.pack(side='right')
        
        # Store card reference
        self.person_cards[person_id] = card_frame
        
        return card_frame
    
    def create_placeholder_face(self, parent):
        """Create placeholder for missing face photo"""
        placeholder = tk.Label(parent, text="👤", font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 40),
                              bg=ModernTheme.COLORS['bg_tertiary'],
                              fg=ModernTheme.COLORS['text_secondary'],
                              width=3, height=2, relief='solid', borderwidth=1)
        placeholder.pack()
    
    def save_person_name(self, person_id, new_name):
        """Save updated person name"""
        try:
            conn = sqlite3.connect(self.database.db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE persons SET name = ? WHERE id = ?", (new_name, person_id))
            conn.commit()
            conn.close()
            
            # Update status
            if hasattr(self, 'status_label'):
                self.status_label.configure(text=f"Updated name for {person_id}", 
                                          fg=ModernTheme.COLORS['success'])
            print(f"✅ Updated name for {person_id}: {new_name}")
            
        except Exception as e:
            print(f"❌ Error updating name: {e}")
            if hasattr(self, 'status_label'):
                self.status_label.configure(text="Error updating name", 
                                          fg=ModernTheme.COLORS['error'])
    
    def view_person_samples(self, person_id):
        """Show all samples for a person in a popup window"""
        samples = self.database.get_person_samples(person_id)
        if not samples:
            return
        
        # Create samples viewer window
        samples_window = tk.Toplevel(self.root)
        samples_window.title(f"Samples for {person_id}")
        samples_window.geometry("800x600")
        samples_window.configure(bg=ModernTheme.COLORS['bg_primary'])
        
        # Header
        header_frame = tk.Frame(samples_window, bg=ModernTheme.COLORS['bg_secondary'], height=50)
        header_frame.pack(fill='x', pady=(0, 10))
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, text=f"👁️ Samples for {person_id}", 
                font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 14, 'bold'),
                bg=ModernTheme.COLORS['bg_secondary'],
                fg=ModernTheme.COLORS['text_primary']).pack(pady=15)
        
        # Samples grid
        samples_frame = tk.Frame(samples_window, bg=ModernTheme.COLORS['bg_primary'])
        samples_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        # Create grid of samples
        cols = 4
        for i, sample in enumerate(samples):
            row = i // cols
            col = i % cols
            
            # Sample card
            sample_card = tk.Frame(samples_frame, bg=ModernTheme.COLORS['bg_secondary'],
                                  relief='raised', borderwidth=1)
            sample_card.grid(row=row, col=col, padx=5, pady=5, sticky='nsew')
            
            # Sample image (if available)
            if sample.get('image_path') and os.path.exists(sample['image_path']):
                try:
                    img = cv2.imread(sample['image_path'])
                    img_resized = cv2.resize(img, (120, 120))
                    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    img_tk = ImageTk.PhotoImage(img_pil)
                    
                    img_label = tk.Label(sample_card, image=img_tk)
                    img_label.image = img_tk
                    img_label.pack(pady=(10, 5))
                except:
                    # Placeholder if image can't be loaded
                    tk.Label(sample_card, text="👂", font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 40),
                            bg=ModernTheme.COLORS['bg_tertiary'],
                            fg=ModernTheme.COLORS['text_secondary']).pack(pady=(10, 5))
            else:
                # Placeholder if no image
                tk.Label(sample_card, text="👂", font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 40),
                        bg=ModernTheme.COLORS['bg_tertiary'],
                        fg=ModernTheme.COLORS['text_secondary']).pack(pady=(10, 5))
            
            # Sample info
            info_text = f"Confidence: {sample.get('confidence', 0.0):.2f}\n"
            info_text += f"Model: {sample.get('model_type', 'Unknown')}\n"
            info_text += f"Dim: {sample.get('feature_dim', 'Unknown')}"
            
            tk.Label(sample_card, text=info_text, font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 8),
                    bg=ModernTheme.COLORS['bg_secondary'],
                    fg=ModernTheme.COLORS['text_secondary']).pack(pady=(0, 10))
        
        # Configure grid weights
        for i in range(cols):
            samples_frame.columnconfigure(i, weight=1)
    
    def update_person_face(self, person_id):
        """Update face photo for a person"""
        if not self.is_running or not self.camera:
            if hasattr(self, 'status_label'):
                self.status_label.configure(text="Start camera first to capture face", 
                                          fg=ModernTheme.COLORS['error'])
            return
        
        try:
            # Capture current frame
            ret, frame = self.camera.read()
            if ret:
                # Resize and store face photo
                face_photo = cv2.resize(frame, (120, 120))
                
                # Save to database
                conn = sqlite3.connect(self.database.db_path)
                cursor = conn.cursor()
                face_photo_blob = pickle.dumps(face_photo)
                cursor.execute(
                    "INSERT OR REPLACE INTO face_photos (person_id, face_photo) VALUES (?, ?)",
                    (person_id, face_photo_blob)
                )
                conn.commit()
                conn.close()
                
                # Update status
                if hasattr(self, 'status_label'):
                    self.status_label.configure(text=f"Updated face for {person_id}", 
                                              fg=ModernTheme.COLORS['success'])
                
                # Refresh database view
                self.refresh_database_view()
                
        except Exception as e:
            print(f"❌ Error updating face: {e}")
            if hasattr(self, 'status_label'):
                self.status_label.configure(text="Error updating face", 
                                          fg=ModernTheme.COLORS['error'])
    
    def delete_person_confirm(self, person_id):
        """Confirm and delete a person"""
        # Simple confirmation (could be enhanced with a proper dialog)
        import tkinter.messagebox as msgbox
        
        if msgbox.askyesno("Confirm Delete", f"Are you sure you want to delete {person_id} and all their samples?"):
            success = self.database.delete_person(person_id)
            if success:
                self.refresh_database_view()
                if hasattr(self, 'status_label'):
                    self.status_label.configure(text=f"Deleted {person_id}", 
                                              fg=ModernTheme.COLORS['success'])
    
    def refresh_database_view(self):
        """Refresh the database view with current data"""
        # Clear existing cards
        for card in self.person_cards.values():
            card.destroy()
        self.person_cards.clear()
        
        # Get updated person list
        persons = self.database.get_persons()
        
        # Create new cards
        for person_id, name, num_samples, created_at in persons:
            self.create_person_card(person_id, name, num_samples, created_at)
        
        # Update person count
        self.person_count_label.configure(text=f"{len(persons)} persons")
        
        # Bind selection event
        self.db_persons_listbox.bind('<<ListboxSelect>>', self.on_person_select)
        
        # Right panel - Actions and details
        right_panel = GlassFrame(db_frame)
        right_panel.pack(side='right', fill='y', padx=(5, 0))
        right_panel.configure(width=350)
        
        self.create_database_controls(right_panel)
    
    def create_database_controls(self, parent):
        """Create database management controls"""
        # Person details
        details_frame = GlassFrame(parent)
        details_frame.pack(fill='x', padx=10, pady=10)
        
        header = tk.Frame(details_frame, bg=ModernTheme.COLORS['bg_secondary'])
        header.pack(fill='x', padx=10, pady=(10, 5))
        
        ttk.Label(header, text="Person Details", style='Title.TLabel').pack(anchor='w')
        
        content = tk.Frame(details_frame, bg=ModernTheme.COLORS['bg_secondary'])
        content.pack(fill='x', padx=10, pady=(0, 10))
        
        # Details display
        self.person_details_text = tk.Text(content, height=8, width=40,
                                          bg=ModernTheme.COLORS['bg_tertiary'],
                                          fg=ModernTheme.COLORS['text_primary'],
                                          font=('Consolas', 9),
                                          relief='flat', borderwidth=1,
                                          state='disabled')
        self.person_details_text.pack(fill='x', pady=(0, 10))
        
        # Actions
        actions_frame = GlassFrame(parent)
        actions_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        header = tk.Frame(actions_frame, bg=ModernTheme.COLORS['bg_secondary'])
        header.pack(fill='x', padx=10, pady=(10, 5))
        
        ttk.Label(header, text="Database Actions", style='Title.TLabel').pack(anchor='w')
        
        content = tk.Frame(actions_frame, bg=ModernTheme.COLORS['bg_secondary'])
        content.pack(fill='x', padx=10, pady=(0, 10))
        
        # Individual actions
        self.view_samples_button = ModernButton(content, text="View Samples",
                                               command=self.view_person_samples,
                                               width=150, height=30,
                                               bg_color=ModernTheme.COLORS['accent_primary'])
        self.view_samples_button.pack(pady=2)
        
        self.delete_person_button = ModernButton(content, text="Delete Person",
                                                command=self.delete_selected_person,
                                                width=150, height=30,
                                                bg_color=ModernTheme.COLORS['error'])
        self.delete_person_button.pack(pady=2)
        
        # Separator
        separator = tk.Frame(content, height=2, bg=ModernTheme.COLORS['border'])
        separator.pack(fill='x', pady=10)
        
        # Bulk actions
        self.export_button = ModernButton(content, text="Export Database",
                                         command=self.export_database,
                                         width=150, height=30,
                                         bg_color=ModernTheme.COLORS['accent_secondary'])
        self.export_button.pack(pady=2)
        
        self.import_button = ModernButton(content, text="Import Database",
                                         command=self.import_database,
                                         width=150, height=30,
                                         bg_color=ModernTheme.COLORS['accent_tertiary'])
        self.import_button.pack(pady=2)
        
        self.clear_all_button = ModernButton(content, text="Clear All Data",
                                            command=self.clear_all_database,
                                            width=150, height=30,
                                            bg_color=ModernTheme.COLORS['error'])
        self.clear_all_button.pack(pady=2)
        
        # Statistics
        stats_frame = GlassFrame(parent)
        stats_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        header = tk.Frame(stats_frame, bg=ModernTheme.COLORS['bg_secondary'])
        header.pack(fill='x', padx=10, pady=(10, 5))
        
        ttk.Label(header, text="Database Statistics", style='Title.TLabel').pack(anchor='w')
        
        self.stats_text = tk.Text(stats_frame, height=10, width=40,
                                 bg=ModernTheme.COLORS['bg_tertiary'],
                                 fg=ModernTheme.COLORS['text_primary'],
                                 font=('Consolas', 9),
                                 relief='flat', borderwidth=1,
                                 state='disabled')
        self.stats_text.pack(fill='both', expand=True, padx=10, pady=(0, 10))
    
    def create_options_tab(self):
        """Create options/settings tab"""
        options_frame = GlassFrame(self.notebook)
        self.notebook.add(options_frame, text="⚙️ Options")
        
        # Create scrollable frame
        canvas = tk.Canvas(options_frame, bg=ModernTheme.COLORS['bg_secondary'])
        scrollbar = ttk.Scrollbar(options_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = GlassFrame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Quality Settings
        self.create_quality_settings(scrollable_frame)
        
        # Detection Settings
        self.create_detection_settings(scrollable_frame)
        
        # Camera Settings
        self.create_camera_settings(scrollable_frame)
        
        # Advanced Settings
        self.create_advanced_settings(scrollable_frame)
        
        # Theme Settings
        self.create_theme_settings(scrollable_frame)
    
    def create_quality_settings(self, parent):
        """Create quality control settings"""
        quality_frame = GlassFrame(parent)
        quality_frame.pack(fill='x', padx=10, pady=10)
        
        header = tk.Frame(quality_frame, bg=ModernTheme.COLORS['bg_secondary'])
        header.pack(fill='x', padx=10, pady=(10, 5))
        
        ttk.Label(header, text="Quality Control", style='Title.TLabel').pack(anchor='w')
        
        content = tk.Frame(quality_frame, bg=ModernTheme.COLORS['bg_secondary'])
        content.pack(fill='x', padx=10, pady=(0, 10))
        
        # Minimum ear size
        size_frame = tk.Frame(content, bg=ModernTheme.COLORS['bg_secondary'])
        size_frame.pack(fill='x', pady=5)
        
        ttk.Label(size_frame, text="Minimum Ear Size:", style='LiquidGlass.TLabel').pack(side='left')
        self.min_size_var = tk.IntVar(value=self.min_ear_size)
        self.min_size_scale = tk.Scale(size_frame, from_=50, to=200, 
                                      variable=self.min_size_var, orient='horizontal',
                                      bg=ModernTheme.COLORS['bg_secondary'],
                                      fg=ModernTheme.COLORS['text_primary'],
                                      highlightthickness=0,
                                      command=self.update_quality_settings)
        self.min_size_scale.pack(side='left', fill='x', expand=True, padx=(10, 5))
        
        self.min_size_label = ttk.Label(size_frame, text=f"{self.min_ear_size}px", 
                                       style='LiquidGlass.TLabel')
        self.min_size_label.pack(side='right')
        
        # Target ear size
        target_frame = tk.Frame(content, bg=ModernTheme.COLORS['bg_secondary'])
        target_frame.pack(fill='x', pady=5)
        
        ttk.Label(target_frame, text="Target Ear Size:", style='LiquidGlass.TLabel').pack(side='left')
        self.target_size_var = tk.IntVar(value=self.target_ear_size)
        self.target_size_scale = tk.Scale(target_frame, from_=100, to=300,
                                         variable=self.target_size_var, orient='horizontal',
                                         bg=ModernTheme.COLORS['bg_secondary'],
                                         fg=ModernTheme.COLORS['text_primary'],
                                         highlightthickness=0,
                                         command=self.update_quality_settings)
        self.target_size_scale.pack(side='left', fill='x', expand=True, padx=(10, 5))
        
        self.target_size_label = ttk.Label(target_frame, text=f"{self.target_ear_size}px",
                                          style='LiquidGlass.TLabel')
        self.target_size_label.pack(side='right')
        
        # Show guidelines
        self.guidelines_var = tk.BooleanVar(value=self.show_guidelines)
        guidelines_check = tk.Checkbutton(content, text="Show Visual Guidelines",
                                         variable=self.guidelines_var,
                                         bg=ModernTheme.COLORS['bg_secondary'],
                                         fg=ModernTheme.COLORS['text_primary'],
                                         selectcolor=ModernTheme.COLORS['bg_tertiary'],
                                         activebackground=ModernTheme.COLORS['bg_secondary'],
                                         command=self.update_quality_settings)
        guidelines_check.pack(anchor='w', pady=5)
    
    def create_detection_settings(self, parent):
        """Create detection settings"""
        detection_frame = GlassFrame(parent)
        detection_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        header = tk.Frame(detection_frame, bg=ModernTheme.COLORS['bg_secondary'])
        header.pack(fill='x', padx=10, pady=(10, 5))
        
        ttk.Label(header, text="Detection Settings", style='Title.TLabel').pack(anchor='w')
        
        content = tk.Frame(detection_frame, bg=ModernTheme.COLORS['bg_secondary'])
        content.pack(fill='x', padx=10, pady=(0, 10))
        
        # Confidence threshold
        conf_frame = tk.Frame(content, bg=ModernTheme.COLORS['bg_secondary'])
        conf_frame.pack(fill='x', pady=5)
        
        ttk.Label(conf_frame, text="Confidence Threshold:", style='LiquidGlass.TLabel').pack(side='left')
        self.confidence_var = tk.DoubleVar(value=0.90)
        self.confidence_scale = tk.Scale(conf_frame, from_=0.3, to=0.95, resolution=0.05,
                                        variable=self.confidence_var, orient='horizontal',
                                        bg=ModernTheme.COLORS['bg_secondary'],
                                        fg=ModernTheme.COLORS['text_primary'],
                                        highlightthickness=0,
                                        command=self.update_detection_settings)
        self.confidence_scale.pack(side='left', fill='x', expand=True, padx=(10, 5))
        
        self.confidence_label = ttk.Label(conf_frame, text="0.90",
                                         style='LiquidGlass.TLabel')
        self.confidence_label.pack(side='right')
        
        # Match margin (best vs second-best)
        margin_frame = tk.Frame(content, bg=ModernTheme.COLORS['bg_secondary'])
        margin_frame.pack(fill='x', pady=5)
        
        ttk.Label(margin_frame, text="Match Margin:", style='LiquidGlass.TLabel').pack(side='left')
        self.match_margin_var = tk.DoubleVar(value=self.match_margin)
        self.match_margin_scale = tk.Scale(margin_frame, from_=0.00, to=0.30, resolution=0.01,
                                          variable=self.match_margin_var, orient='horizontal',
                                          bg=ModernTheme.COLORS['bg_secondary'],
                                          fg=ModernTheme.COLORS['text_primary'],
                                          highlightthickness=0,
                                          command=self.update_detection_settings)
        self.match_margin_scale.pack(side='left', fill='x', expand=True, padx=(10, 5))
        
        self.match_margin_label = ttk.Label(margin_frame, text=f"{self.match_margin:.2f}",
                                           style='LiquidGlass.TLabel')
        self.match_margin_label.pack(side='right')
        
        # Max samples for enrollment
        samples_frame = tk.Frame(content, bg=ModernTheme.COLORS['bg_secondary'])
        samples_frame.pack(fill='x', pady=5)
        
        ttk.Label(samples_frame, text="Max Enrollment Samples:", style='LiquidGlass.TLabel').pack(side='left')
        self.max_samples_var = tk.IntVar(value=self.max_samples)
        self.max_samples_scale = tk.Scale(samples_frame, from_=3, to=10,
                                         variable=self.max_samples_var, orient='horizontal',
                                         bg=ModernTheme.COLORS['bg_secondary'],
                                         fg=ModernTheme.COLORS['text_primary'],
                                         highlightthickness=0,
                                         command=self.update_detection_settings)
        self.max_samples_scale.pack(side='left', fill='x', expand=True, padx=(10, 5))
        
        self.max_samples_label = ttk.Label(samples_frame, text=f"{self.max_samples}",
                                          style='LiquidGlass.TLabel')
        self.max_samples_label.pack(side='right')
    
    def create_camera_settings(self, parent):
        """Create camera settings"""
        camera_frame = GlassFrame(parent)
        camera_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        header = tk.Frame(camera_frame, bg=ModernTheme.COLORS['bg_secondary'])
        header.pack(fill='x', padx=10, pady=(10, 5))
        
        ttk.Label(header, text="Camera Settings", style='Title.TLabel').pack(anchor='w')
        
        content = tk.Frame(camera_frame, bg=ModernTheme.COLORS['bg_secondary'])
        content.pack(fill='x', padx=10, pady=(0, 10))
        
        # Camera resolution
        res_frame = tk.Frame(content, bg=ModernTheme.COLORS['bg_secondary'])
        res_frame.pack(fill='x', pady=5)
        
        ttk.Label(res_frame, text="Camera Resolution:", style='LiquidGlass.TLabel').pack(side='left')
        self.resolution_var = tk.StringVar(value="640x480")
        resolution_combo = ttk.Combobox(res_frame, textvariable=self.resolution_var,
                                       values=["320x240", "640x480", "800x600", "1024x768", "1280x720"],
                                       state="readonly", style='LiquidGlass.TCombobox')
        resolution_combo.pack(side='right', padx=(10, 0))
        
        # Auto-exposure
        self.auto_exposure_var = tk.BooleanVar(value=True)
        auto_exp_check = tk.Checkbutton(content, text="Auto Exposure",
                                       variable=self.auto_exposure_var,
                                       bg=ModernTheme.COLORS['bg_secondary'],
                                       fg=ModernTheme.COLORS['text_primary'],
                                       selectcolor=ModernTheme.COLORS['bg_tertiary'],
                                       activebackground=ModernTheme.COLORS['bg_secondary'])
        auto_exp_check.pack(anchor='w', pady=5)
    
    def create_advanced_settings(self, parent):
        """Create advanced settings"""
        advanced_frame = GlassFrame(parent)
        advanced_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        header = tk.Frame(advanced_frame, bg=ModernTheme.COLORS['bg_secondary'])
        header.pack(fill='x', padx=10, pady=(10, 5))
        
        ttk.Label(header, text="Advanced Settings", style='Title.TLabel').pack(anchor='w')
        
        content = tk.Frame(advanced_frame, bg=ModernTheme.COLORS['bg_secondary'])
        content.pack(fill='x', padx=10, pady=(0, 10))
        
        # Debug mode
        self.debug_mode_var = tk.BooleanVar(value=False)
        debug_check = tk.Checkbutton(content, text="Debug Mode (Verbose Logging)",
                                    variable=self.debug_mode_var,
                                    bg=ModernTheme.COLORS['bg_secondary'],
                                    fg=ModernTheme.COLORS['text_primary'],
                                    selectcolor=ModernTheme.COLORS['bg_tertiary'],
                                    activebackground=ModernTheme.COLORS['bg_secondary'])
        debug_check.pack(anchor='w', pady=2)
        
        # Save samples
        self.save_samples_var = tk.BooleanVar(value=True)
        save_check = tk.Checkbutton(content, text="Save Enrollment Samples Locally",
                                   variable=self.save_samples_var,
                                   bg=ModernTheme.COLORS['bg_secondary'],
                                   fg=ModernTheme.COLORS['text_primary'],
                                   selectcolor=ModernTheme.COLORS['bg_tertiary'],
                                   activebackground=ModernTheme.COLORS['bg_secondary'])
        save_check.pack(anchor='w', pady=2)
        
        # GPU acceleration
        self.use_gpu_var = tk.BooleanVar(value=torch.cuda.is_available())
        gpu_check = tk.Checkbutton(content, text="Use GPU Acceleration (if available)",
                                  variable=self.use_gpu_var,
                                  bg=ModernTheme.COLORS['bg_secondary'],
                                  fg=ModernTheme.COLORS['text_primary'],
                                  selectcolor=ModernTheme.COLORS['bg_tertiary'],
                                  activebackground=ModernTheme.COLORS['bg_secondary'])
        gpu_check.pack(anchor='w', pady=2)
        
        if not torch.cuda.is_available():
            gpu_check.configure(state='disabled')
    
    def create_theme_settings(self, parent):
        """Create theme settings"""
        theme_frame = GlassFrame(parent)
        theme_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        header = tk.Frame(theme_frame, bg=ModernTheme.COLORS['bg_secondary'])
        header.pack(fill='x', padx=10, pady=(10, 5))
        
        ttk.Label(header, text="Theme Settings", style='Title.TLabel').pack(anchor='w')
        
        content = tk.Frame(theme_frame, bg=ModernTheme.COLORS['bg_secondary'])
        content.pack(fill='x', padx=10, pady=(0, 10))
        
        # Theme selector (for future expansion)
        theme_frame_inner = tk.Frame(content, bg=ModernTheme.COLORS['bg_secondary'])
        theme_frame_inner.pack(fill='x', pady=5)
        
        ttk.Label(theme_frame_inner, text="Theme:", style='LiquidGlass.TLabel').pack(side='left')
        self.theme_var = tk.StringVar(value="Liquid Glass")
        theme_combo = ttk.Combobox(theme_frame_inner, textvariable=self.theme_var,
                                  values=["Liquid Glass", "Dark Mode", "Light Mode"],
                                  state="readonly", style='LiquidGlass.TCombobox')
        theme_combo.pack(side='right', padx=(10, 0))
        theme_combo.configure(state='disabled')  # Only Liquid Glass for now
        
        # Animations
        self.animations_var = tk.BooleanVar(value=True)
        anim_check = tk.Checkbutton(content, text="Enable Animations",
                                   variable=self.animations_var,
                                   bg=ModernTheme.COLORS['bg_secondary'],
                                   fg=ModernTheme.COLORS['text_primary'],
                                   selectcolor=ModernTheme.COLORS['bg_tertiary'],
                                   activebackground=ModernTheme.COLORS['bg_secondary'])
        anim_check.pack(anchor='w', pady=5)
    
    # Core functionality methods
    def load_all_models(self):
        """Load available models (Excellent, Ultimate, and TIMM)"""
        # Initialize model storage
        self.excellent_models = []
        self.ultimate_models = []
        self.timm_models = []
        
        # Find excellent models
        excellent_patterns = [
            "excellent_ear_model_best.pth",
            "excellent_ear_model_*.pth"
        ]
        
        for pattern in excellent_patterns:
            self.excellent_models.extend(Path(".").glob(pattern))
        
        # Find ultimate models
        ultimate_patterns = [
            "efficientnet_b4_ultimate_best.pth",
            "efficientnet_b4_ultimate_final.pth",
            "efficientnet_b4_ultimate_epoch_*.pth"
        ]
        
        for pattern in ultimate_patterns:
            self.ultimate_models.extend(Path(".").glob(pattern))
        
        # Get TIMM models (both local and remote)
        self.timm_models = self.load_timm_models()
        
        # Sort TIMM models by performance (best first), but prioritize retrained model
        self.timm_models.sort(key=lambda x: (
            0 if x["name"] == "efficientnet_b5" else 1,  # Prioritize efficientnet_b5 (retrained model)
            -x["accuracy"]  # Then by accuracy (negative for descending)
        ))
        
        # Sort other models by preference
        self.excellent_models.sort(key=lambda x: (
            0 if 'best' in x.name else 1,
            x.name
        ))
        
        self.ultimate_models.sort(key=lambda x: (
            0 if 'best' in x.name else 1,
            1 if 'final' in x.name else 2,
            x.name
        ))
        
        # Update model selector based on current type
        self.update_model_selector()
        
        # Load YOLO models
        self.load_yolo_models()
    
    def load_timm_models(self):
        """Load TIMM models (both local and remote) with correct naming"""
        timm_models = []
        
        # First, find local TIMM models
        local_timm_patterns = [
            "*timm*.pth",
            "*timm*.pt", 
            "*efficientnet*.pth",
            "*regnet*.pth",
            "*convnext*.pth",
            "*mobilenet*.pth",
            "*densenet*.pth",
            "*resnet*.pth",
            "*resnext*.pth",
            "*swin*.pth",
            "*vit*.pth"
        ]
        
        local_models = []
        for pattern in local_timm_patterns:
            local_models.extend(Path(".").glob(pattern))
        
        # Add local models with "Local" indicator
        for model_path in local_models:
            model_name = model_path.stem  # Remove extension
            timm_models.append({
                "name": model_name,
                "performance": "💾 Local Model",
                "accuracy": 75.0,  # Default accuracy for local models
                "is_local": True,
                "path": str(model_path)
            })
        
        # Add remote TIMM models with correct names (no timm_ prefix)
        remote_models = [
            # High Performance Models (>80% accuracy) - CORRECT TIMM NAMES
            {"name": "efficientnetv2_m", "performance": "🟢 Excellent (85.2%)", "accuracy": 85.2},
            {"name": "efficientnet_b5", "display_name": "efficientnet_retrained_final", "performance": "🟢 Excellent (84.1%)", "accuracy": 84.1},
            {"name": "regnetx_32gf", "performance": "🟢 Excellent (83.7%)", "accuracy": 83.7},
            {"name": "efficientnetv2_s", "performance": "🟢 Excellent (82.9%)", "accuracy": 82.9},
            {"name": "convnext_small", "performance": "🟢 Excellent (82.1%)", "accuracy": 82.1},
            {"name": "efficientnet_b6", "performance": "🟢 Excellent (81.8%)", "accuracy": 81.8},
            {"name": "tf_efficientnet_b6", "performance": "🟢 Excellent (81.6%)", "accuracy": 81.6},
            {"name": "regnetx_16gf", "performance": "🟢 Excellent (81.5%)", "accuracy": 81.5},
            {"name": "resnext50_32x4d", "performance": "🟢 Excellent (80.3%)", "accuracy": 80.3},
            
            # Good Performance Models (60-80% accuracy) - CORRECT TIMM NAMES
            {"name": "efficientnet_b3", "performance": "🟡 Good (79.2%)", "accuracy": 79.2},
            {"name": "regnetx_32gf", "performance": "🟡 Good (78.9%)", "accuracy": 78.9},
            {"name": "efficientnet_b7", "performance": "🟡 Good (78.1%)", "accuracy": 78.1},
            {"name": "tf_efficientnet_b7", "performance": "🟡 Good (77.9%)", "accuracy": 77.9},
            {"name": "resnext101_32x8d", "performance": "🟡 Good (77.8%)", "accuracy": 77.8},
            {"name": "wide_resnet50_2", "performance": "🟡 Good (76.4%)", "accuracy": 76.4},
            {"name": "efficientnet_b2", "performance": "🟡 Good (75.1%)", "accuracy": 75.1},
            {"name": "regnetx_16gf", "performance": "🟡 Good (73.2%)", "accuracy": 73.2},
            {"name": "densenet169", "performance": "🟡 Good (72.8%)", "accuracy": 72.8},
            {"name": "regnetx_8gf", "performance": "🟡 Good (71.5%)", "accuracy": 71.5},
            {"name": "efficientnet_b1", "performance": "🟡 Good (70.9%)", "accuracy": 70.9},
            {"name": "regnetx_8gf", "performance": "🟡 Good (69.7%)", "accuracy": 69.7},
            {"name": "densenet121", "performance": "🟡 Good (68.4%)", "accuracy": 68.4},
            {"name": "regnetx_4gf", "performance": "🟡 Good (67.1%)", "accuracy": 67.1},
            {"name": "efficientnet_b0", "performance": "🟡 Good (66.8%)", "accuracy": 66.8},
            {"name": "regnetx_4gf", "performance": "🟡 Good (65.3%)", "accuracy": 65.3},
            {"name": "regnetx_3_2gf", "performance": "🟡 Good (64.7%)", "accuracy": 64.7},
            {"name": "regnetx_3_2gf", "performance": "🟡 Good (63.9%)", "accuracy": 63.9},
            {"name": "resnet101", "performance": "🟡 Good (62.4%)", "accuracy": 62.4},
            {"name": "regnetx_1_6gf", "performance": "🟡 Good (61.8%)", "accuracy": 61.8},
            {"name": "regnetx_1_6gf", "performance": "🟡 Good (60.5%)", "accuracy": 60.5},
            
            # Fair Performance Models (40-60% accuracy) - CORRECT TIMM NAMES
            {"name": "regnetx_800mf", "performance": "🟠 Fair (59.1%)", "accuracy": 59.1},
            {"name": "regnetx_800mf", "performance": "🟠 Fair (57.8%)", "accuracy": 57.8},
            {"name": "regnetx_600mf", "performance": "🟠 Fair (56.3%)", "accuracy": 56.3},
            {"name": "regnetx_600mf", "performance": "🟠 Fair (54.9%)", "accuracy": 54.9},
            {"name": "regnetx_400mf", "performance": "🟠 Fair (53.2%)", "accuracy": 53.2},
            {"name": "regnetx_400mf", "performance": "🟠 Fair (51.7%)", "accuracy": 51.7},
            {"name": "regnetx_200mf", "performance": "🟠 Fair (50.4%)", "accuracy": 50.4},
            {"name": "regnetx_200mf", "performance": "🟠 Fair (49.1%)", "accuracy": 49.1},
            {"name": "mobilenetv3_large_100", "performance": "🟠 Fair (44.2%)", "accuracy": 44.2},
            {"name": "convnext_base", "performance": "🟠 Fair (41.9%)", "accuracy": 41.9},
            {"name": "densenet201", "performance": "🟠 Fair (41.6%)", "accuracy": 41.6},
            {"name": "swin_tiny_patch4_window7_224", "performance": "🟠 Fair (40.6%)", "accuracy": 40.6},
            
            # Poor Performance Models (<40% accuracy) - CORRECT TIMM NAMES
            {"name": "resnet50", "performance": "🔴 Poor (39.0%)", "accuracy": 39.0},
            {"name": "efficientnet_b4", "performance": "🔴 Poor (38.6%)", "accuracy": 38.6},
            {"name": "vit_base_patch16_224", "performance": "🔴 Poor (37.3%)", "accuracy": 37.3},
            {"name": "inception_resnet_v2", "performance": "🔴 Poor (32.5%)", "accuracy": 32.5},
        ]
        
        # Add is_local=False to remote models
        for model in remote_models:
            model["is_local"] = False
        
        # Combine local and remote models
        timm_models.extend(remote_models)
        
        print(f"📋 Found {len([m for m in timm_models if m.get('is_local', False)])} local TIMM models")
        print(f"📋 Added {len(remote_models)} remote TIMM models")
        
        return timm_models
    
    def on_model_type_changed(self, event=None):
        """Handle model type selection change (modern interface)"""
        # Modern interface doesn't use model_type_selector dropdown
        # Model type is changed via settings dialog
        self.update_model_selector()
        
        # Update model label (if it exists in old interface)
        if hasattr(self, 'model_label'):
            self.model_label.configure(text=f"{self.current_model_type} Model")
    
    def update_model_selector(self):
        """Update model selector for modern interface (no dropdown needed)"""
        if self.current_model_type == "Excellent":
            models = self.excellent_models
            model_names = [model.name for model in models]
            self.model_paths = [str(model) for model in models]
            no_models_text = "No Excellent models found"
        elif self.current_model_type == "Ultimate":
            models = self.ultimate_models
            model_names = [model.name for model in models]
            self.model_paths = [str(model) for model in models]
            no_models_text = "No Ultimate models found"
        else:  # TIMM Models
            models = self.timm_models
            model_names = [f"{model.get('display_name', model['name'])} - {model['performance']}" for model in models]
            self.model_paths = [model['name'] for model in models]  # Store the actual TIMM model names
            no_models_text = "TIMM library not available"
        
        # Modern interface doesn't use model_selector dropdown
        # Just store the model data for internal use
        print(f"Updated model selector for {self.current_model_type}: {len(model_names) if model_names else 0} models available")
    
    def prioritize_model(self, model_name):
        """Move the specified model to the front of the model_paths list"""
        if hasattr(self, 'model_paths') and self.model_paths:
            if model_name in self.model_paths:
                # Remove from current position and add to front
                self.model_paths.remove(model_name)
                self.model_paths.insert(0, model_name)
                print(f"✅ Prioritized model: {model_name}")
            else:
                print(f"⚠️ Model not found in paths: {model_name}")
    
    def prioritize_retrained_model(self):
        """Prioritize the retrained model (efficientnet_b5) as default"""
        if self.current_model_type == "TIMM Models":
            self.prioritize_model("efficientnet_b5")
            print("🎯 Set retrained model (efficientnet_b5) as default")
    
    def load_yolo_models(self):
        """Load available YOLO models"""
        yolo_models = []
        yolo_patterns = [
            "best.pt",
            "*.pt",
            "runs/finetune/*/weights/best.pt",
            "runs/finetune/*/weights/*.pt"
        ]
        
        for pattern in yolo_patterns:
            for path in Path(".").glob(pattern):
                if path.suffix == '.pt' and path.is_file():
                    yolo_models.append(path)
        
        # Remove duplicates and sort
        yolo_models = list(set(yolo_models))
        yolo_models.sort(key=lambda x: (
            0 if x.name == 'best.pt' else 1,
            str(x)
        ))
        
        # Update selector
        model_names = [str(model) for model in yolo_models]
        self.yolo_paths = model_names
        
        # Modern interface doesn't use yolo_selector dropdown
        # Just store the model data for internal use
        print(f"Found {len(model_names)} YOLO models: {model_names if model_names else 'None'}")
    
    def find_cameras(self):
        """Find available cameras for modern interface"""
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
        
        # Modern interface doesn't use camera_combo - set default camera directly
        if self.available_cameras:
            self.current_camera_index = self.available_cameras[0]
            self.selected_camera = self.available_cameras[0]
        else:
            # Fallback to camera 0
            self.available_cameras = [0]
            self.current_camera_index = 0
            self.selected_camera = 0
        
        print(f"Found cameras: {self.available_cameras}")
        print(f"Selected default camera: {self.current_camera_index}")
    
    def auto_load_model(self):
        """Auto-load saved model state"""
        try:
            saved_model_type, saved_feature_dim, saved_model_path, saved_model_name = self.database.get_current_model_state()
            
            if saved_model_type in ["Excellent", "Ultimate"] and saved_model_path:
                # Set the model type
                self.model_type_selector.set(saved_model_type)
                self.current_model_type = saved_model_type
                self.update_model_selector()
                
                # Try to find and select the saved model
                model_names = self.model_selector['values']
                for i, model_path in enumerate(self.model_paths):
                    if model_path == saved_model_path:
                        self.model_selector.current(i)
                        break
                
                # Auto-load the models
                self.load_models()
                
        except Exception as e:
            print(f"Auto-load failed: {e}")
    
    def auto_load_models(self):
        """Auto-load models for modern interface"""
        try:
            # Update status
            if hasattr(self, 'model_status_header'):
                self.model_status_header.configure(
                    text="🟡 Model: Loading...",
                    fg=ModernTheme.COLORS['warning']
                )
            
            print("🤖 Auto-loading models...")
            
            # Try to load saved model state first
            saved_model_type, saved_feature_dim, saved_model_path, saved_model_name = self.database.get_current_model_state()
            if saved_model_type and saved_model_path:
                print(f"📖 Loaded model state: {saved_model_type} ({saved_feature_dim}D) - {saved_model_name}")
                print(f"📖 DEBUG: Exact model path loaded from database")
                self.current_model_type = saved_model_type
            else:
                print("📖 No saved model state found - using default TIMM Models with retrained model")
                print("📖 DEBUG: No model state in database")
                # Set default to retrained model
                self.current_model_type = "TIMM Models"
                # After loading all models, prioritize the retrained model
                self.root.after(100, lambda: self.prioritize_retrained_model())
            
            # Load the models
            success = self.load_models()
            
            if not success:
                print("❌ Auto-load failed, models need to be loaded manually")
                if hasattr(self, 'model_status_header'):
                    self.model_status_header.configure(
                        text="🔴 Model: Click Load Models",
                        fg=ModernTheme.COLORS['error']
                    )
                # Ensure Load Models button is prominent when auto-load fails
                self.update_load_models_button_state()
            
        except Exception as e:
            print(f"Auto-load failed: {e}")
            if hasattr(self, 'model_status_header'):
                self.model_status_header.configure(
                    text="🔴 Model: Click Load Models",
                    fg=ModernTheme.COLORS['error']
                )
    
    def load_models(self):
        """Load selected models - Using Universal Model Loader for robustness"""
        print("🚀 LOAD MODELS BUTTON CLICKED!")
        print(f"Current model type: {self.current_model_type}")
        print(f"Available model paths: {getattr(self, 'model_paths', 'None')}")
        print(f"Available YOLO paths: {getattr(self, 'yolo_paths', 'None')}")
        try:
            # Load YOLO model (modern interface uses first available)
            selected_yolo = self.yolo_paths[0] if self.yolo_paths else None
            yolo_loaded = False
            
            if selected_yolo:
                try:
                    self.yolo_model = YOLO(selected_yolo)
                    # Update old status indicators if they exist
                    if hasattr(self, 'yolo_status'):
                        self.yolo_status.configure(text="YOLO: Loaded ✓", 
                                                 foreground=ModernTheme.COLORS['success'])
                    if hasattr(self, 'yolo_indicator'):
                        self.yolo_indicator.set_status("active")
                    yolo_loaded = True
                    print(f"✓ YOLO model loaded: {selected_yolo}")
                except Exception as e:
                    print(f"✗ YOLO loading failed: {e}")
                    if hasattr(self, 'yolo_status'):
                        self.yolo_status.configure(text="YOLO: Failed ✗", 
                                                 foreground=ModernTheme.COLORS['error'])
                    if hasattr(self, 'yolo_indicator'):
                        self.yolo_indicator.set_status("error")
            
            # Load feature extraction model (modern interface uses first available)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Always use GPU if available
            selected_model_name = self.model_paths[0] if self.model_paths else None
            model_type = self.current_model_type
            
            no_model_texts = [f"No {model_type} models found", ""]
            if not selected_model_name or selected_model_name in no_model_texts:
                if hasattr(self, 'model_status'):
                    self.model_status.configure(text=f"{model_type} Model: No model selected ✗",
                                              foreground=ModernTheme.COLORS['error'])
                if hasattr(self, 'model_indicator'):
                    self.model_indicator.set_status("error")
                print(f"Error: Please select a {model_type} model")
                return False
            
            # Get full path (modern interface uses first available model)
            selected_model = selected_model_name
            
            feature_loaded = False
            try:
                print(f"Loading {model_type} model: {selected_model}")
                
                # Handle TIMM models separately (they don't use file paths)
                if model_type == "TIMM Models":
                    print("Creating TIMM feature extractor...")
                    self.feature_extractor = self.create_timm_feature_extractor(
                        selected_model, device
                    )
                    feature_loaded = self.feature_extractor is not None
                
                # Use Universal Model Loader if available for file-based models
                elif UNIVERSAL_LOADER_AVAILABLE and model_type in ["Excellent", "Ultimate"]:
                    print("Using Universal Model Loader for robust loading...")
                    self.feature_extractor = UniversalFeatureExtractor(
                        model_path=selected_model,
                        device=device,
                        verbose=True
                    )
                    
                    # Check if model loaded successfully
                    if self.feature_extractor.model is not None:
                        # Update model type based on detected architecture
                        detected_type = self.feature_extractor.model_info.get('type', model_type)
                        if detected_type == 'excellent':
                            self.current_model_type = 'Excellent'
                        elif detected_type == 'ultimate':
                            self.current_model_type = 'Ultimate'
                        
                        feature_loaded = True
                        print(f"✅ Model loaded with Universal Loader")
                        print(f"   Detected type: {detected_type}")
                        print(f"   Feature dimension: {self.feature_extractor.feature_dim}")
                
                # Fall back to original loaders if Universal Loader not available
                elif model_type == "Ultimate":
                    self.feature_extractor = UltimateFeatureExtractor(
                        model_path=selected_model, 
                        device=device
                    )
                    feature_loaded = self.feature_extractor.model is not None
                else:  # Excellent
                    self.feature_extractor = EfficientNetFeatureExtractor(
                        model_path=selected_model, 
                        device=device
                    )
                    feature_loaded = self.feature_extractor.model is not None
                
                if feature_loaded:
                    # Update old status indicators if they exist
                    if hasattr(self, 'model_status'):
                        self.model_status.configure(
                            text=f"{self.current_model_type} Model: Loaded ✓", 
                            foreground=ModernTheme.COLORS['success']
                        )
                    if hasattr(self, 'model_indicator'):
                        self.model_indicator.set_status("active")
                    
                    # Update modern header status
                    if hasattr(self, 'model_status_header'):
                        display_name = "efficientnet_retrained_final" if selected_model == "efficientnet_b5" else selected_model
                        self.model_status_header.configure(
                            text=f"🟢 Model: {display_name}",
                            fg=ModernTheme.COLORS['success']
                        )
                    
                    print(f"✓ {self.current_model_type} model loaded: {selected_model}")
                    
                    # Save model state
                    model_name = Path(selected_model).name if hasattr(Path(selected_model), 'name') else str(selected_model)
                    self.database.save_current_model_state(self.current_model_type, 
                                                          self.feature_extractor.feature_dim, 
                                                          selected_model, model_name)
                else:
                    # Update old status indicators if they exist
                    if hasattr(self, 'model_status'):
                        self.model_status.configure(text=f"{model_type} Model: Failed ✗", 
                                                      foreground=ModernTheme.COLORS['error'])
                    if hasattr(self, 'model_indicator'):
                        self.model_indicator.set_status("error")
                    
                    # Update modern header status
                    if hasattr(self, 'model_status_header'):
                        self.model_status_header.configure(
                            text="🔴 Model: Failed",
                            fg=ModernTheme.COLORS['error']
                        )
                    
            except Exception as e:
                print(f"✗ {model_type} model loading failed: {e}")
                import traceback
                traceback.print_exc()
                if hasattr(self, 'model_status'):
                    self.model_status.configure(text=f"{model_type} Model: Failed ✗", 
                                              foreground=ModernTheme.COLORS['error'])
                if hasattr(self, 'model_indicator'):
                    self.model_indicator.set_status("error")
            
            # Update database if models loaded successfully
            if yolo_loaded and feature_loaded:
                self.update_database()
                self.update_load_models_button_state()  # Update button appearance
                print(f"✅ Models loaded successfully!\n"
                    f"YOLO: {Path(selected_yolo).name}\n"
                    f"{model_type} Model: {Path(selected_model).name}")
                return True
            else:
                error_msg = "Failed to load:\n"
                if not yolo_loaded:
                    error_msg += "- YOLO model\n"
                if not feature_loaded:
                    error_msg += f"- {model_type} model\n"
                print(f"❌ Error: {error_msg}")
                return False
                
        except Exception as e:
            print(f"❌ Model loading failed: {str(e)}")
            return False
    
    def create_timm_feature_extractor(self, model_name, device):
        """Create a TIMM-based feature extractor with automatic download"""
        try:
            import timm
            
            class TimmFeatureExtractor:
                def __init__(self, model_name, device='cpu', timm_models_list=None):
                    self.device = device
                    self.model_name = model_name
                    self.feature_dim = 2048  # Standard feature dimension
                    self.model = None
                    self.is_local = False
                    self.model_path = None
                    
                    # Check if this is a local model
                    if timm_models_list:
                        for timm_model in timm_models_list:
                            if timm_model['name'] == model_name and timm_model.get('is_local', False):
                                self.is_local = True
                                self.model_path = timm_model['path']
                                break
                    
                    # Setup transforms first
                    self.transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                           std=[0.229, 0.224, 0.225])
                    ])
                    
                    # Load model with progress indication
                    self.load_model()
                
                def load_model(self):
                    """Load TIMM model (local or remote) with automatic download"""
                    try:
                        if self.is_local:
                            return self.load_local_model()
                        else:
                            return self.load_remote_model()
                        
                    except Exception as e:
                        print(f"❌ Failed to load TIMM model: {e}")
                        import traceback
                        traceback.print_exc()
                        return False
                
                def load_local_model(self):
                    """Load local TIMM model from file"""
                    print(f"💾 Loading local TIMM model: {self.model_name}")
                    print(f"   Path: {self.model_path}")
                    
                    try:
                        # Load checkpoint
                        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                        
                        # Try to determine the architecture from filename or checkpoint
                        arch_name = self.model_name.lower()
                        if 'efficientnet' in arch_name:
                            if 'b0' in arch_name:
                                base_model = 'efficientnet_b0'
                            elif 'b1' in arch_name:
                                base_model = 'efficientnet_b1'
                            elif 'b2' in arch_name:
                                base_model = 'efficientnet_b2'
                            elif 'b3' in arch_name:
                                base_model = 'efficientnet_b3'
                            elif 'b4' in arch_name:
                                base_model = 'efficientnet_b4'
                            elif 'b5' in arch_name:
                                base_model = 'efficientnet_b5'
                            else:
                                base_model = 'efficientnet_b4'  # Default
                        else:
                            # Try to guess from available models
                            available_models = timm.list_models()
                            base_model = None
                            for model in available_models:
                                if model in arch_name or any(part in model for part in arch_name.split('_')):
                                    base_model = model
                                    break
                            if not base_model:
                                base_model = 'efficientnet_b4'  # Fallback
                        
                        print(f"   Detected base architecture: {base_model}")
                        
                        # Create base model
                        self.model = timm.create_model(base_model, pretrained=False, num_classes=0)
                        
                        # Load weights
                        if 'model_state_dict' in checkpoint:
                            self.model.load_state_dict(checkpoint['model_state_dict'])
                        elif 'state_dict' in checkpoint:
                            self.model.load_state_dict(checkpoint['state_dict'])
                        else:
                            self.model.load_state_dict(checkpoint)
                        
                        self.model.to(self.device)
                        self.model.eval()
                        
                        # Get feature dimension
                        with torch.no_grad():
                            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                            features = self.model(dummy_input)
                            if len(features.shape) == 2:
                                self.feature_dim = features.shape[1]
                            else:
                                features_flat = features.view(features.size(0), -1)
                                self.feature_dim = features_flat.shape[1]
                        
                        print(f"✅ Local TIMM model loaded successfully!")
                        print(f"   Feature dimension: {self.feature_dim}")
                        return True
                        
                    except Exception as e:
                        print(f"❌ Failed to load local model: {e}")
                        return False
                
                def load_remote_model(self):
                    """Load remote TIMM model with automatic download"""
                    print(f"🔄 Loading remote TIMM model: {self.model_name}")
                    print("   This may take a moment if downloading for the first time...")
                    
                    # Check if model exists in TIMM registry
                    available_models = timm.list_models()
                    if self.model_name not in available_models:
                        print(f"❌ Model '{self.model_name}' not found in TIMM registry")
                        print(f"   Available models: {len(available_models)} total")
                        # Try to find similar models
                        similar = [m for m in available_models if any(part in m for part in self.model_name.split('_'))][:5]
                        if similar:
                            print(f"   Similar models: {similar}")
                        return False
                    
                    # Create model with automatic pretrained weight download
                    print(f"📥 Downloading/Loading pretrained weights for {self.model_name}...")
                    try:
                        self.model = timm.create_model(
                            self.model_name, 
                            pretrained=True,  # This will download weights if not cached
                            num_classes=0,    # Return features, not classifications
                            drop_rate=0.0,    # No dropout during inference
                            drop_path_rate=0.0
                        )
                        print(f"✅ Successfully loaded pretrained weights for {self.model_name}")
                    except RuntimeError as e:
                        if "No pretrained weights exist" in str(e):
                            print(f"⚠️ No pretrained weights available for {self.model_name}")
                            print(f"   Falling back to random initialization...")
                            self.model = timm.create_model(
                                self.model_name, 
                                pretrained=False,  # Use random weights
                                num_classes=0,
                                drop_rate=0.0,
                                drop_path_rate=0.0
                            )
                            print(f"✅ Model created with random weights: {self.model_name}")
                        else:
                            raise e
                    
                    # Move to device
                    self.model.to(self.device)
                    self.model.eval()
                    
                    # Get actual feature dimension from model
                    print("🔍 Determining feature dimensions...")
                    with torch.no_grad():
                        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                        features = self.model(dummy_input)
                        if len(features.shape) == 2:
                            self.feature_dim = features.shape[1]
                        else:
                            # Handle different output formats
                            features_flat = features.view(features.size(0), -1)
                            self.feature_dim = features_flat.shape[1]
                    
                    print(f"✅ Remote TIMM model loaded successfully!")
                    print(f"   Model: {self.model_name}")
                    print(f"   Feature dimension: {self.feature_dim}")
                    print(f"   Device: {self.device}")
                    return True
                
                def extract_features(self, image, skip_normalization=False):
                    """Extract features from ear crop using TIMM model"""
                    if self.model is None:
                        print("❌ TIMM model not loaded")
                        return None
                    
                    try:
                        # Ensure model is in evaluation mode
                        self.model.eval()
                        
                        # Handle different input types
                        if isinstance(image, Image.Image):
                            image = np.array(image)
                        
                        # Ensure correct color format (RGB)
                        if isinstance(image, np.ndarray):
                            if len(image.shape) == 3 and image.shape[2] == 3:
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Apply transforms
                        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
                        
                        # Extract features
                        with torch.no_grad():
                            features = self.model(input_tensor)
                        
                        # Handle different output formats
                        if len(features.shape) > 2:
                            features = features.view(features.size(0), -1)
                        
                        final_features = features.cpu().numpy().flatten()
                        
                        # L2 normalize features for better similarity matching
                        if not skip_normalization:
                            norm = np.linalg.norm(final_features)
                            final_features = final_features / (norm + 1e-8)
                        
                        return final_features
                        
                    except Exception as e:
                        print(f"TIMM feature extraction error: {e}")
                        import traceback
                        traceback.print_exc()
                        return None
            
            # Create the extractor
            extractor = TimmFeatureExtractor(model_name, device, self.timm_models)
            
            # Check if model loaded successfully
            if extractor.model is None:
                return None
            
            return extractor
            
        except ImportError:
            messagebox.showerror("Error", 
                "TIMM library is not installed. Please install it with:\npip install timm")
            return None
        except Exception as e:
            print(f"Error creating TIMM feature extractor: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Event handlers and utility methods
    def update_quality_settings(self, *args):
        """Update quality control settings"""
        self.min_ear_size = self.min_size_var.get()
        self.target_ear_size = self.target_size_var.get()
        self.show_guidelines = self.guidelines_var.get()
        
        self.min_size_label.configure(text=f"{self.min_ear_size}px")
        self.target_size_label.configure(text=f"{self.target_ear_size}px")
    
    def update_detection_settings(self, *args):
        """Update detection settings"""
        self.confidence_threshold = self.confidence_var.get()
        self.max_samples = self.max_samples_var.get()
        # Update margin
        if hasattr(self, 'match_margin_var'):
            self.match_margin = self.match_margin_var.get()
        
        self.confidence_label.configure(text=f"{self.confidence_threshold:.2f}")
        self.max_samples_label.configure(text=f"{self.max_samples}")
        if hasattr(self, 'match_margin_label'):
            self.match_margin_label.configure(text=f"{self.match_margin:.2f}")
    
    def on_tab_changed(self, event):
        """Handle tab change events"""
        selected_tab = event.widget.tab('current')['text']
        
        if "Database" in selected_tab:
            self.update_database_display()
        elif "Options" in selected_tab:
            pass  # Options tab doesn't need special handling
    
    def start_identification(self):
        """Start camera for identification mode"""
        self.current_mode = "identify"
        self.start_camera()
    
    def start_enrollment(self):
        """Start enrollment process"""
        if not self.person_name_var.get().strip():
            messagebox.showerror("Error", "Please enter a person name before starting enrollment")
            return
        
        self.current_mode = "enroll"
        self.sample_count = 0
        self.enrollment_samples = []
        self.enrollment_images = []
        self.clear_preview_grid()
        self.update_sample_counter()
        
        # Switch to enrollment tab if not already there
        self.notebook.select(1)  # Enrollment tab
        
        self.start_camera()
    
    def stop_enrollment(self):
        """Stop enrollment process"""
        self.stop_camera()
        
        # If we have samples, show them for review
        if self.enrollment_samples:
            self.update_preview_grid()
    
    def start_camera(self):
        """Start camera capture"""
        if not self.yolo_model or not self.feature_extractor:
            messagebox.showerror("Error", "Please load models first")
            return
        
        if not self.available_cameras:
            messagebox.showerror("Error", "No cameras available")
            return
        
        try:
            # Get selected camera (modern interface uses pre-selected camera)
            if hasattr(self, 'camera_combo') and self.camera_combo.current() >= 0:
                self.current_camera_index = self.available_cameras[self.camera_combo.current()]
            # Modern interface already has current_camera_index set
            
            # Open camera
            self.camera = cv2.VideoCapture(self.current_camera_index)
            if not self.camera.isOpened():
                raise Exception(f"Cannot open camera {self.current_camera_index}")
            
            # Configure camera resolution
            if hasattr(self, 'resolution_var'):
                resolution = self.resolution_var.get().split('x')
                width, height = int(resolution[0]), int(resolution[1])
            else:
                # Default resolution for modern interface
                width, height = 640, 480
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Start processing
            self.is_running = True
            
            # Update UI (old interface only)
            if self.current_mode == "identify" and hasattr(self, 'start_button'):
                self.start_button.text = "Running..."
                self.start_button.draw_button()
            elif hasattr(self, 'start_enroll_button'):
                self.start_enroll_button.text = "Capturing..."
                self.start_enroll_button.draw_button()
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
            # Start display update
            self.update_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
    
    def stop_camera(self):
        """Stop camera capture"""
        self.is_running = False
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        # Update UI (old interface only)
        if self.current_mode == "identify" and hasattr(self, 'start_button'):
            self.start_button.text = "Start Camera"
            self.start_button.draw_button()
        elif hasattr(self, 'start_enroll_button'):
            self.start_enroll_button.text = "Start Enrollment"
            self.start_enroll_button.draw_button()
    
    def camera_loop(self):
        """Main camera processing loop"""
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
                    if hasattr(self, 'fps_label_footer'):
                        self.fps_label_footer.configure(text=f"FPS: {fps:.1f}")
                    fps_counter = 0
                    fps_start = time.time()
                
            except Exception as e:
                if self.debug_mode_var.get():
                    print(f"Processing error: {e}")
    
    def process_detections(self, frame, results):
        """Process ear detections"""
        # Get best detection
        best_box = results.boxes[0]
        confidence = best_box.conf.item()
        box_coords = best_box.xyxy[0].cpu().numpy()
        
        # Calculate ear size
        x1, y1, x2, y2 = map(int, box_coords)
        ear_width = x2 - x1
        ear_height = y2 - y1
        ear_size = min(ear_width, ear_height)
        
        # Quality check
        if ear_size < self.min_ear_size:
            return
        
        # Crop ear
        ear_crop = frame[y1:y2, x1:x2]
        if ear_crop.size == 0:
            return
        
        # Store detection for manual capture
        self.last_detection = {
            'ear_crop': ear_crop,
            'confidence': confidence,
            'ear_size': ear_size
        }
        
        if self.current_mode == "enroll":
            self.process_enrollment_detection(ear_crop, confidence, ear_size)
        else:
            self.process_identification_detection(ear_crop, confidence, ear_size)
    
    def process_enrollment_detection(self, ear_crop, confidence, ear_size):
        """Process enrollment detection"""
        if self.sample_count >= self.max_samples:
            return
        
        # Extract features
        features = self.feature_extractor.extract_features(ear_crop)
        if features is None:
            return
        
        # Store sample
        sample_data = {
            'features': features,
            'image': ear_crop.copy(),
            'confidence': confidence,
            'ear_size': ear_size,
            'timestamp': datetime.now()
        }
        
        self.enrollment_samples.append(sample_data)
        self.enrollment_images.append(ear_crop.copy())
        self.sample_count += 1
        
        print(f"📸 Captured sample {self.sample_count}, total images: {len(self.enrollment_images)}")
        
        # Update counter
        self.update_sample_counter()
        
        # Save sample image if enabled
        if self.save_samples_var.get():
            self.save_enrollment_sample(ear_crop, self.sample_count)
        
        # Update preview
        self.update_preview_grid()
        
        # Provide guidance for next sample
        if self.sample_count < self.max_samples:
            self.show_enrollment_guidance()
        else:
            if hasattr(self, 'instructions_text'):
                self.instructions_text.configure(state='normal')
                self.instructions_text.delete('1.0', tk.END)
                self.instructions_text.insert('1.0', 
                    f"✅ All {self.max_samples} samples captured!\n\n"
                    "Review the samples in the preview below.\n"
                    "If satisfied, click 'Accept Samples' to save.\n"
                    "Otherwise, click 'Reject' to try again.")
                self.instructions_text.configure(state='disabled')
            
            # Auto-stop enrollment
            self.stop_enrollment()
    
    def process_identification_detection(self, ear_crop, confidence, ear_size):
        """Process identification detection with streamlined 3-second analysis"""
        print(f"DEBUG: process_identification_detection - knn_model: {self.knn_model is not None}, person_ids: {len(self.person_ids)}")
        if not self.knn_model or len(self.person_ids) == 0:
            print("DEBUG: No KNN model or person_ids - cannot perform identification")
            if hasattr(self, 'results_text'):
                self.results_text.insert(tk.END, "No enrolled persons\n")
                self.results_text.see(tk.END)
            return
        
        # Extract features
        features = self.feature_extractor.extract_features(ear_crop)
        if features is None:
            if hasattr(self, 'results_text'):
                self.results_text.insert(tk.END, "❌ Feature extraction failed\n")
                self.results_text.see(tk.END)
            return
        
        # Handle streamlined identification state machine
        current_time = time.time()
        
        if self.identification_state == "idle":
            # Start analysis period
            self.identification_state = "analyzing"
            self.identification_start_time = current_time
            self.identification_results = []
            self.update_status_for_analysis()
        
        elif self.identification_state == "analyzing":
            # Collect results during analysis period
            self.collect_identification_sample(features)
            
            # Check if analysis period is complete
            if current_time - self.identification_start_time >= self.analysis_duration:
                self.finalize_identification_analysis()
        
        elif self.identification_state in ["showing_result", "timeout"]:
            # Don't process during result display or timeout
            return
    
    def reset_identification_state(self):
        """Reset identification state to idle"""
        self.identification_state = "idle"
        self.identification_results = []
        if self.current_popup:
            self.current_popup.destroy()
            self.current_popup = None
        
        # Keep face image display - don't clear it automatically
        # The face image will remain visible until a new person is identified
    
    def update_status_for_analysis(self):
        """Update status display during analysis"""
        if hasattr(self, 'status_label'):
            self.status_label.configure(
                text="🔍 Analyzing... (3s)",
                fg=ModernTheme.COLORS['accent_primary']
            )
    
    def collect_identification_sample(self, features):
        """Collect identification samples during analysis period"""
        # Verify feature normalization for Ultimate models
        feature_norm = np.linalg.norm(features)
        if self.current_model_type == "Ultimate" and abs(feature_norm - 1.0) > 0.1:
            if self.debug_mode_var.get():
                print(f"⚠️ Feature norm: {feature_norm:.6f} (expected ~1.0 for Ultimate)")
            # Re-normalize if needed
            features = features / (feature_norm + 1e-8)
        
        # Perform identification
        try:
            n_persons = len(set(self.person_ids))
            n_samples = len(self.person_ids)
            
            # Adaptive k selection
            if n_persons == 1:
                k = min(n_samples, 5)
                distance_threshold = 0.35
            elif n_persons <= 3:
                k = min(n_samples, 7)
                distance_threshold = 0.32
            else:
                k = min(n_samples, min(10, n_samples // 2))
                distance_threshold = 0.30
            
            # Get nearest neighbors
            distances, indices = self.knn_model.kneighbors([features], n_neighbors=k)
            distances = distances[0]
            indices = indices[0]
            
            # Count votes for each person
            person_votes = {}
            person_distances = {}
            
            for i, idx in enumerate(indices):
                person_id = self.person_ids[idx]
                distance = distances[i]
                
                if person_id not in person_votes:
                    person_votes[person_id] = 0
                    person_distances[person_id] = []
                
                person_votes[person_id] += 1
                person_distances[person_id].append(distance)
            
            # Find best match
            if person_votes:
                best_person = max(person_votes.keys(), key=lambda x: person_votes[x])
                best_votes = person_votes[best_person]
                avg_distance = np.mean(person_distances[best_person])
                
                # Apply thresholds
                if avg_distance <= distance_threshold and best_votes >= self.min_votes_for_person:
                    # Get person name
                    try:
                        person_name = self.database.get_person_name(best_person)
                    except AttributeError:
                        print(f"❌ Database object doesn't have get_person_name method")
                        print(f"Database type: {type(self.database)}")
                        print(f"Database methods: {[method for method in dir(self.database) if not method.startswith('_')]}")
                        person_name = "Unknown"
                    confidence = 1.0 - avg_distance
                    
                    # Store result
                    result = {
                        'person_id': best_person,
                        'person_name': person_name,
                        'confidence': confidence,
                        'distance': avg_distance,
                        'votes': best_votes
                    }
                    self.identification_results.append(result)
                else:
                    # No match
                    self.identification_results.append({
                        'person_id': None,
                        'person_name': 'Unknown',
                        'confidence': 0.0,
                        'distance': avg_distance,
                        'votes': 0
                    })
            else:
                self.identification_results.append({
                    'person_id': None,
                    'person_name': 'Unknown',
                    'confidence': 0.0,
                    'distance': 1.0,
                    'votes': 0
                })
                
        except Exception as e:
            print(f"❌ Identification error: {e}")
            self.identification_results.append({
                'person_id': None,
                'person_name': 'Error',
                'confidence': 0.0,
                'distance': 1.0,
                'votes': 0
            })
    
    def finalize_identification_analysis(self):
        """Finalize identification analysis and show results"""
        print(f"DEBUG: finalize_identification_analysis - {len(self.identification_results)} results")
        if not self.identification_results:
            print("DEBUG: No identification results - showing 'No Detection'")
            self.show_identification_popup({
                'person_name': 'No Detection',
                'confidence': 0.0,
                'person_id': None
            })
            return
        
        # Find the single highest confidence match (not averaged)
        best_result = None
        best_confidence = 0.0
        
        print(f"DEBUG: Analyzing {len(self.identification_results)} results:")
        for i, result in enumerate(self.identification_results):
            print(f"  Result {i+1}: {result['person_name']} - Confidence: {result['confidence']:.3f}")
            if result['confidence'] > best_confidence:
                best_confidence = result['confidence']
                best_result = result
        
        print(f"DEBUG: Best result: {best_result['person_name'] if best_result else 'None'} - Confidence: {best_confidence:.3f}")
        
        # Show result popup with the highest confidence match
        if best_result:
            final_result = {
                'person_name': best_result['person_name'],
                'confidence': best_result['confidence'],
                'person_id': best_result['person_id']
            }
        else:
            final_result = {
                'person_name': 'Unknown',
                'confidence': 0.0,
                'person_id': None
            }
        
        self.show_identification_popup(final_result)
    
    def show_identification_popup(self, result):
        """Show identification result in popup and results text"""
        self.identification_state = "showing_result"
        
        # Display result in the results text area
        if hasattr(self, 'results_text'):
            self.results_text.configure(state='normal')
            self.results_text.delete('1.0', tk.END)
            
            if result['person_name'] == 'No Detection':
                self.results_text.insert(tk.END, "❌ No Detection\n\nNo ear detected in the image.\nPlease ensure your ear is clearly visible.")
            elif result['person_name'] == 'Unknown':
                self.results_text.insert(tk.END, f"❓ Unknown Person\n\nConfidence: {result['confidence']:.2%}\n\nThis person is not in the database.")
            else:
                confidence_percent = result['confidence'] * 100
                self.results_text.insert(tk.END, f"✅ {result['person_name']}\n\n")
                self.results_text.insert(tk.END, f"Confidence: {confidence_percent:.1f}%\n")
                self.results_text.insert(tk.END, f"Person ID: {result['person_id']}\n\n")
                
                if confidence_percent >= 80:
                    self.results_text.insert(tk.END, "🎯 High confidence match!")
                elif confidence_percent >= 60:
                    self.results_text.insert(tk.END, "✅ Good match")
                elif confidence_percent >= 40:
                    self.results_text.insert(tk.END, "⚠️ Low confidence match")
                else:
                    self.results_text.insert(tk.END, "❌ Very low confidence")
            
            self.results_text.configure(state='disabled')
            self.results_text.see(tk.END)
        
        # Get face photo if person is identified
        face_photo = None
        if result['person_id']:
            face_photo = self.database.get_person_face_photo(result['person_id'])
        
        # Update face image display in header
        self.update_face_image_display(face_photo, result)
        
        # Create popup (optional - you can comment this out if you don't want popups)
        try:
            self.current_popup = IdentificationPopup(
                self.main_frame,
                result,
                face_photo=face_photo,
                timeout_callback=self.on_identification_timeout,
                override_callback=self.on_identification_override
            )
            
            # Start timeout timer
            self.root.after(int(self.result_timeout * 1000), self.on_identification_timeout)
        except Exception as e:
            print(f"DEBUG: Popup creation failed: {e}")
            # If popup fails, just reset the state
            self.root.after(1000, self.reset_identification_state)
    
    def on_identification_timeout(self):
        """Handle identification result timeout"""
        if self.current_popup:
            self.current_popup.destroy()
            self.current_popup = None
        self.identification_state = "timeout"
        self.root.after(1000, self.reset_identification_state)  # 1 second cooldown
    
    def update_face_image_display(self, face_photo, result):
        """Update the face image display in the header"""
        if hasattr(self, 'face_image_label'):
            if face_photo is not None:
                try:
                    # Resize face photo for display
                    face_display = cv2.resize(face_photo, (50, 50))
                    face_rgb = cv2.cvtColor(face_display, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb)
                    face_tk = ImageTk.PhotoImage(face_pil)
                    
                    # Update the label with the face image
                    self.face_image_label.configure(image=face_tk, text="")
                    self.face_image_label.image = face_tk  # Keep reference
                    
                    print(f"✅ Face image displayed for {result['person_name']}")
                except Exception as e:
                    print(f"❌ Error displaying face image: {e}")
                    # Fallback to person icon
                    self.face_image_label.configure(image="", text="👤")
            else:
                # No face photo available, show person icon
                self.face_image_label.configure(image="", text="👤")
    
    def on_identification_override(self):
        """Handle manual override of identification timeout"""
        if self.current_popup:
            self.current_popup.destroy()
            self.current_popup = None
        self.reset_identification_state()
        
        # Optimized matching strategy (legacy code to be removed)
        try:
            n_persons = len(set(self.person_ids))
            n_samples = len(self.person_ids)
            
            # Adaptive k selection - key improvement
            if n_persons == 1:
                # Single person: use all samples but higher threshold
                k = min(n_samples, 5)
                confidence_boost = 0.05  # Require higher confidence
            elif n_persons <= 3:
                # Few persons: use more samples per person
                k = min(n_samples, 7)
                confidence_boost = 0.02
            else:
                # Many persons: use fewer samples for speed
                k = min(n_samples, max(3, n_persons))
                confidence_boost = 0.0
            
            distances, indices = self.knn_model.kneighbors([features], n_neighbors=k)
            dists = distances[0]
            idxs = indices[0]
            
            # Enhanced scoring strategy
            person_scores = {}
            person_weights = {}
            person_samples = {}
            
            for i, (dist, idx) in enumerate(zip(dists, idxs)):
                pid = self.person_ids[idx]
                similarity = 1.0 - float(dist)
                
                # Weight by rank (closer matches matter more)
                rank_weight = 1.0 / (i + 1)
                
                # Weight by similarity (higher similarity gets more weight)
                sim_weight = similarity ** 2
                
                # Combined weight
                total_weight = rank_weight * sim_weight
                
                if pid not in person_scores:
                    person_scores[pid] = []
                    person_weights[pid] = []
                    person_samples[pid] = 0
                
                person_scores[pid].append(similarity)
                person_weights[pid].append(total_weight)
                person_samples[pid] += 1
            
            # Compute weighted average scores
            final_scores = {}
            for pid in person_scores:
                scores = np.array(person_scores[pid])
                weights = np.array(person_weights[pid])
                
                # Weighted average with bias toward higher scores
                if len(scores) > 1:
                    # Use weighted average of top scores
                    weighted_avg = np.average(scores, weights=weights)
                    # Boost score if person has multiple high-quality matches
                    consistency_bonus = min(0.05, 0.01 * person_samples[pid])
                    final_scores[pid] = weighted_avg + consistency_bonus
                else:
                    final_scores[pid] = scores[0]
            
            # Rank by final scores
            ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
            
            if not ranked:
                self.results_text.insert(tk.END, "❌ No match (no neighbors)\n")
                self.results_text.see(tk.END)
                return
            
            best_pid, best_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else 0.0
            margin = best_score - second_score
            sample_count = person_samples.get(best_pid, 0)
            
            # Adaptive thresholding based on model type and database size
            if self.current_model_type == "Ultimate":
                # Ultimate models are more reliable, can use lower thresholds
                base_threshold = max(0.80, self.confidence_threshold - 0.05)
                min_margin = max(0.05, self.match_margin - 0.02)
            elif self.current_model_type == "TIMM Models":
                # TIMM models vary in performance - use adaptive thresholds based on model
                model_name = self.feature_extractor.model_name if hasattr(self.feature_extractor, 'model_name') else ""
                
                # Find the model's performance rating
                model_accuracy = 50.0  # Default
                for timm_model in self.timm_models:
                    if timm_model['name'] in model_name:
                        model_accuracy = timm_model['accuracy']
                        break
                
                # Adjust thresholds based on model performance
                if model_accuracy >= 80:  # Excellent models
                    base_threshold = max(0.75, self.confidence_threshold - 0.10)
                    min_margin = max(0.03, self.match_margin - 0.03)
                elif model_accuracy >= 60:  # Good models  
                    base_threshold = max(0.80, self.confidence_threshold - 0.05)
                    min_margin = max(0.05, self.match_margin - 0.02)
                else:  # Fair/Poor models
                    base_threshold = max(0.85, self.confidence_threshold)
                    min_margin = self.match_margin
            else:
                # Excellent models need higher thresholds
                base_threshold = self.confidence_threshold
                min_margin = self.match_margin
            
            # Apply confidence boost for single person scenarios
            effective_threshold = base_threshold + confidence_boost
            
            # Decision logic
            passed_conf = best_score >= effective_threshold
            passed_margin = (margin >= min_margin) or (n_persons == 1)  # Skip margin check for single person
            passed_samples = sample_count >= 1  # At least one match required
            
            # Quality check: ensure the match makes sense
            if best_score > 0.98 and n_persons > 1:
                # Suspiciously high confidence with multiple people - be more careful
                passed_conf = passed_conf and (margin > 0.05)
            
            if self.debug_mode_var.get():
                print(f"🔍 Enhanced Match Analysis:")
                print(f"   Model: {self.current_model_type}, Feature norm: {feature_norm:.6f}")
                print(f"   Database: {n_persons} persons, {n_samples} samples, k={k}")
                print(f"   Best: {best_score:.3f}, Second: {second_score:.3f}, Margin: {margin:.3f}")
                print(f"   Thresholds: conf≥{effective_threshold:.3f}, margin≥{min_margin:.3f}")
                print(f"   Passed: conf={passed_conf}, margin={passed_margin}, samples={passed_samples}")
            
            if passed_conf and passed_margin and passed_samples:
                # Resolve name
                person_name = "Unknown"
                for p_id, name, _, _ in self.database.get_persons():
                    if p_id == best_pid:
                        person_name = name
                        break
                
                # Show confidence level
                if best_score >= 0.95:
                    conf_level = "HIGH"
                elif best_score >= 0.85:
                    conf_level = "MEDIUM"
                else:
                    conf_level = "LOW"
                
                self.results_text.insert(tk.END, 
                    f"✅ MATCH: {person_name} ({conf_level} confidence: {best_score:.3f})\n")
            else:
                # Show why it failed
                reasons = []
                if not passed_conf:
                    reasons.append(f"conf {best_score:.3f}<{effective_threshold:.3f}")
                if not passed_margin:
                    reasons.append(f"margin {margin:.3f}<{min_margin:.3f}")
                if not passed_samples:
                    reasons.append("insufficient samples")
                
                reason_str = ", ".join(reasons)
                self.results_text.insert(tk.END, f"❌ No match ({reason_str})\n")
            
            self.results_text.see(tk.END)
            
        except Exception as e:
            if self.debug_mode_var.get():
                print(f"Identification error: {e}")
                import traceback
                traceback.print_exc()
    
    def add_guidelines(self, frame):
        """Add visual guidelines to frame"""
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Target area (green)
        target_size = self.target_ear_size
        half_target = target_size // 2
        cv2.rectangle(frame, 
                     (center_x - half_target, center_y - half_target),
                     (center_x + half_target, center_y + half_target),
                     (0, 255, 0), 2)
        
        # Minimum area (orange)
        min_size = self.min_ear_size
        half_min = min_size // 2
        cv2.rectangle(frame, 
                     (center_x - half_min, center_y - half_min),
                     (center_x + half_min, center_y + half_min),
                     (0, 165, 255), 1)
        
        # Labels
        cv2.putText(frame, f"Target: {target_size}px", 
                   (center_x - half_target, center_y - half_target - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(frame, "Position ear within green box for best quality", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def update_display(self):
        """Update video display for modern interface"""
        if not self.is_running:
            return
        
        try:
            frame = self.frame_queue.get_nowait()
            
            # Convert and resize for display (optimized for 1024x600 screen)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize to fit the video panel (approximately 700x400 for the main video area)
            frame_resized = cv2.resize(frame_rgb, (700, 400))
            
            # Convert to PhotoImage
            image = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image)
            
            # Update video label
            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo
            
        except queue.Empty:
            pass
        except Exception as e:
            if hasattr(self, 'debug_mode_var') and self.debug_mode_var.get():
                print(f"Display error: {e}")
        
        # Schedule next update
        self.root.after(30, self.update_display)
    
    # Enrollment-specific methods
    def update_sample_counter(self):
        """Update sample counter display for modern interface"""
        if hasattr(self, 'sample_counter'):
            self.sample_counter.configure(text=f"Samples: {self.sample_count}/{self.max_samples}")
            
            # Update progress color
            if self.sample_count >= self.max_samples:
                self.sample_counter.configure(fg=ModernTheme.COLORS['success'])
            elif self.sample_count >= self.max_samples // 2:
                self.sample_counter.configure(fg=ModernTheme.COLORS['warning'])
            else:
                self.sample_counter.configure(fg=ModernTheme.COLORS['accent_primary'])
    
    def show_enrollment_guidance(self):
        """Show guidance for next enrollment sample"""
        guidance_messages = [
            "Great! Now move slightly to the left for the next sample.",
            "Perfect! Now turn your head slightly right for variety.",
            "Excellent! Now tilt your head slightly up for the next sample.",
            "Good! Now try a slightly different angle for the final sample."
        ]
        
        if self.sample_count <= len(guidance_messages):
            message = guidance_messages[self.sample_count - 1]
            if hasattr(self, 'instructions_text'):
                self.instructions_text.configure(state='normal')
                self.instructions_text.delete('1.0', tk.END)
                self.instructions_text.insert('1.0', f"Sample {self.sample_count} captured!\n\n{message}")
                self.instructions_text.configure(state='disabled')
    
    def save_enrollment_sample(self, ear_crop, sample_num):
        """Save enrollment sample to local storage"""
        try:
            person_name = self.person_name_var.get().strip().replace(' ', '_')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{person_name}_{timestamp}_sample_{sample_num}.jpg"
            filepath = self.samples_dir / filename
            
            cv2.imwrite(str(filepath), ear_crop)
            
        except Exception as e:
            if self.debug_mode_var.get():
                print(f"Error saving sample: {e}")
    
    def clear_preview_grid(self):
        """Clear the preview grid"""
        if hasattr(self, 'preview_frame'):
            for widget in self.preview_frame.winfo_children():
                widget.destroy()
                # Remove placeholder reference
                if hasattr(self, 'preview_placeholder'):
                    self.preview_placeholder = None
    
    def update_preview_grid(self):
        """Update the sample preview grid with delete functionality"""
        print(f"🔄 Updating preview grid with {len(self.enrollment_images)} images")
        self.clear_preview_grid()
        
        if not self.enrollment_images:
            print("⚠️ No enrollment images to display")
            # Show placeholder message
            placeholder = tk.Label(self.preview_frame, 
                                 text="No samples captured yet\nStart enrollment to begin",
                                 bg=ModernTheme.COLORS['bg_tertiary'],
                                 fg=ModernTheme.COLORS['text_secondary'],
                                 font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 10))
            placeholder.pack(expand=True, fill='both', padx=10, pady=10)
            return
            
        # Create grid of sample previews with delete buttons
        cols = min(3, len(self.enrollment_images))
        for i, img in enumerate(self.enrollment_images):
            row = i // cols
            col = i % cols
            
            # Create container for each sample
            sample_container = tk.Frame(self.preview_frame, bg=ModernTheme.COLORS['bg_secondary'])
            sample_container.grid(row=row, column=col, padx=5, pady=5)
            
            # Resize image for preview
            try:
                img_resized = cv2.resize(img, (100, 100))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                photo = ImageTk.PhotoImage(img_pil)
                print(f"✅ Created preview image {i+1}")
            except Exception as e:
                print(f"❌ Error creating preview image {i+1}: {e}")
                continue
            
            # Create preview label
            preview_label = tk.Label(sample_container, image=photo,
                                   bg=ModernTheme.COLORS['bg_tertiary'],
                                   relief='solid', borderwidth=2)
            preview_label.image = photo  # Keep reference
            preview_label.pack(pady=(5, 2))
            
            # Sample info
            info_label = tk.Label(sample_container, text=f"Sample {i+1}",
                                bg=ModernTheme.COLORS['bg_secondary'],
                                fg=ModernTheme.COLORS['text_primary'],
                                font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 9, 'bold'))
            info_label.pack()
            
            # Quality indicator
            if i < len(self.enrollment_samples):
                confidence = self.enrollment_samples[i].get('confidence', 0.0)
                quality_color = ModernTheme.COLORS['success'] if confidence > 0.8 else ModernTheme.COLORS['warning']
                quality_label = tk.Label(sample_container, text=f"Quality: {confidence:.2f}",
                                       bg=ModernTheme.COLORS['bg_secondary'],
                                       fg=quality_color,
                                       font=(ModernTheme.TYPOGRAPHY['font_family_fallback'], 8))
                quality_label.pack()
            
            # Delete button
            delete_btn = ModernButton(sample_container, text="🗑️", 
                                    command=lambda idx=i: self.delete_sample(idx),
                                    width=30, height=25,
                                    bg_color=ModernTheme.COLORS['error'])
            delete_btn.pack(pady=(2, 5))
    
    def delete_sample(self, sample_index):
        """Delete a specific enrollment sample"""
        if 0 <= sample_index < len(self.enrollment_samples):
            # Remove from both lists
            del self.enrollment_samples[sample_index]
            del self.enrollment_images[sample_index]
            
            # Update sample count
            self.sample_count = len(self.enrollment_samples)
            self.update_sample_counter()
            
            # Refresh preview
            self.update_preview_grid()
            
            # Update status
            if hasattr(self, 'status_label'):
                self.status_label.configure(text=f"Deleted sample {sample_index + 1}", 
                                          fg=ModernTheme.COLORS['warning'])
            
            print(f"🗑️ Deleted sample {sample_index + 1}, {self.sample_count} samples remaining")
    
    def capture_face_photo(self):
        """Capture face photo from current camera feed"""
        if not self.is_running or not self.camera:
            self.status_label.configure(text="Start camera first to capture face photo", 
                                      fg=ModernTheme.COLORS['error'])
            return
        
        try:
            # Capture current frame
            ret, frame = self.camera.read()
            if ret:
                # Resize and store face photo
                face_photo = cv2.resize(frame, (120, 120))
                self.current_face_photo = face_photo.copy()
                
                # Update preview
                self.update_face_photo_preview()
                
                self.status_label.configure(text="Face photo captured!", 
                                          fg=ModernTheme.COLORS['success'])
            else:
                self.status_label.configure(text="Failed to capture face photo", 
                                          fg=ModernTheme.COLORS['error'])
        except Exception as e:
            self.status_label.configure(text=f"Face capture error: {str(e)}", 
                                      fg=ModernTheme.COLORS['error'])
    
    def select_face_photo(self):
        """Select face photo from file"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Face Photo",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                # Load and resize image
                face_photo = cv2.imread(file_path)
                if face_photo is not None:
                    face_photo = cv2.resize(face_photo, (120, 120))
                    self.current_face_photo = face_photo.copy()
                    
                    # Update preview
                    self.update_face_photo_preview()
                    
                    self.status_label.configure(text="Face photo selected!", 
                                              fg=ModernTheme.COLORS['success'])
                else:
                    self.status_label.configure(text="Failed to load face photo", 
                                              fg=ModernTheme.COLORS['error'])
        except Exception as e:
            self.status_label.configure(text=f"Face selection error: {str(e)}", 
                                      fg=ModernTheme.COLORS['error'])
    
    def update_face_photo_preview(self):
        """Update the face photo preview"""
        if self.current_face_photo is not None:
            try:
                # Convert to RGB and create PhotoImage
                face_rgb = cv2.cvtColor(self.current_face_photo, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)
                face_photo = ImageTk.PhotoImage(face_pil)
                
                # Update label
                self.face_photo_label.configure(image=face_photo, text="")
                self.face_photo_label.image = face_photo  # Keep reference
            except Exception as e:
                print(f"Error updating face photo preview: {e}")
        else:
            # Show placeholder
            self.face_photo_label.configure(image="", text="No Photo")
    
    def accept_samples(self):
        """Accept and save enrollment samples"""
        if not self.enrollment_samples:
            messagebox.showwarning("Warning", "No samples to accept")
            return
        
        if len(self.enrollment_samples) < 3:
            messagebox.showwarning("Warning", "Need at least 3 samples for enrollment")
            return
        
        try:
            # Save to database
            person_id = str(uuid.uuid4())
            person_name = self.person_name_var.get().strip()
            
            feature_vectors = [sample['features'] for sample in self.enrollment_samples]
            confidences = [sample['confidence'] for sample in self.enrollment_samples]
            
            # Generate image paths for saved samples
            image_paths = []
            if self.save_samples_var.get():
                for i, sample in enumerate(self.enrollment_samples):
                    person_name_safe = person_name.replace(' ', '_')
                    timestamp = sample['timestamp'].strftime("%Y%m%d_%H%M%S")
                    filename = f"{person_name_safe}_{timestamp}_sample_{i+1}.jpg"
                    filepath = self.samples_dir / filename
                    image_paths.append(str(filepath))
                    
                    # Save the image if not already saved
                    if not filepath.exists():
                        cv2.imwrite(str(filepath), sample['image'])
            else:
                image_paths = [None] * len(self.enrollment_samples)
            
            # Check if face_photo parameter is supported
            try:
                success = self.database.add_person(
                    person_id, person_name, feature_vectors,
                    confidences=confidences, 
                    model_type=self.current_model_type, 
                    feature_dim=self.feature_extractor.feature_dim,
                    image_paths=image_paths,
                    face_photo=self.current_face_photo
                )
            except TypeError:
                # Fallback for older database version
                print("⚠️ Using fallback add_person without face_photo")
            success = self.database.add_person(
                person_id, person_name, feature_vectors,
                confidences=confidences, 
                model_type=self.current_model_type, 
                feature_dim=self.feature_extractor.feature_dim,
                image_paths=image_paths
            )
                
                # Save face photo separately if provided
            if self.current_face_photo is not None:
                try:
                    conn = sqlite3.connect(self.database.db_path)
                    cursor = conn.cursor()
                    face_photo_blob = pickle.dumps(self.current_face_photo)
                    cursor.execute(
                        "INSERT OR REPLACE INTO face_photos (person_id, face_photo) VALUES (?, ?)",
                        (person_id, face_photo_blob)
                    )
                    conn.commit()
                    conn.close()
                    print("✅ Face photo saved separately")
                except Exception as e:
                    print(f"❌ Error saving face photo separately: {e}")
            
            if success:
                messagebox.showinfo("Success", 
                    f"Successfully enrolled {person_name} with {len(feature_vectors)} samples!")
                
                # Clear enrollment data
                self.clear_enrollment_data()
                
                # Update database display
                self.update_database()
                
            else:
                messagebox.showerror("Error", "Failed to save enrollment to database")
                
        except Exception as e:
            messagebox.showerror("Error", f"Enrollment failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def reject_samples(self):
        """Reject current samples and restart"""
        result = messagebox.askyesno("Confirm", 
            "Are you sure you want to reject these samples and start over?")
        
        if result:
            self.clear_enrollment_data()
    
    def clear_enrollment_samples(self):
        """Clear all enrollment samples"""
        result = messagebox.askyesno("Confirm", 
            "Are you sure you want to clear all samples?")
        
        if result:
            self.clear_enrollment_data()
    
    def clear_enrollment_data(self):
        """Clear all enrollment data"""
        self.enrollment_samples = []
        self.enrollment_images = []
        self.sample_count = 0
        self.current_face_photo = None
        self.update_sample_counter()
        self.clear_preview_grid()
        self.update_face_photo_preview()
        
        # Reset instructions if they exist
        if hasattr(self, 'instructions_text'):
            self.instructions_text.configure(state='normal')
            self.instructions_text.delete('1.0', tk.END)
            self.instructions_text.insert('1.0', """Welcome to Enrollment!

1. Enter your full name above
2. Click 'Start Enrollment' 
3. Position your ear in the green target box
4. Hold still when prompted for each sample
5. Move slightly between samples for variety
6. Review samples before confirming
7. Click 'Save Enrollment' to complete

Tips:
• Good lighting improves quality
• Keep ear clearly visible
• Avoid hair covering the ear
• Stay within the target area""")
        self.instructions_text.configure(state='disabled')
    
    # Database methods
    def update_database(self):
        """Update database and kNN model with compatibility checking"""
        try:
            if not self.feature_extractor:
                print("⚠️ No feature extractor loaded")
                return
            
            # Load features from database with detailed compatibility checking
            person_ids, features = self.database.get_all_features(
                model_type=self.current_model_type, 
                feature_dim=self.feature_extractor.feature_dim
            )
            
            if len(person_ids) > 0 and isinstance(features, np.ndarray):
                # Validate feature dimensions
                expected_dim = self.feature_extractor.feature_dim
                actual_dim = features.shape[1] if len(features.shape) > 1 else len(features[0])
                
                if expected_dim != actual_dim:
                    print(f"⚠️ Feature dimension mismatch: Expected {expected_dim}D, got {actual_dim}D")
                    
                    # Offer to clear database or suggest switching models
                    response = messagebox.askyesnocancel(
                        "Database Compatibility Issue",
                        f"The current {self.current_model_type} model expects {expected_dim}D features, "
                        f"but the database contains {actual_dim}D features.\n\n"
                        f"Options:\n"
                        f"• Yes: Clear database and start fresh\n"
                        f"• No: Keep database (may cause errors)\n"
                        f"• Cancel: Do nothing\n\n"
                        f"What would you like to do?"
                    )
                    
                    if response is True:  # Yes - clear database
                        self.clear_all_database()
                        return
                    elif response is False:  # No - continue with warning
                        print("⚠️ Continuing with incompatible features - expect errors")
                    else:  # Cancel - do nothing
                        return
                
                # Build kNN model
                self.person_ids = person_ids
                self.feature_database = features
                
                n_samples = features.shape[0]
                n_persons = len(set(person_ids))
                
                # Optimized kNN configuration for Ultimate models
                max_neighbors = min(n_samples, max(3, min(10, n_persons * 2)))
                
                # Use different algorithms based on dataset size for better performance
                if n_samples < 100:
                    algorithm = 'brute'  # More accurate for small datasets
                elif n_samples < 1000:
                    algorithm = 'ball_tree'  # Good balance
                else:
                    algorithm = 'kd_tree'  # Faster for large datasets
                
                self.knn_model = NearestNeighbors(
                    n_neighbors=max_neighbors,
                    metric='cosine',
                    algorithm=algorithm,
                    n_jobs=-1  # Use all CPU cores for faster search
                )
                self.knn_model.fit(features)
                
                print(f"✅ Database updated: {n_samples} samples from {n_persons} persons")
                
                # Check for potential model mismatch by examining feature distributions
                if n_samples > 0:
                    feature_norms = np.linalg.norm(features, axis=1)
                    avg_norm = np.mean(feature_norms)
                    print(f"🔍 Average feature norm: {avg_norm:.6f} (should be ~1.0 for normalized features)")
                    
                    if avg_norm < 0.5 or avg_norm > 2.0:
                        print("⚠️ Unusual feature norms detected - possible model mismatch")
                        
            else:
                # Clear kNN model if no features
                self.knn_model = None
                self.person_ids = []
                self.feature_database = None
                
                # Check if database has any features at all
                conn = sqlite3.connect(self.database.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM features")
                total_features = cursor.fetchone()[0]
                conn.close()
                
                if total_features > 0:
                    print(f"⚠️ Database has {total_features} features but none are compatible with current model")
                    print("💡 Consider clearing the database or loading a compatible model")
                else:
                    print("📊 Database is empty - ready for new enrollments")
            
            # Update persons list
            self.update_persons_list()
            
        except Exception as e:
            print(f"Database update error: {e}")
            import traceback
            traceback.print_exc()
    
    def update_persons_list(self):
        """Update the persons list display"""
        persons = self.database.get_persons()
        
        # Update main tab persons list (if it exists)
        if hasattr(self, 'persons_listbox'):
            self.persons_listbox.delete(0, tk.END)
            for person_id, name, num_samples, created_at in persons:
                display_text = f"{name} ({num_samples} samples)"
                self.persons_listbox.insert(tk.END, display_text)
        
        # Update database tab with new card-based system
        if hasattr(self, 'person_cards'):
            self.refresh_database_view()
        elif hasattr(self, 'db_persons_listbox'):
            # Fallback for old system
            self.db_persons_listbox.delete(0, tk.END)
        for person_id, name, num_samples, created_at in persons:
            display_text = f"{name} ({num_samples} samples)"
            self.db_persons_listbox.insert(tk.END, display_text)
        
        # Update person count (if it exists)
        if hasattr(self, 'person_count_label'):
            self.person_count_label.configure(text=f"{len(persons)} persons")
    
    def update_database_display(self):
        """Update database tab display"""
        self.update_persons_list()
        self.update_database_stats()
    
    def update_database_stats(self):
        """Update database statistics display"""
        try:
            persons = self.database.get_persons()
            total_persons = len(persons)
            
            if total_persons > 0:
                total_samples = sum(num_samples for _, _, num_samples, _ in persons)
                avg_samples = total_samples / total_persons
                
                # Get creation dates
                dates = [created_at for _, _, _, created_at in persons]
                oldest = min(dates) if dates else "N/A"
                newest = max(dates) if dates else "N/A"
                
                stats_text = f"""Database Statistics:

Total Persons: {total_persons}
Total Samples: {total_samples}
Average Samples per Person: {avg_samples:.1f}

Oldest Entry: {oldest}
Newest Entry: {newest}

Model Information:
Type: {self.current_model_type} ({self.current_model_type} EfficientNet)
Feature Dimension: {self.feature_extractor.feature_dim if self.feature_extractor else 'N/A'}
Device: {'CUDA' if torch.cuda.is_available() and self.use_gpu_var.get() else 'CPU'}

Database File: {self.database.db_path}
Sample Storage: {self.samples_dir}"""
            else:
                stats_text = """Database Statistics:

No persons enrolled yet.

To get started:
1. Select model type (Excellent/Ultimate)
2. Load the selected model
3. Go to Enrollment tab
4. Enter a person's name
5. Capture samples
6. Save enrollment

The database will automatically
store all biometric data securely."""
            
            self.stats_text.configure(state='normal')
            self.stats_text.delete('1.0', tk.END)
            self.stats_text.insert('1.0', stats_text)
            self.stats_text.configure(state='disabled')
            
        except Exception as e:
            if self.debug_mode_var.get():
                print(f"Stats update error: {e}")
    
    # Database management methods (placeholder implementations)
    def on_person_select(self, event):
        """Handle person selection in database tab"""
        selection = self.db_persons_listbox.curselection()
        if not selection:
            return
        
        # Get selected person info
        selected_text = self.db_persons_listbox.get(selection[0])
        person_name = selected_text.split(' (')[0]
        
        # Find person details
        persons = self.database.get_persons()
        for person_id, name, num_samples, created_at in persons:
            if name == person_name:
                details = f"""Person Details:

Name: {name}
ID: {person_id}
Samples: {num_samples}
Enrolled: {created_at}

Model: {self.current_model_type} EfficientNet
Status: Active"""
                
                self.person_details_text.configure(state='normal')
                self.person_details_text.delete('1.0', tk.END)
                self.person_details_text.insert('1.0', details)
                self.person_details_text.configure(state='disabled')
                break
    
    def view_person_samples(self):
        """View samples for selected person"""
        selection = self.db_persons_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a person to view samples")
            return
        
        selected_text = self.db_persons_listbox.get(selection[0])
        person_name = selected_text.split(' (')[0]
        
        # Find person ID
        persons = self.database.get_persons()
        person_id = None
        for p_id, name, _, _ in persons:
            if name == person_name:
                person_id = p_id
                break
        
        if not person_id:
            messagebox.showerror("Error", "Could not find person in database")
            return
        
        # Get samples
        samples = self.database.get_person_samples(person_id)
        if not samples:
            messagebox.showinfo("No Samples", f"No samples found for {person_name}")
            return
        
        # Create sample viewer window
        self.create_sample_viewer_window(person_name, samples)
    
    def create_sample_viewer_window(self, person_name, samples):
        """Create a window to view person samples"""
        viewer_window = tk.Toplevel(self.root)
        viewer_window.title(f"Sample Viewer - {person_name}")
        viewer_window.geometry("800x600")
        viewer_window.configure(bg=ModernTheme.COLORS['bg_primary'])
        
        # Header
        header_frame = GlassFrame(viewer_window)
        header_frame.pack(fill='x', padx=10, pady=10)
        
        title_label = ttk.Label(header_frame, 
                               text=f"Samples for {person_name} ({len(samples)} samples)",
                               style='Title.TLabel')
        title_label.pack(pady=10)
        
        # Create scrollable frame for samples
        canvas = tk.Canvas(viewer_window, bg=ModernTheme.COLORS['bg_secondary'])
        scrollbar = ttk.Scrollbar(viewer_window, orient="vertical", command=canvas.yview)
        scrollable_frame = GlassFrame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=(0, 10))
        scrollbar.pack(side="right", fill="y", padx=(0, 10), pady=(0, 10))
        
        # Display samples
        for i, sample in enumerate(samples):
            self.create_sample_display(scrollable_frame, i+1, sample)
        
        # Close button
        close_frame = tk.Frame(viewer_window, bg=ModernTheme.COLORS['bg_primary'])
        close_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        close_button = ModernButton(close_frame, text="Close", 
                                   command=viewer_window.destroy,
                                   width=100, height=30,
                                   bg_color=ModernTheme.COLORS['accent_primary'])
        close_button.pack(side='right')
    
    def create_sample_display(self, parent, sample_num, sample):
        """Create display for individual sample"""
        sample_frame = GlassFrame(parent)
        sample_frame.pack(fill='x', padx=10, pady=5)
        
        # Sample header
        header = tk.Frame(sample_frame, bg=ModernTheme.COLORS['bg_secondary'])
        header.pack(fill='x', padx=10, pady=(10, 5))
        
        ttk.Label(header, text=f"Sample #{sample_num}", 
                 style='Title.TLabel').pack(side='left')
        
        # Sample info
        info_frame = tk.Frame(sample_frame, bg=ModernTheme.COLORS['bg_secondary'])
        info_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        # Left side - sample image (if available)
        left_frame = tk.Frame(info_frame, bg=ModernTheme.COLORS['bg_secondary'])
        left_frame.pack(side='left', padx=(0, 10))
        
        # Try to find and display sample image
        sample_image_path = self.find_sample_image(sample)
        if sample_image_path and Path(sample_image_path).exists():
            try:
                # Load and display image
                img = cv2.imread(sample_image_path)
                if img is not None:
                    img_resized = cv2.resize(img, (120, 120))
                    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    photo = ImageTk.PhotoImage(img_pil)
                    
                    img_label = tk.Label(left_frame, image=photo,
                                       bg=ModernTheme.COLORS['bg_tertiary'],
                                       relief='solid', borderwidth=1)
                    img_label.image = photo  # Keep reference
                    img_label.pack()
                else:
                    # Placeholder if image can't be loaded
                    placeholder = tk.Label(left_frame, text="Image\nNot Available",
                                         width=15, height=8,
                                         bg=ModernTheme.COLORS['bg_tertiary'],
                                         fg=ModernTheme.COLORS['text_secondary'],
                                         relief='solid', borderwidth=1)
                    placeholder.pack()
            except Exception as e:
                placeholder = tk.Label(left_frame, text="Image\nLoad Error",
                                     width=15, height=8,
                                     bg=ModernTheme.COLORS['bg_tertiary'],
                                     fg=ModernTheme.COLORS['error'],
                                     relief='solid', borderwidth=1)
                placeholder.pack()
        else:
            # Placeholder if no image path
            placeholder = tk.Label(left_frame, text="No Image\nAvailable",
                                 width=15, height=8,
                                 bg=ModernTheme.COLORS['bg_tertiary'],
                                 fg=ModernTheme.COLORS['text_secondary'],
                                 relief='solid', borderwidth=1)
            placeholder.pack()
        
        # Right side - sample details
        right_frame = tk.Frame(info_frame, bg=ModernTheme.COLORS['bg_secondary'])
        right_frame.pack(side='left', fill='both', expand=True)
        
        details_text = tk.Text(right_frame, height=8, width=50,
                              bg=ModernTheme.COLORS['bg_tertiary'],
                              fg=ModernTheme.COLORS['text_primary'],
                              font=('Consolas', 9),
                              relief='flat', borderwidth=1,
                              state='normal')
        
        # Create sample details
        feature_dim = len(sample['features']) if sample['features'] is not None else 0
        feature_hash = hash(tuple(sample['features'][:10])) if sample['features'] is not None and len(sample['features']) >= 10 else 0
        
        # Get feature preview (first 5 elements for readability)
        feature_preview = sample['features'][:5] if sample['features'] is not None else 'No features'
        
        details = f"""Sample Information:

Confidence: {sample['confidence']:.3f}
Created: {sample['created_at']}
Model Type: {sample.get('model_type', 'Unknown')}
Feature Dimension: {feature_dim}
Feature Hash: {feature_hash:08x}

Feature Vector Preview:
{feature_preview}

Image Path: {sample.get('image_path', 'Not saved')}
Status: {'✓ Valid' if sample['features'] is not None else '✗ Invalid'}"""
        
        details_text.insert('1.0', details)
        details_text.configure(state='disabled')
        details_text.pack(fill='both', expand=True)
    
    def find_sample_image(self, sample):
        """Try to find the sample image file"""
        # Check if image path is stored in database
        if sample.get('image_path') and Path(sample['image_path']).exists():
            return sample['image_path']
        
        # Try to find in samples directory
        sample_files = list(self.samples_dir.glob("*.jpg"))
        if sample_files:
            # Return the first found sample (could be improved with better matching)
            return str(sample_files[0])
        
        return None
    
    def delete_selected_person(self):
        """Delete selected person from database"""
        selection = self.db_persons_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a person to delete")
            return
        
        selected_text = self.db_persons_listbox.get(selection[0])
        person_name = selected_text.split(' (')[0]
        
        result = messagebox.askyesno("Confirm Deletion", 
            f"Are you sure you want to delete '{person_name}' and all their biometric data?\n\n"
            "This action cannot be undone!")
        
        if result:
            try:
                # Find person ID
                persons = self.database.get_persons()
                for person_id, name, _, _ in persons:
                    if name == person_name:
                        success = self.database.delete_person(person_id)
                        if success:
                            messagebox.showinfo("Success", f"Successfully deleted '{person_name}'")
                            self.update_database()
                        else:
                            messagebox.showerror("Error", f"Failed to delete '{person_name}'")
                        break
            except Exception as e:
                messagebox.showerror("Error", f"Deletion failed: {str(e)}")
    
    def export_database(self):
        """Export database to file"""
        messagebox.showinfo("Feature Coming Soon", 
            "Database export functionality will be implemented in a future update.")
    
    def import_database(self):
        """Import database from file"""
        messagebox.showinfo("Feature Coming Soon", 
            "Database import functionality will be implemented in a future update.")
    
    def clear_all_database(self):
        """Clear entire database"""
        persons = self.database.get_persons()
        if not persons:
            messagebox.showinfo("Info", "Database is already empty")
            return
        
        result = messagebox.askyesno("Confirm Clear All", 
            f"Are you sure you want to delete ALL {len(persons)} enrolled persons?\n\n"
            "⚠️ This action cannot be undone!\n"
            "All biometric data will be permanently lost!")
        
        if result:
            try:
                success = self.database.clear_all_persons()
                if success:
                    messagebox.showinfo("Success", "All persons have been cleared from the database")
                    self.update_database()
                else:
                    messagebox.showerror("Error", "Failed to clear database")
            except Exception as e:
                messagebox.showerror("Error", f"Clear operation failed: {str(e)}")


def main():
    """Main application entry point"""
    root = tk.Tk()
    app = EarBiometricsGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        if hasattr(app, 'camera') and app.camera:
            app.camera.release()


if __name__ == "__main__":
    main()
