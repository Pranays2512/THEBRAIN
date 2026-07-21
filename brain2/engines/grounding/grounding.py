#!/usr/bin/env python3
"""
grounding.py — Unified Grounding Pipeline

Consolidates the fragments of grounding logic:
1. grounding.py (category/symbol grounding)
2. ground_numeric.py (continuous quantity grounding)
3. ground_reason.py (perception -> inference loop)
4. ground_to_binding.py (perception -> C++ memory loop)

This provides a unified `GroundingPipeline` class that handles all grounding capabilities.
"""

import numpy as np
import brain2
from engines.reasoning.reasoning_engine import ReasoningEngine

class GroundingPipeline:
    def __init__(self, brain=None, n_dims=32):
        self.n_dims = n_dims
        self.brain = brain or brain2.Brain(som_rows=4, som_cols=4, n_dims=n_dims)
        self.rng = np.random.default_rng(42)
        
        # Numeric grounding state
        self.numeric_axes = {}
        self.numeric_decoders = {}
        
        # Categorical grounding state
        self.category_centroids = {}
        
        self.AXIS = self._vec("__magnitude__")
        
    def _vec(self, sym):
        r = np.random.default_rng(abs(hash(sym)) % (2 ** 32))
        v = r.standard_normal(self.n_dims).astype("float32")
        return v / np.linalg.norm(v)
        
    def _cos(self, a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na and nb else 0.0

    # ── Categorical Grounding ──

    def ground_categories(self, labeled_data):
        """
        Grounds categorical symbols based on labeled examples.
        labeled_data: list of tuples (vector, label_string)
        """
        acts = {}
        for v, k in labeled_data:
            acts.setdefault(k, []).append(np.asarray(self.brain.som.activation_map(v)))
        
        for k, v in acts.items():
            self.category_centroids[k] = np.mean(v, axis=0)
            
    def recognize_category(self, v):
        """Recognize a category from a raw vector."""
        if not self.category_centroids:
            return None
        a = np.asarray(self.brain.som.activation_map(v))
        return max(self.category_centroids, key=lambda k: self._cos(a, self.category_centroids[k]))

    # ── Numeric Grounding ──

    def calibrate_numeric_sensors(self, attributes, labeled_observations):
        """
        Calibrates numeric decoders from labeled observations.
        labeled_observations: list of dicts mapping attribute name to numeric value
        """
        for a in attributes:
            if a not in self.numeric_axes:
                self.numeric_axes[a] = self.rng.standard_normal(self.n_dims)
                
        # Generate encoded vectors for the observations
        V = []
        for obs in labeled_observations:
            v = sum(obs[a] * self.numeric_axes[a] for a in attributes) + 0.05 * self.rng.standard_normal(self.n_dims)
            V.append(v.astype("float32"))
            
        V = np.array(V)
        for a in attributes:
            y = np.array([obs[a] for obs in labeled_observations])
            self.numeric_decoders[a] = np.linalg.lstsq(V, y, rcond=None)[0]
            
    def decode_numeric(self, v, attributes=None):
        """Decode continuous quantities from a raw vector."""
        if attributes is None:
            attributes = list(self.numeric_decoders.keys())
        return {a: float(v @ self.numeric_decoders[a]) for a in attributes}

    # ── Memory Binding ──
    
    def bind_perceived_quantities(self, entity_name, decoded_vals):
        """Writes perceived/decoded numeric values into the C++ BindingMemory."""
        for a, val in decoded_vals.items():
            self.brain.binding.bind(
                self._vec(entity_name), 
                self._vec(a), 
                (val * self.AXIS).astype("float32")
            )
            
    def query_bound_quantity(self, entity_name, relation_name, conf_threshold=0.9):
        """Reads a numeric value from the C++ BindingMemory."""
        o, conf = self.brain.binding.query(self._vec(entity_name), self._vec(relation_name))
        if conf >= conf_threshold:
            return float(np.dot(np.asarray(o), self.AXIS))
        return None
