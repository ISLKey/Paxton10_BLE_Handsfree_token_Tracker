"""
BLE Enhanced Terminal Visual Mapping System - PHASE 2: INTERACTIVE FLOOR PLAN v3.3.0
Features: Drag & Drop Functionality, ESP32 Node Positioning, Real-Time Device Tracking
Python 3.13 Compatible with Enhanced Console Experience
"""

import os
import sys
import json
import threading
import signal
import atexit
import uuid
import base64
import io
import csv
import socket
import subprocess
import platform
import logging
import time
import math
import statistics
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string, send_file
from flask_cors import CORS
import paho.mqtt.client as mqtt
from collections import defaultdict, Counter, deque

# Enhanced Terminal Colors and Symbols
class TerminalColors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    ORANGE = '\033[38;5;208m'  # Orange color
    DARK_RED = '\033[38;5;88m'  # Dark red color
    GRAY = '\033[90m'  # Gray color
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    @staticmethod
    def colorize(text, color):
        return f"{color}{text}{TerminalColors.END}"

# Distance-based visual indicators
def get_distance_symbol_and_color(distance):
    """Get emoji symbol and color based on distance"""
    if distance < 1:
        return '🟢', '#00ff00', 'VERY_CLOSE'
    elif distance < 2:
        return '🟡', '#ffff00', 'CLOSE'
    elif distance < 4:
        return '🟠', '#ff8800', 'MEDIUM'
    elif distance < 6:
        return '🔴', '#ff0000', 'FAR'
    else:
        return '⚫', '#800000', 'VERY_FAR'

def get_device_name_and_emoji(device_id):
    """Get friendly device name and emoji based on device ID patterns"""
    device_id_lower = device_id.lower()
    
    # iPhone patterns
    if any(pattern in device_id_lower for pattern in ['iphone', 'ios', 'apple']):
        return '📱 iPhone', '📱'
    
    # Android patterns
    elif any(pattern in device_id_lower for pattern in ['android', 'samsung', 'pixel', 'lg', 'htc']):
        return '📱 Android', '📱'
    
    # Wearable patterns
    elif any(pattern in device_id_lower for pattern in ['watch', 'fitbit', 'garmin', 'apple watch']):
        return '⌚ Smartwatch', '⌚'
    
    # Tracker patterns
    elif any(pattern in device_id_lower for pattern in ['tile', 'airtag', 'tracker']):
        return '🏷️ Tracker', '🏷️'
    
    # Default
    else:
        return f'📟 Device-{device_id[-4:]}', '📟'

def print_banner():
    """Print enhanced startup banner"""
    print(f"\n{TerminalColors.CYAN}{'='*70}{TerminalColors.END}")
    print(f"{TerminalColors.BOLD}{TerminalColors.BLUE}🗺️  BLE TRIANGULATION ENHANCED VISUAL MAPPING SYSTEM{TerminalColors.END}")
    print(f"{TerminalColors.CYAN}{'='*70}{TerminalColors.END}")
    print(f"{TerminalColors.GREEN}✨ REAL-TIME TRIANGULATION WITH MULTIPLE ESP32 NODES{TerminalColors.END}")
    print(f"{TerminalColors.YELLOW}📍 Accurate Device Positioning Using Circle Intersections{TerminalColors.END}")
    print(f"{TerminalColors.MAGENTA}🔴 Multi-Node Distance Measurements for Precision{TerminalColors.END}")
    print(f"{TerminalColors.BLUE}📈 Trilateration with Least Squares Error Correction{TerminalColors.END}")
    print(f"{TerminalColors.WHITE}🏢 Interactive Floor Plan Drawing Tools{TerminalColors.END}")
    print(f"{TerminalColors.CYAN}🎯 Enhanced Positioning Accuracy with 2+ Nodes{TerminalColors.END}")
    print(f"{TerminalColors.CYAN}{'='*70}{TerminalColors.END}")

# Global variables
app = Flask(__name__)
mqtt_client = None
devices_data = {}
esp32_nodes = {}
device_positions = {}  # Store calculated device positions
movement_trails = {}   # Store movement history for trails
prediction_data = {}   # Store data for predictive paths

# Enhanced triangulation data storage
device_node_data = defaultdict(dict)  # {device_id: {node_id: {distance, confidence, timestamp}}}
triangulation_timeout = 5  # seconds to keep node data for triangulation

# Database setup
class Database:
    def __init__(self, db_path='ble_tracking.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with all required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Settings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_mac TEXT UNIQUE,
                name TEXT,
                avatar_color TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # ESP32 nodes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS esp32_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT UNIQUE,
                name TEXT,
                room_name TEXT,
                x_position REAL DEFAULT 0,
                y_position REAL DEFAULT 0,
                max_distance REAL DEFAULT 16.0,
                floor_plan_id INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Add max_distance column if it doesn't exist (for existing databases)
        try:
            cursor.execute('ALTER TABLE esp32_nodes ADD COLUMN max_distance REAL DEFAULT 16.0')
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Floor plans table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS floor_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                data TEXT,
                scale_factor REAL DEFAULT 50.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Device positions table for real-time tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS device_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT,
                x_position REAL,
                y_position REAL,
                confidence REAL,
                esp32_node TEXT,
                distance REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Movement trails table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS movement_trails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT,
                x_position REAL,
                y_position REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_setting(self, key, default=None):
        """Get setting value"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT value FROM settings WHERE key = ?', (key,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else default
    
    def set_setting(self, key, value):
        """Set setting value"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)', (key, value))
        conn.commit()
        conn.close()
    
    def get_users(self):
        """Get all users"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, device_mac, name, avatar_color, created_at FROM users ORDER BY name')
        users = []
        for row in cursor.fetchall():
            users.append({
                'id': row[0],
                'device_mac': row[1],
                'name': row[2],
                'avatar_color': row[3],
                'created_at': row[4]
            })
        conn.close()
        return users
    
    def add_user(self, device_mac, name, avatar_color):
        """Add new user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (device_mac, name, avatar_color) VALUES (?, ?, ?)', 
                         (device_mac, name, avatar_color))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False
    
    def delete_user(self, user_id):
        """Delete user by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success
    
    def get_esp32_nodes(self):
        """Get all ESP32 nodes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, node_id, name, room_name, x_position, y_position, max_distance, floor_plan_id, created_at FROM esp32_nodes ORDER BY name')
        nodes = []
        for row in cursor.fetchall():
            nodes.append({
                'id': row[0],
                'node_id': row[1],
                'name': row[2],
                'room_name': row[3],
                'x_position': row[4],
                'y_position': row[5],
                'max_distance': row[6] or 16.0,  # Default to 16m if None
                'floor_plan_id': row[7],
                'created_at': row[8]
            })
        conn.close()
        return nodes
    
    def update_esp32_node_by_id(self, node_db_id, **kwargs):
        """Update ESP32 node by database ID"""
        if not kwargs:
            return False
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build dynamic update query
        set_clauses = []
        values = []
        for key, value in kwargs.items():
            if key in ['name', 'room_name', 'x_position', 'y_position', 'max_distance', 'floor_plan_id']:
                set_clauses.append(f'{key} = ?')
                values.append(value)
        
        if not set_clauses:
            conn.close()
            return False
        
        values.append(node_db_id)
        query = f'UPDATE esp32_nodes SET {", ".join(set_clauses)} WHERE id = ?'
        
        cursor.execute(query, values)
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success
    
    def add_or_update_esp32_node(self, node_id, name=None, room_name=None, x_position=0, y_position=0, max_distance=16.0):
        """Add or update ESP32 node"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if node exists
        cursor.execute('SELECT id FROM esp32_nodes WHERE node_id = ?', (node_id,))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing node
            cursor.execute('''
                UPDATE esp32_nodes 
                SET name = COALESCE(?, name), 
                    room_name = COALESCE(?, room_name),
                    x_position = COALESCE(?, x_position),
                    y_position = COALESCE(?, y_position),
                    max_distance = COALESCE(?, max_distance)
                WHERE node_id = ?
            ''', (name, room_name, x_position, y_position, max_distance, node_id))
        else:
            # Add new node
            cursor.execute('''
                INSERT INTO esp32_nodes (node_id, name, room_name, x_position, y_position, max_distance) 
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (node_id, name or f'ESP32-{node_id}', room_name or 'Unassigned', x_position, y_position, max_distance))
        
        conn.commit()
        conn.close()
        return True
    
    def save_device_position(self, device_id, x_position, y_position, confidence, esp32_node, distance):
        """Save device position for real-time tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO device_positions (device_id, x_position, y_position, confidence, esp32_node, distance)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (device_id, x_position, y_position, confidence, esp32_node, distance))
        conn.commit()
        conn.close()
    
    def add_movement_trail(self, device_id, x_position, y_position):
        """Add point to movement trail"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO movement_trails (device_id, x_position, y_position)
            VALUES (?, ?, ?)
        ''', (device_id, x_position, y_position))
        
        # Keep only last 50 trail points per device
        cursor.execute('''
            DELETE FROM movement_trails 
            WHERE device_id = ? AND id NOT IN (
                SELECT id FROM movement_trails 
                WHERE device_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 50
            )
        ''', (device_id, device_id))
        
        conn.commit()
        conn.close()
    
    def get_movement_trails(self, device_id=None):
        """Get movement trails for devices"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if device_id:
            cursor.execute('''
                SELECT device_id, x_position, y_position, timestamp 
                FROM movement_trails 
                WHERE device_id = ? 
                ORDER BY timestamp DESC LIMIT 50
            ''', (device_id,))
        else:
            cursor.execute('''
                SELECT device_id, x_position, y_position, timestamp 
                FROM movement_trails 
                ORDER BY device_id, timestamp DESC
            ''')
        
        trails = {}
        for row in cursor.fetchall():
            device_id = row[0]
            if device_id not in trails:
                trails[device_id] = []
            trails[device_id].append({
                'x': row[1],
                'y': row[2],
                'timestamp': row[3]
            })
        
        conn.close()
        return trails

# Initialize database
db = Database()

def calculate_device_position(device_data):
    """Calculate device position using weighted triangulation that prioritizes stronger signals"""
    try:
        device_id = device_data.get('device_id')
        current_esp32_node = device_data.get('esp32_node')
        current_distance = float(device_data.get('distance', 0))
        current_confidence = float(device_data.get('confidence', 0))
        current_time = time.time()
        
        # Store current node data
        device_node_data[device_id][current_esp32_node] = {
            'distance': current_distance,
            'confidence': current_confidence,
            'timestamp': current_time
        }
        
        # Clean old data (remove data older than triangulation_timeout)
        for node_id in list(device_node_data[device_id].keys()):
            if current_time - device_node_data[device_id][node_id]['timestamp'] > triangulation_timeout:
                del device_node_data[device_id][node_id]
        
        # Get ESP32 node positions from database
        nodes = db.get_esp32_nodes()
        node_positions = {}
        for node in nodes:
            node_positions[node['node_id']] = (node['x_position'], node['y_position'])
        
        # Get available node data for this device
        available_nodes = []
        for node_id, node_data in device_node_data[device_id].items():
            if node_id in node_positions:
                available_nodes.append({
                    'node_id': node_id,
                    'position': node_positions[node_id],
                    'distance': node_data['distance'],
                    'confidence': node_data['confidence']
                })
        
        # Sort by distance (closest first) for priority weighting
        available_nodes.sort(key=lambda x: x['distance'])
        
        print(f"🎯 Triangulation for {device_id}: {len(available_nodes)} nodes available")
        for node in available_nodes:
            print(f"   📡 {node['node_id']}: {node['distance']:.1f}m (conf: {node['confidence']:.1f})")
        
        if len(available_nodes) == 0:
            # No nodes available, use default position
            return 400, 300, 0
            
        elif len(available_nodes) == 1:
            # Single node - position around the node
            node = available_nodes[0]
            distance_pixels = node['distance'] * 50  # Convert to pixels
            
            # Use device ID hash for consistent angle, but add some randomness
            base_angle = hash(device_id) % 360
            angle_variation = (hash(device_id + str(int(current_time/10))) % 60) - 30  # ±30 degrees
            angle = base_angle + angle_variation
            angle_rad = math.radians(angle)
            
            x = node['position'][0] + distance_pixels * math.cos(angle_rad)
            y = node['position'][1] + distance_pixels * math.sin(angle_rad)
            
            confidence = node['confidence']
            print(f"📍 Single-node positioning: ({x:.1f}, {y:.1f})")
            
        elif len(available_nodes) == 2:
            # Two nodes - use weighted approach based on signal strength
            node1, node2 = available_nodes[0], available_nodes[1]  # node1 is closer
            
            # Calculate distance ratio - if one is much closer, prioritize it
            distance_ratio = node2['distance'] / node1['distance'] if node1['distance'] > 0 else 10
            confidence_ratio = node1['confidence'] / node2['confidence'] if node2['confidence'] > 0 else 1
            
            print(f"📊 Distance ratio: {distance_ratio:.2f}, Confidence ratio: {confidence_ratio:.2f}")
            
            # If one node is significantly closer (3x or more), use weighted positioning instead of intersection
            if distance_ratio >= 3.0 or confidence_ratio >= 2.0:
                # Heavily weight the closer/stronger node
                weight1 = 0.8  # 80% weight to closer node
                weight2 = 0.2  # 20% weight to farther node
                
                # Calculate weighted position
                x1, y1 = node1['position']
                x2, y2 = node2['position']
                
                # Position closer to the stronger node
                distance_pixels1 = node1['distance'] * 50
                angle = hash(device_id) % 360
                angle_rad = math.radians(angle)
                
                # Base position around stronger node
                base_x = x1 + distance_pixels1 * math.cos(angle_rad)
                base_y = y1 + distance_pixels1 * math.sin(angle_rad)
                
                # Slight adjustment toward weaker node
                x = base_x * weight1 + x2 * weight2
                y = base_y * weight1 + y2 * weight2
                
                confidence = node1['confidence'] * weight1 + node2['confidence'] * weight2
                print(f"📍 Weighted positioning (strong signal priority): ({x:.1f}, {y:.1f})")
                
            else:
                # Distances are similar, use intersection method
                x1, y1 = node1['position']
                x2, y2 = node2['position']
                r1 = node1['distance'] * 50  # Convert to pixels
                r2 = node2['distance'] * 50
                
                # Calculate intersection points
                intersections = calculate_circle_intersections(x1, y1, r1, x2, y2, r2)
                
                if intersections:
                    # Choose the intersection point (for now, take the first one)
                    x, y = intersections[0]
                    confidence = (node1['confidence'] + node2['confidence']) / 2
                    print(f"📍 Two-node intersection: ({x:.1f}, {y:.1f})")
                else:
                    # Circles don't intersect, use weighted average position
                    total_weight = node1['confidence'] + node2['confidence']
                    x = (x1 * node1['confidence'] + x2 * node2['confidence']) / total_weight
                    y = (y1 * node1['confidence'] + y2 * node2['confidence']) / total_weight
                    confidence = total_weight / 2
                    print(f"⚠️ No intersection, using weighted average: ({x:.1f}, {y:.1f})")
                
        else:
            # Three or more nodes - check if one is significantly stronger
            strongest_node = available_nodes[0]  # Already sorted by distance
            other_nodes = available_nodes[1:]
            
            # Calculate if strongest node is dominant
            avg_other_distance = sum(node['distance'] for node in other_nodes) / len(other_nodes)
            dominance_ratio = avg_other_distance / strongest_node['distance'] if strongest_node['distance'] > 0 else 1
            
            if dominance_ratio >= 4.0:  # One node is much stronger
                # Use weighted approach favoring the dominant node
                total_weight = sum(1.0 / (node['distance'] + 0.1) for node in available_nodes)  # Inverse distance weighting
                
                x = sum(node['position'][0] / (node['distance'] + 0.1) for node in available_nodes) / total_weight
                y = sum(node['position'][1] / (node['distance'] + 0.1) for node in available_nodes) / total_weight
                confidence = sum(node['confidence'] / (node['distance'] + 0.1) for node in available_nodes) / total_weight
                
                print(f"📍 Weighted multi-node (dominant signal): ({x:.1f}, {y:.1f})")
            else:
                # Use trilateration with least squares
                x, y, confidence = trilaterate_least_squares(available_nodes)
                print(f"🎯 Multi-node trilateration: ({x:.1f}, {y:.1f}) confidence: {confidence:.1f}")
        
        # Keep within canvas bounds
        x = max(50, min(750, x))
        y = max(50, min(550, y))
        
        return x, y, confidence
        
    except Exception as e:
        print(f"❌ Error in triangulation: {e}")
        return 400, 300, 0

def calculate_circle_intersections(x1, y1, r1, x2, y2, r2):
    """Calculate intersection points of two circles"""
    try:
        # Distance between centers
        d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Check if circles intersect
        if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
            return None
        
        # Calculate intersection points
        a = (r1**2 - r2**2 + d**2) / (2 * d)
        h = math.sqrt(r1**2 - a**2)
        
        # Point on line between centers
        px = x1 + a * (x2 - x1) / d
        py = y1 + a * (y2 - y1) / d
        
        # Intersection points
        x3 = px + h * (y2 - y1) / d
        y3 = py - h * (x2 - x1) / d
        x4 = px - h * (y2 - y1) / d
        y4 = py + h * (x2 - x1) / d
        
        return [(x3, y3), (x4, y4)]
        
    except:
        return None

def trilaterate_least_squares(nodes):
    """Trilaterate position using least squares method for 3+ nodes"""
    try:
        if len(nodes) < 3:
            return 400, 300, 0
        
        # Set up matrices for least squares solution
        # We solve: A * [x, y] = b
        A = []
        b = []
        
        # Use first node as reference
        x1, y1 = nodes[0]['position']
        r1 = nodes[0]['distance'] * 50
        
        for i in range(1, len(nodes)):
            xi, yi = nodes[i]['position']
            ri = nodes[i]['distance'] * 50
            
            # Linear equation: 2(xi-x1)*x + 2(yi-y1)*y = xi²+yi²-ri² - (x1²+y1²-r1²)
            A.append([2*(xi-x1), 2*(yi-y1)])
            b.append(xi**2 + yi**2 - ri**2 - (x1**2 + y1**2 - r1**2))
        
        # Solve using least squares (simplified)
        if len(A) >= 2:
            # Use first two equations for basic solution
            a11, a12 = A[0]
            a21, a22 = A[1]
            b1, b2 = b[0], b[1]
            
            # Solve 2x2 system
            det = a11*a22 - a12*a21
            if abs(det) > 1e-10:
                x = (b1*a22 - b2*a12) / det
                y = (a11*b2 - a21*b1) / det
                
                # Calculate confidence based on how well the solution fits all nodes
                total_error = 0
                for node in nodes:
                    nx, ny = node['position']
                    expected_dist = math.sqrt((x-nx)**2 + (y-ny)**2) / 50  # Convert back to meters
                    actual_dist = node['distance']
                    error = abs(expected_dist - actual_dist)
                    total_error += error
                
                # Convert error to confidence (lower error = higher confidence)
                avg_error = total_error / len(nodes)
                confidence = max(0, 100 - avg_error * 20)  # Scale error to confidence
                
                return x, y, confidence
        
        # Fallback to weighted average if least squares fails
        total_weight = sum(node['confidence'] for node in nodes)
        x = sum(node['position'][0] * node['confidence'] for node in nodes) / total_weight
        y = sum(node['position'][1] * node['confidence'] for node in nodes) / total_weight
        confidence = total_weight / len(nodes)
        
        return x, y, confidence
        
    except Exception as e:
        print(f"❌ Trilateration error: {e}")
        return 400, 300, 0

def update_movement_trail(device_id, x, y):
    """Update movement trail for a device"""
    if device_id not in movement_trails:
        movement_trails[device_id] = deque(maxlen=20)  # Keep last 20 positions
    
    # Only add if position changed significantly
    if not movement_trails[device_id] or \
       (abs(movement_trails[device_id][-1]['x'] - x) > 5 or 
        abs(movement_trails[device_id][-1]['y'] - y) > 5):
        
        movement_trails[device_id].append({
            'x': x,
            'y': y,
            'timestamp': datetime.now().isoformat()
        })
        
        # Save to database
        db.add_movement_trail(device_id, x, y)

def predict_next_position(device_id, current_x, current_y):
    """Predict next position based on movement history"""
    if device_id not in movement_trails or len(movement_trails[device_id]) < 3:
        return None
    
    trail = list(movement_trails[device_id])
    
    # Calculate velocity from last few positions
    recent_positions = trail[-3:]
    
    # Simple linear prediction
    dx = recent_positions[-1]['x'] - recent_positions[-2]['x']
    dy = recent_positions[-1]['y'] - recent_positions[-2]['y']
    
    # Predict position 5 seconds ahead (assuming 1 update per second)
    predicted_x = current_x + dx * 5
    predicted_y = current_y + dy * 5
    
    return {'x': predicted_x, 'y': predicted_y}

def send_max_distance_mqtt(node_db_id, max_distance):
    """Send max distance setting to ESP32 via MQTT"""
    try:
        # Get the node_id from database
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT node_id FROM esp32_nodes WHERE id = ?', (node_db_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            node_id = result[0]
            topic = f"espresense/rooms/{node_id}/max_distance/set"
            
            # Use the global MQTT client if available
            if 'mqtt_client' in globals() and mqtt_client:
                mqtt_client.publish(topic, str(int(max_distance)))
                print(f"{TerminalColors.GREEN}📡 MQTT: Set max distance {max_distance}m for node {node_id}{TerminalColors.END}")
            else:
                print(f"{TerminalColors.YELLOW}⚠️ MQTT client not available to send max distance command{TerminalColors.END}")
    except Exception as e:
        print(f"{TerminalColors.RED}❌ Error sending max distance MQTT: {e}{TerminalColors.END}")

def get_network_info():
    """Get current network information"""
    try:
        # Get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        
        return {
            'local_ip': local_ip,
            'subnet_mask': '255.255.255.0',  # Default
            'gateway': '.'.join(local_ip.split('.')[:-1]) + '.1',  # Assume .1 gateway
            'dns_servers': ['8.8.8.8', '8.8.4.4']
        }
    except:
        return {
            'local_ip': '127.0.0.1',
            'subnet_mask': '255.255.255.0',
            'gateway': '127.0.0.1',
            'dns_servers': ['8.8.8.8', '8.8.4.4']
        }

def on_connect(client, userdata, flags, rc):
    """MQTT connection callback"""
    if rc == 0:
        print(f"{TerminalColors.GREEN}✅ MQTT CONNECTED{TerminalColors.END}")
        # Subscribe only to device topics, not room telemetry
        client.subscribe("espresense/devices/+/+")
        print(f"{TerminalColors.CYAN}📡 Subscribed to espresense device topics only{TerminalColors.END}")
    else:
        print(f"{TerminalColors.RED}❌ MQTT CONNECTION FAILED: {rc}{TerminalColors.END}")

def on_message(client, userdata, msg):
    """MQTT message callback with enhanced terminal logging"""
    try:
        topic_parts = msg.topic.split('/')
        # Now we expect: espresense/devices/[device_id]/[esp32_node]
        if len(topic_parts) >= 4 and topic_parts[1] == 'devices':
            device_id = topic_parts[2]
            esp32_node = topic_parts[3]
            
            # Parse message
            try:
                data = json.loads(msg.payload.decode())
                distance = data.get('distance', 0)
                confidence = data.get('confidence', 0)
                rssi = data.get('rssi', None)  # Get RSSI value if available
            except:
                distance = float(msg.payload.decode()) if msg.payload.decode().replace('.', '').isdigit() else 0
                confidence = 85  # Default confidence
                rssi = None  # No RSSI available in simple format
            
            # Get device name and emoji
            device_name, device_emoji = get_device_name_and_emoji(device_id)
            
            # Get distance symbol and color
            distance_symbol, distance_color, distance_status = get_distance_symbol_and_color(distance)
            
            # Store device data
            devices_data[device_id] = {
                'device_id': device_id,
                'device_name': device_name,
                'device_emoji': device_emoji,
                'esp32_node': esp32_node,
                'distance': distance,
                'confidence': confidence,
                'rssi': rssi,  # Store RSSI value
                'distance_symbol': distance_symbol,
                'distance_color': distance_color,
                'distance_status': distance_status,
                'last_seen': datetime.now().isoformat()
            }
            
            # Calculate device position
            x, y, pos_confidence = calculate_device_position(devices_data[device_id])
            
            # Update position data
            device_positions[device_id] = {
                'x': x,
                'y': y,
                'confidence': pos_confidence,
                'esp32_node': esp32_node,
                'distance': distance
            }
            
            # Update movement trail
            update_movement_trail(device_id, x, y)
            
            # Save to database
            db.save_device_position(device_id, x, y, pos_confidence, esp32_node, distance)
            
            # Store ESP32 node info
            if esp32_node not in esp32_nodes:
                esp32_nodes[esp32_node] = {
                    'node_id': esp32_node,
                    'last_seen': datetime.now().isoformat(),
                    'device_count': 0
                }
                
                # Add to database
                db.add_or_update_esp32_node(esp32_node)
                
                print(f"\n{TerminalColors.BOLD}{TerminalColors.BLUE}📡 NEW ESP32 NODE DETECTED{TerminalColors.END}")
                print(f"   {TerminalColors.CYAN}Node ID:{TerminalColors.END} {esp32_node}")
                print(f"   {TerminalColors.CYAN}Name:{TerminalColors.END} ESP32-{esp32_node}")
                print(f"   {TerminalColors.CYAN}Position:{TerminalColors.END} (400, 300) [Default]")
                print(f"{TerminalColors.CYAN}{'─'*50}{TerminalColors.END}")
            
            esp32_nodes[esp32_node]['last_seen'] = datetime.now().isoformat()
            
            # Enhanced terminal output
            current_time = datetime.now().strftime("%H:%M:%S")
            
            print(f"\n{TerminalColors.BOLD}{TerminalColors.GREEN}📍 DEVICE TRACKING UPDATE{TerminalColors.END}")
            print(f"   {TerminalColors.CYAN}Time:{TerminalColors.END} {current_time}")
            print(f"   {TerminalColors.CYAN}Device:{TerminalColors.END} {device_emoji} {device_name}")
            print(f"   {TerminalColors.CYAN}ESP32 Node:{TerminalColors.END} {esp32_node}")
            print(f"   {TerminalColors.CYAN}Distance:{TerminalColors.END} {distance_symbol} {distance:.1f}m ({distance_status})")
            print(f"   {TerminalColors.CYAN}Confidence:{TerminalColors.END} {confidence}%")
            
            # Add RSSI to logging if available
            if rssi is not None:
                # Color code RSSI values for easy reading
                if rssi >= -50:
                    rssi_color = TerminalColors.GREEN  # Excellent signal
                    rssi_status = "Excellent"
                elif rssi >= -60:
                    rssi_color = TerminalColors.YELLOW  # Good signal
                    rssi_status = "Good"
                elif rssi >= -70:
                    rssi_color = TerminalColors.ORANGE  # Fair signal
                    rssi_status = "Fair"
                elif rssi >= -80:
                    rssi_color = TerminalColors.RED  # Poor signal
                    rssi_status = "Poor"
                else:
                    rssi_color = TerminalColors.DARK_RED  # Very poor signal
                    rssi_status = "Very Poor"
                
                print(f"   {TerminalColors.CYAN}RSSI:{TerminalColors.END} {rssi_color}{rssi} dBm ({rssi_status}){TerminalColors.END}")
            else:
                print(f"   {TerminalColors.CYAN}RSSI:{TerminalColors.END} {TerminalColors.GRAY}Not Available{TerminalColors.END}")
            
            print(f"   {TerminalColors.CYAN}Position:{TerminalColors.END} ({x:.0f}, {y:.0f})")
            
            # Show movement if available
            if device_id in movement_trails and len(movement_trails[device_id]) > 1:
                prev_pos = movement_trails[device_id][-2]
                movement_distance = math.sqrt((x - prev_pos['x'])**2 + (y - prev_pos['y'])**2)
                print(f"   {TerminalColors.CYAN}Movement:{TerminalColors.END} 📈 {movement_distance:.0f} pixels")
                
                # Show prediction
                prediction = predict_next_position(device_id, x, y)
                if prediction:
                    print(f"   {TerminalColors.CYAN}Predicted:{TerminalColors.END} 🔮 ({prediction['x']:.0f}, {prediction['y']:.0f})")
            
            print(f"{TerminalColors.CYAN}{'─'*50}{TerminalColors.END}")
            
    except Exception as e:
        print(f"{TerminalColors.RED}❌ Error processing MQTT message: {e}{TerminalColors.END}")

def start_mqtt_client():
    """Start MQTT client"""
    global mqtt_client
    
    try:
        network_info = get_network_info()
        broker_ip = db.get_setting('mqtt_broker_ip', network_info['local_ip'])
        broker_port = int(db.get_setting('mqtt_broker_port', '1883'))
        
        print(f"{TerminalColors.YELLOW}🔄 CONNECTING TO MQTT BROKER{TerminalColors.END}")
        print(f"   {TerminalColors.CYAN}Broker:{TerminalColors.END} {broker_ip}:{broker_port}")
        
        mqtt_client = mqtt.Client()
        mqtt_client.on_connect = on_connect
        mqtt_client.on_message = on_message
        
        mqtt_client.connect(broker_ip, broker_port, 60)
        mqtt_client.loop_start()
        
        return True
    except Exception as e:
        print(f"{TerminalColors.RED}❌ MQTT connection failed: {e}{TerminalColors.END}")
        return False

# Flask routes
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/stats')
def get_stats():
    """Get system statistics"""
    timeout_seconds = int(db.get_setting('device_timeout_seconds', '120'))
    
    active_devices = 0
    for device_data in devices_data.values():
        if 'last_seen' in device_data:
            try:
                last_seen_dt = datetime.fromisoformat(device_data['last_seen'])
                time_since_seen = datetime.now() - last_seen_dt
                if time_since_seen.total_seconds() <= timeout_seconds:
                    active_devices += 1
            except:
                pass
    
    users = db.get_users()
    esp32_nodes_list = db.get_esp32_nodes()
    
    return jsonify({
        'success': True,
        'stats': {
            'total_devices': active_devices,
            'total_users': len(users),
            'esp32_nodes': len(esp32_nodes_list),
            'total_rooms': len(set(node['room_name'] for node in esp32_nodes_list if node['room_name'] != 'Unassigned'))
        }
    })

@app.route('/api/devices/real-time')
def get_real_time_devices():
    """Get real-time device data with positions"""
    timeout_seconds = int(db.get_setting('device_timeout_seconds', '120'))
    
    active_devices = []
    for device_id, device_data in devices_data.items():
        if 'last_seen' in device_data:
            try:
                last_seen_dt = datetime.fromisoformat(device_data['last_seen'])
                time_since_seen = datetime.now() - last_seen_dt
                if time_since_seen.total_seconds() <= timeout_seconds:
                    # Add position data
                    device_with_position = device_data.copy()
                    if device_id in device_positions:
                        device_with_position.update(device_positions[device_id])
                    
                    # Add movement trail
                    if device_id in movement_trails:
                        device_with_position['trail'] = list(movement_trails[device_id])
                    
                    # Add prediction
                    if device_id in device_positions:
                        pos = device_positions[device_id]
                        prediction = predict_next_position(device_id, pos['x'], pos['y'])
                        if prediction:
                            device_with_position['prediction'] = prediction
                    
                    # Check if device is assigned to a user
                    users = db.get_users()
                    device_with_position['is_assigned'] = any(user['device_mac'] == device_id for user in users)
                    
                    active_devices.append(device_with_position)
            except:
                pass
    
    return jsonify({
        'success': True, 
        'devices': active_devices, 
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/device_node_data')
def get_device_node_data():
    """Get current device-node data for signal strength visualization"""
    try:
        # Convert defaultdict to regular dict for JSON serialization
        data = {}
        for device_id, node_data in device_node_data.items():
            data[device_id] = dict(node_data)
        
        return jsonify({
            'success': True,
            'device_node_data': data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/esp32-nodes')
def get_esp32_nodes():
    """Get ESP32 nodes with positions"""
    nodes = db.get_esp32_nodes()
    return jsonify({'success': True, 'esp32_nodes': nodes})

@app.route('/api/esp32-nodes/<int:node_db_id>', methods=['PUT'])
def update_esp32_node(node_db_id):
    """Update ESP32 node position and details"""
    try:
        data = request.get_json()
        
        update_data = {}
        if 'name' in data:
            update_data['name'] = data['name']
        if 'room_name' in data:
            update_data['room_name'] = data['room_name']
        if 'x_position' in data:
            update_data['x_position'] = data['x_position']
        if 'y_position' in data:
            update_data['y_position'] = data['y_position']
        if 'max_distance' in data:
            update_data['max_distance'] = float(data['max_distance'])
        if 'floor_plan_id' in data:
            update_data['floor_plan_id'] = data['floor_plan_id']
        
        success = db.update_esp32_node_by_id(node_db_id, **update_data)
        
        if success:
            # If max_distance was updated, send MQTT command to ESP32
            if 'max_distance' in data:
                send_max_distance_mqtt(node_db_id, data['max_distance'])
            
            return jsonify({'success': True, 'message': 'ESP32 node updated successfully!'})
        else:
            return jsonify({'success': False, 'error': 'Node not found or update failed'}), 404
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/movement-trails')
def get_movement_trails():
    """Get movement trails for all devices"""
    trails = db.get_movement_trails()
    return jsonify({'success': True, 'trails': trails})

@app.route('/api/movement-trails/<device_id>')
def get_device_trail(device_id):
    """Get movement trail for specific device"""
    trails = db.get_movement_trails(device_id)
    return jsonify({'success': True, 'trail': trails.get(device_id, [])})

# User management routes (from original)
@app.route('/api/users')
def get_users():
    users = db.get_users()
    return jsonify({'success': True, 'users': users})

@app.route('/api/users', methods=['POST'])
def add_user():
    try:
        data = request.get_json()
        device_mac = data.get('device_mac', '').strip()
        name = data.get('name', '').strip()
        avatar_color = data.get('avatar_color', '#667eea')
        
        if not device_mac or not name:
            return jsonify({'success': False, 'error': 'Device MAC and name are required'}), 400
        
        success = db.add_user(device_mac, name, avatar_color)
        
        if success:
            return jsonify({'success': True, 'message': 'User added successfully!'})
        else:
            return jsonify({'success': False, 'error': 'Device MAC already exists'}), 400
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        success = db.delete_user(user_id)
        
        if success:
            return jsonify({'success': True, 'message': 'User deleted successfully!'})
        else:
            return jsonify({'success': False, 'error': 'User not found'}), 404
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/devices/list')
def get_devices_list():
    """Get list of all detected BLE devices for dropdown"""
    timeout_seconds = int(db.get_setting('device_timeout_seconds', '120'))
    
    device_list = []
    for device_id, device_data in devices_data.items():
        if 'last_seen' in device_data:
            try:
                last_seen_dt = datetime.fromisoformat(device_data['last_seen'])
                time_since_seen = datetime.now() - last_seen_dt
                is_active = time_since_seen.total_seconds() <= timeout_seconds
                
                device_list.append({
                    'device_id': device_id,
                    'device_mac': device_id,
                    'device_name': device_data.get('device_name', device_id),
                    'esp32_node': device_data.get('esp32_node', 'Unknown'),
                    'distance': device_data.get('distance', 0),
                    'confidence': device_data.get('confidence', 0),
                    'distance_color': device_data.get('distance_color', '#666'),
                    'distance_symbol': device_data.get('distance_symbol', '📟'),
                    'is_active': is_active,
                    'last_seen': device_data['last_seen']
                })
            except:
                pass
    
    device_list.sort(key=lambda x: (not x['is_active'], x['device_id']))
    return jsonify({'success': True, 'devices': device_list})

# Settings routes (from original)
@app.route('/api/settings')
def get_settings():
    """Get all settings"""
    settings = {
        'device_timeout_seconds': db.get_setting('device_timeout_seconds', '120'),
        'mqtt_broker_ip': db.get_setting('mqtt_broker_ip', '127.0.0.1'),
        'mqtt_broker_port': db.get_setting('mqtt_broker_port', '1883')
    }
    return jsonify({'success': True, 'settings': settings})

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update settings"""
    try:
        data = request.get_json()
        
        for key, value in data.items():
            if key in ['device_timeout_seconds', 'mqtt_broker_ip', 'mqtt_broker_port']:
                db.set_setting(key, str(value))
        
        return jsonify({'success': True, 'message': 'Settings updated successfully!'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# HTML Template with Phase 2 Features - FIXED VERSION
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BLE Enhanced Terminal Visual Mapping System - Phase 2 Fixed</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .header { 
            background: rgba(255,255,255,0.95); 
            padding: 1rem; 
            text-align: center; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .header h1 { 
            color: #2c3e50; 
            font-size: 1.8rem; 
            margin-bottom: 0.5rem;
        }
        .status-badge { 
            background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
            color: white; 
            padding: 0.5rem 1rem; 
            border-radius: 15px;
            display: inline-block;
            font-size: 0.8rem;
        }
        .container { 
            max-width: 1800px; 
            margin: 1rem auto; 
            padding: 0 1rem; 
        }
        .nav-tabs {
            display: flex;
            background: rgba(255,255,255,0.95);
            border-radius: 10px 10px 0 0;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        .nav-tab {
            flex: 1;
            padding: 1rem;
            text-align: center;
            cursor: pointer;
            background: rgba(255,255,255,0.8);
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .nav-tab.active {
            background: white;
            border-bottom-color: #667eea;
            color: #667eea;
        }
        .nav-tab:hover {
            background: rgba(255,255,255,0.9);
        }
        .tab-content {
            display: none;
            background: rgba(255,255,255,0.95);
            border-radius: 0 0 10px 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            min-height: 70vh;
        }
        .tab-content.active {
            display: block;
        }
        .stats-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); 
            gap: 1rem; 
            margin-bottom: 1.5rem; 
        }
        .stat-card { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            padding: 1.5rem; 
            border-radius: 10px; 
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .stat-number { font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem; }
        .stat-label { font-size: 0.9rem; opacity: 0.9; }
        
        /* Visual Mapping Styles */
        .mapping-container {
            display: flex;
            height: 80vh;
            gap: 1rem;
        }
        .mapping-sidebar {
            width: 300px;
            background: #f8f9fa;
            border-radius: 10px;
            padding: 1rem;
            overflow-y: auto;
        }
        .mapping-canvas-container {
            flex: 1;
            background: white;
            border-radius: 10px;
            padding: 1rem;
            position: relative;
        }
        #visualMappingCanvas {
            border: 2px solid #ddd;
            border-radius: 5px;
            cursor: crosshair;
        }
        .layer-controls {
            margin-bottom: 1rem;
        }
        .layer-control {
            display: flex;
            align-items: center;
            margin: 0.5rem 0;
        }
        .layer-control input[type="checkbox"] {
            margin-right: 0.5rem;
        }
        .device-legend {
            margin-top: 1rem;
            padding: 1rem;
            background: #e9ecef;
            border-radius: 5px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin: 0.3rem 0;
        }
        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }
        
        /* Interactive Floor Plan Styles */
        .floorplan-container {
            display: flex;
            height: 80vh;
            gap: 1rem;
        }
        .floorplan-sidebar {
            width: 250px;
            background: #f8f9fa;
            border-radius: 10px;
            padding: 1rem;
            overflow-y: auto;
        }
        .floorplan-canvas-container {
            flex: 1;
            background: white;
            border-radius: 10px;
            padding: 1rem;
            position: relative;
        }
        #floorPlanCanvas {
            border: 2px solid #ddd;
            border-radius: 5px;
            cursor: crosshair;
        }
        .tool-btn, .furniture-btn {
            background: #fff;
            border: 2px solid #ddd;
            padding: 0.5rem;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.8rem;
            text-align: center;
            margin: 0.2rem;
        }
        .tool-btn:hover, .furniture-btn:hover {
            border-color: #667eea;
            background: #f0f4ff;
        }
        .tool-btn.active {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
        .furniture-btn.selected {
            background: #764ba2;
            color: white;
            border-color: #764ba2;
        }
        #dropZone {
            border: 2px dashed #ccc;
            padding: 1rem;
            margin: 0.5rem 0;
            text-align: center;
            border-radius: 5px;
            transition: all 0.3s ease;
        }
        #dropZone.dragover {
            border-color: #667eea;
            background: #f0f4ff;
        }
        
        .btn { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            border: none; 
            padding: 0.75rem 1.5rem; 
            border-radius: 20px; 
            cursor: pointer; 
            margin: 0.25rem;
            font-weight: 600;
            font-size: 0.8rem;
            transition: all 0.3s ease;
        }
        .btn:hover { 
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .node-item {
            background: white;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 5px;
            border-left: 3px solid #667eea;
        }
        .device-item {
            background: #f8f9fa;
            padding: 0.75rem;
            margin: 0.5rem 0;
            border-radius: 5px;
            border-left: 3px solid #667eea;
        }
        .device-item.active {
            border-left-color: #2ecc71;
        }
        .device-item.inactive {
            border-left-color: #e74c3c;
        }
        .terminal-display {
            background: #1e1e1e;
            color: #00ff00;
            padding: 1rem;
            border-radius: 10px;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            max-height: 400px;
            overflow-y: auto;
            margin: 1rem 0;
        }
        .user-filter {
            background: #e3f2fd;
            padding: 0.5rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🗺️ BLE Enhanced Terminal Visual Mapping System - Phase 2 Fixed</h1>
        <p>REAL-TIME LOCATION TRACKING • ESP32 Node Positioning • Movement Trails • Interactive Floor Plans • User Filtering</p>
        <div class="status-badge">✅ PHASE 2 FIXED: ALL ISSUES RESOLVED • PYTHON 3.13 COMPATIBLE • v2.0.2</div>
    </div>

    <div class="container">
        <div class="nav-tabs">
            <div class="nav-tab active" onclick="switchTab('overview')">📊 Overview</div>
            <div class="nav-tab" onclick="switchTab('visual-mapping')">🗺️ Visual Mapping</div>
            <div class="nav-tab" onclick="switchTab('floorplan')">🏗️ Interactive Floor Plan</div>
            <div class="nav-tab" onclick="switchTab('devices')">📱 Live Devices</div>
            <div class="nav-tab" onclick="switchTab('users')">👥 Users</div>
            <div class="nav-tab" onclick="switchTab('nodes')">📡 ESP32 Nodes</div>
            <div class="nav-tab" onclick="switchTab('settings')">⚙️ Settings</div>
        </div>

        <!-- Overview Tab -->
        <div id="overview-tab" class="tab-content active">
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number" id="totalDevices">0</div>
                    <div class="stat-label">Active Devices</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="totalUsers">0</div>
                    <div class="stat-label">Users</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="esp32Nodes">0</div>
                    <div class="stat-label">ESP32 Nodes</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="totalRooms">0</div>
                    <div class="stat-label">Rooms</div>
                </div>
            </div>
            
            <div style="text-align: center; padding: 2rem;">
                <h2>🎯 Phase 2 Fixed: All Issues Resolved!</h2>
                <div class="terminal-display">
                    <div>📡 NEW ESP32 NODE DETECTED</div>
                    <div>   Node ID: 6c8a8d</div>
                    <div>   Name: ESP32-6c8a8d</div>
                    <div>   Position: (400, 300)</div>
                    <div>──────────────────────────────────────────────────</div>
                    <div>📍 DEVICE TRACKING UPDATE</div>
                    <div>   Time: 14:23:45</div>
                    <div>   Device: 📱 iPhone</div>
                    <div>   ESP32 Node: 6c8a8d</div>
                    <div>   Distance: 🟢 1.2m (CLOSE)</div>
                    <div>   Confidence: 85%</div>
                    <div>   Position: (620, 380)</div>
                    <div>   Movement: 📈 25 pixels</div>
                    <div>   Predicted: 🔮 (645, 405)</div>
                    <div>──────────────────────────────────────────────────</div>
                </div>
                <p><strong>✅ Fixed: ESP32 Node Management, Visual Mapping, Interactive Floor Plans, User Filtering!</strong></p>
            </div>
        </div>

        <!-- Visual Mapping Tab -->
        <div id="visual-mapping-tab" class="tab-content">
            <div class="mapping-container">
                <!-- Sidebar -->
                <div class="mapping-sidebar">
                    <h3>🎛️ Layer Controls</h3>
                    
                    <!-- User Filter -->
                    <div class="user-filter">
                        <h4>👥 User Filter</h4>
                        <div style="margin: 0.5rem 0;">
                            <input type="radio" name="userFilter" value="all" id="showAllDevices" checked onchange="updateDisplay()">
                            <label for="showAllDevices">Show All Devices</label>
                        </div>
                        <div style="margin: 0.5rem 0;">
                            <input type="radio" name="userFilter" value="users" id="showUsersOnly" onchange="updateDisplay()">
                            <label for="showUsersOnly">Show Assigned Users Only</label>
                        </div>
                    </div>
                    
                    <div class="layer-controls">
                        <div class="layer-control">
                            <input type="checkbox" id="showNodes" checked onchange="updateDisplay()">
                            <label for="showNodes">📡 ESP32 Nodes</label>
                        </div>
                        <div class="layer-control">
                            <input type="checkbox" id="showDevices" checked onchange="updateDisplay()">
                            <label for="showDevices">📱 BLE Devices</label>
                        </div>
                        <div class="layer-control">
                            <input type="checkbox" id="showSignalLines" checked onchange="updateDisplay()">
                            <label for="showSignalLines">📶 Signal Lines</label>
                        </div>
                        <div class="layer-control">
                            <input type="checkbox" id="showTrails" checked onchange="updateDisplay()">
                            <label for="showTrails">📈 Movement Trails</label>
                        </div>
                        <div class="layer-control">
                            <input type="checkbox" id="showPredictions" checked onchange="updateDisplay()">
                            <label for="showPredictions">🔮 Predictions</label>
                        </div>
                        <div class="layer-control">
                            <input type="checkbox" id="showCoverage" onchange="updateDisplay()">
                            <label for="showCoverage">📶 Coverage Areas</label>
                        </div>
                    </div>

                    <h3>🎨 Device Legend</h3>
                    <div class="device-legend">
                        <div class="legend-item">
                            <div class="legend-color" style="background: #00ff00;"></div>
                            <span>🟢 Very Close (< 1m)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #ffff00;"></div>
                            <span>🟡 Close (1-2m)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #ff8800;"></div>
                            <span>🟠 Medium (2-4m)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #ff0000;"></div>
                            <span>🔴 Far (4-6m)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #800000;"></div>
                            <span>⚫ Very Far (> 6m)</span>
                        </div>
                    </div>

                    <h3>📡 ESP32 Nodes</h3>
                    <div id="nodesList"></div>

                    <button class="btn" onclick="refreshMapping()">🔄 Refresh Mapping</button>
                </div>

                <!-- Canvas Container -->
                <div class="mapping-canvas-container">
                    <canvas id="visualMappingCanvas" width="800" height="600"></canvas>
                    <div style="margin-top: 1rem; font-size: 0.9rem; color: #666;">
                        <strong>Instructions:</strong> 
                        • Drag ESP32 nodes to position them on the floor plan
                        • Watch real-time device movement with color-coded distance indicators
                        • Toggle layers to focus on specific data
                        • Use user filter to show only assigned users
                    </div>
                </div>
            </div>
        </div>

        <!-- Interactive Floor Plan Tab -->
        <div id="floorplan-tab" class="tab-content">
            <div class="floorplan-container">
                <!-- Toolbar -->
                <div class="floorplan-sidebar">
                    <h3>🛠️ Drawing Tools</h3>
                    
                    <!-- Tool Selection -->
                    <div style="margin-bottom: 1rem;">
                        <h4>Tools</h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                            <button class="tool-btn active" data-tool="select" onclick="selectTool('select')">🖱️ Select</button>
                            <button class="tool-btn" data-tool="pan" onclick="selectTool('pan')">✋ Pan</button>
                            <button class="tool-btn" data-tool="pen" onclick="selectTool('pen')">✏️ Pen</button>
                            <button class="tool-btn" data-tool="line" onclick="selectTool('line')">📏 Line</button>
                            <button class="tool-btn" data-tool="rectangle" onclick="selectTool('rectangle')">⬜ Rectangle</button>
                            <button class="tool-btn" data-tool="circle" onclick="selectTool('circle')">⭕ Circle</button>
                            <button class="tool-btn" data-tool="polygon" onclick="selectTool('polygon')">🔷 Polygon</button>
                            <button class="tool-btn" data-tool="text" onclick="selectTool('text')">📝 Text</button>
                        </div>
                    </div>

                    <!-- File Operations -->
                    <div style="margin-bottom: 1rem;">
                        <h4>File Operations</h4>
                        <button class="btn" onclick="importFloorPlan()">📁 Import SVG/PNG</button>
                        <button class="btn" onclick="saveFloorPlan()">💾 Save Floor Plan</button>
                        <button class="btn" onclick="loadFloorPlan()">📂 Load Floor Plan</button>
                        <button class="btn" onclick="clearCanvas()">🗑️ Clear Canvas</button>
                        
                        <!-- Drag and Drop Area -->
                        <div id="dropZone">
                            Drop SVG/PNG files here
                        </div>
                    </div>

                    <!-- Coordinate System -->
                    <div style="margin-bottom: 1rem;">
                        <h4>Scale & Measurements</h4>
                        <div style="margin: 0.5rem 0;">
                            <label>Scale (pixels per meter):</label>
                            <input type="number" id="scaleInput" value="50" min="1" max="1000" style="width: 100%; padding: 0.3rem;">
                        </div>
                        <button class="btn" onclick="setReferenceScale()">📐 Set Reference Scale</button>
                        <div id="scaleInfo" style="font-size: 0.8rem; color: #666; margin-top: 0.5rem;">
                            Current: 1 pixel = 0.02m
                        </div>
                    </div>

                    <!-- Furniture Library -->
                    <div style="margin-bottom: 1rem;">
                        <h4>🪑 Furniture Library</h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                            <button class="furniture-btn" data-furniture="desk" onclick="selectFurniture('desk')">🖥️ Desk</button>
                            <button class="furniture-btn" data-furniture="chair" onclick="selectFurniture('chair')">🪑 Chair</button>
                            <button class="furniture-btn" data-furniture="table" onclick="selectFurniture('table')">🪑 Table</button>
                            <button class="furniture-btn" data-furniture="printer" onclick="selectFurniture('printer')">🖨️ Printer</button>
                            <button class="furniture-btn" data-furniture="door" onclick="selectFurniture('door')">🚪 Door</button>
                            <button class="furniture-btn" data-furniture="window" onclick="selectFurniture('window')">🪟 Window</button>
                            <button class="furniture-btn" data-furniture="plant" onclick="selectFurniture('plant')">🪴 Plant</button>
                            <button class="furniture-btn" data-furniture="sofa" onclick="selectFurniture('sofa')">🛋️ Sofa</button>
                        </div>
                    </div>
                </div>

                <!-- Canvas Container -->
                <div class="floorplan-canvas-container">
                    <canvas id="floorPlanCanvas" width="800" height="600"></canvas>
                    <div style="margin-top: 1rem; font-size: 0.9rem; color: #666;">
                        <strong>Instructions:</strong> 
                        • Select tools from the sidebar to draw floor plans
                        • Import SVG/PNG files or draw from scratch
                        • Use furniture library to add common objects
                        • Set scale for accurate measurements
                    </div>
                </div>
            </div>
        </div>

        <!-- Live Devices Tab -->
        <div id="devices-tab" class="tab-content">
            <h2>📱 Live BLE Devices</h2>
            <div id="devicesList"></div>
        </div>

        <!-- Users Tab -->
        <div id="users-tab" class="tab-content">
            <h2>👥 User Management</h2>
            
            <div style="background: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
                <h3>➕ Add New User</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr auto; gap: 1rem; align-items: end;">
                    <div>
                        <label>Device MAC Address:</label>
                        <select id="deviceMacSelect" style="width: 100%; padding: 0.5rem;">
                            <option value="">Select a detected device...</option>
                        </select>
                    </div>
                    <div>
                        <label>User Name:</label>
                        <input type="text" id="userName" placeholder="Enter user name" style="width: 100%; padding: 0.5rem;">
                    </div>
                    <div>
                        <label>Avatar Color:</label>
                        <input type="color" id="avatarColor" value="#667eea" style="width: 100%; padding: 0.5rem;">
                    </div>
                    <button class="btn" onclick="addUser()">Add User</button>
                </div>
            </div>
            
            <div id="usersList"></div>
        </div>

        <!-- ESP32 Nodes Tab -->
        <div id="nodes-tab" class="tab-content">
            <h2>📡 ESP32 Node Management</h2>
            <div id="esp32NodesManagement"></div>
        </div>

        <!-- Settings Tab -->
        <div id="settings-tab" class="tab-content">
            <h2>⚙️ System Settings</h2>
            
            <div style="background: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
                <h3>📡 MQTT Configuration</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr auto; gap: 1rem; align-items: end;">
                    <div>
                        <label>MQTT Broker IP:</label>
                        <input type="text" id="mqttBrokerIp" placeholder="192.168.1.75" style="width: 100%; padding: 0.5rem;">
                    </div>
                    <div>
                        <label>MQTT Broker Port:</label>
                        <input type="number" id="mqttBrokerPort" value="1883" style="width: 100%; padding: 0.5rem;">
                    </div>
                    <button class="btn" onclick="updateMqttSettings()">Update MQTT</button>
                </div>
            </div>
            
            <div style="background: white; padding: 1.5rem; border-radius: 10px;">
                <h3>⏱️ Device Timeout</h3>
                <div style="display: grid; grid-template-columns: 1fr auto; gap: 1rem; align-items: end;">
                    <div>
                        <label>Device Timeout (seconds):</label>
                        <input type="number" id="deviceTimeout" value="120" min="30" max="3600" style="width: 100%; padding: 0.5rem;">
                    </div>
                    <button class="btn" onclick="updateDeviceTimeout()">Update Timeout</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let canvas, ctx;
        let floorPlanCanvas, floorPlanCtx;
        let esp32Nodes = [];
        let realTimeDevices = [];
        let isDragging = false;
        let dragNode = null;
        let dragOffset = { x: 0, y: 0 };
        
        // Floor plan variables
        let currentTool = 'select';
        let isDrawing = false;
        let startX, startY;
        let floorPlanObjects = [];
        let backgroundImage = null;
        let scale = 50; // pixels per meter

        // Initialize visual mapping canvas
        function initCanvas() {
            canvas = document.getElementById('visualMappingCanvas');
            if (!canvas) return;
            
            ctx = canvas.getContext('2d');
            
            // Add mouse event listeners
            canvas.addEventListener('mousedown', onMouseDown);
            canvas.addEventListener('mousemove', onMouseMove);
            canvas.addEventListener('mouseup', onMouseUp);
            canvas.addEventListener('mouseleave', onMouseUp);
            
            drawCanvas();
        }

        // Initialize floor plan canvas
        function initFloorPlanCanvas() {
            floorPlanCanvas = document.getElementById('floorPlanCanvas');
            if (!floorPlanCanvas) return;
            
            floorPlanCtx = floorPlanCanvas.getContext('2d');
            
            // Add mouse event listeners for floor plan
            floorPlanCanvas.addEventListener('mousedown', onFloorPlanMouseDown);
            floorPlanCanvas.addEventListener('mousemove', onFloorPlanMouseMove);
            floorPlanCanvas.addEventListener('mouseup', onFloorPlanMouseUp);
            
            // Add drag and drop support
            const dropZone = document.getElementById('dropZone');
            if (dropZone) {
                dropZone.addEventListener('dragover', handleDragOver);
                dropZone.addEventListener('drop', handleDrop);
            }
            
            drawFloorPlan();
        }

        function onMouseDown(e) {
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Check if clicking on an ESP32 node
            for (let node of esp32Nodes) {
                const nodeX = node.x_position || 400;
                const nodeY = node.y_position || 300;
                
                if (Math.sqrt((x - nodeX) ** 2 + (y - nodeY) ** 2) < 20) {
                    isDragging = true;
                    dragNode = node;
                    dragOffset.x = x - nodeX;
                    dragOffset.y = y - nodeY;
                    canvas.style.cursor = 'grabbing';
                    break;
                }
            }
        }

        function onMouseMove(e) {
            if (!isDragging || !dragNode) return;
            
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left - dragOffset.x;
            const y = e.clientY - rect.top - dragOffset.y;
            
            // Update node position
            dragNode.x_position = Math.max(20, Math.min(canvas.width - 20, x));
            dragNode.y_position = Math.max(20, Math.min(canvas.height - 20, y));
            
            drawCanvas();
        }

        function onMouseUp(e) {
            if (isDragging && dragNode) {
                // Save node position to database
                fetch(`/api/esp32-nodes/${dragNode.id}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        x_position: dragNode.x_position,
                        y_position: dragNode.y_position
                    })
                });
            }
            
            isDragging = false;
            dragNode = null;
            canvas.style.cursor = 'default';
        }

        function drawCanvas() {
            if (!ctx || !canvas) return;
            
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw grid
            drawGrid();
            
            // Draw layers based on controls
            if (document.getElementById('showCoverage')?.checked) {
                drawCoverageAreas();
            }
            
            // Draw signal strength lines BEFORE devices so devices appear on top
            if (document.getElementById('showDevices')?.checked && document.getElementById('showSignalLines')?.checked) {
                drawSignalStrengthLines();
            }
            
            if (document.getElementById('showTrails')?.checked) {
                drawMovementTrails();
            }
            
            if (document.getElementById('showNodes')?.checked) {
                drawESP32Nodes();
            }
            
            if (document.getElementById('showDevices')?.checked) {
                drawBLEDevices();
            }
            
            if (document.getElementById('showPredictions')?.checked) {
                drawPredictions();
            }
        }

        function drawGrid() {
            ctx.strokeStyle = '#f0f0f0';
            ctx.lineWidth = 1;
            
            // Draw vertical lines
            for (let x = 0; x < canvas.width; x += 50) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, canvas.height);
                ctx.stroke();
            }
            
            // Draw horizontal lines
            for (let y = 0; y < canvas.height; y += 50) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(canvas.width, y);
                ctx.stroke();
            }
        }

        function drawESP32Nodes() {
            esp32Nodes.forEach(node => {
                const x = node.x_position || 400;
                const y = node.y_position || 300;
                
                // Draw node
                ctx.fillStyle = '#667eea';
                ctx.beginPath();
                ctx.arc(x, y, 15, 0, 2 * Math.PI);
                ctx.fill();
                
                // Draw node icon
                ctx.fillStyle = 'white';
                ctx.font = '16px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('📡', x, y + 5);
                
                // Draw node label
                ctx.fillStyle = '#333';
                ctx.font = '12px Arial';
                ctx.fillText(node.name || node.node_id, x, y + 35);
            });
        }

        function drawBLEDevices() {
            // Filter devices based on user filter
            const userFilter = document.querySelector('input[name="userFilter"]:checked')?.value || 'all';
            let devicesToShow = realTimeDevices;
            
            if (userFilter === 'users') {
                devicesToShow = realTimeDevices.filter(device => device.is_assigned);
            }
            
            devicesToShow.forEach(device => {
                if (!device.x || !device.y) return;
                
                const x = device.x;
                const y = device.y;
                
                // Get color based on distance
                const color = device.distance_color || '#666';
                
                // Draw device
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(x, y, 8, 0, 2 * Math.PI);
                ctx.fill();
                
                // Draw device emoji
                ctx.fillStyle = 'white';
                ctx.font = '12px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(device.device_emoji || '📱', x, y + 3);
                
                // Draw device name
                ctx.fillStyle = '#333';
                ctx.font = '10px Arial';
                ctx.fillText(device.device_name || device.device_id, x, y - 15);
                
                // Draw user indicator if assigned
                if (device.is_assigned) {
                    ctx.strokeStyle = '#2ecc71';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.arc(x, y, 12, 0, 2 * Math.PI);
                    ctx.stroke();
                }
            });
        }

        function drawSignalStrengthLines() {
            // Get user filter setting
            const userFilter = document.querySelector('input[name="userFilter"]:checked')?.value || 'all';
            
            // Get current device-node data from the backend
            fetch('/api/device_node_data')
                .then(response => response.json())
                .then(data => {
                    if (!data.success) return;
                    
                    const deviceNodeData = data.device_node_data;
                    
                    // Draw lines for each device to its detecting nodes
                    Object.keys(deviceNodeData).forEach(deviceId => {
                        const device = realTimeDevices.find(d => d.device_id === deviceId);
                        if (!device || !device.x || !device.y) return;
                        
                        // Apply user filter - only show signal lines for devices that pass the filter
                        if (userFilter === 'users' && !device.is_assigned) {
                            return; // Skip this device if showing users only and device is not assigned
                        }
                        
                        const deviceX = device.x;
                        const deviceY = device.y;
                        
                        // Draw line to each detecting node
                        Object.keys(deviceNodeData[deviceId]).forEach(nodeId => {
                            const nodeData = deviceNodeData[deviceId][nodeId];
                            const node = esp32Nodes.find(n => n.node_id === nodeId);
                            
                            if (!node) return;
                            
                            const nodeX = node.x_position || 400;
                            const nodeY = node.y_position || 300;
                            const distance = nodeData.distance;
                            const confidence = nodeData.confidence;
                            
                            // Get color based on distance (same as device colors)
                            let lineColor;
                            if (distance < 1) {
                                lineColor = '#2ecc71';      // Green - Very close
                            } else if (distance < 2) {
                                lineColor = '#f1c40f';      // Yellow - Close
                            } else if (distance < 4) {
                                lineColor = '#e67e22';      // Orange - Medium
                            } else if (distance < 6) {
                                lineColor = '#e74c3c';      // Red - Far
                            } else {
                                lineColor = '#8b0000';      // Dark red - Very far
                            }
                            
                            // Draw the signal line
                            ctx.strokeStyle = lineColor;
                            ctx.lineWidth = Math.max(1, confidence / 20); // Thicker line for higher confidence
                            ctx.setLineDash([5, 3]); // Dashed line
                            
                            ctx.beginPath();
                            ctx.moveTo(deviceX, deviceY);
                            ctx.lineTo(nodeX, nodeY);
                            ctx.stroke();
                            
                            // Draw distance label on the line
                            const midX = (deviceX + nodeX) / 2;
                            const midY = (deviceY + nodeY) / 2;
                            
                            ctx.fillStyle = lineColor;
                            ctx.font = '10px Arial';
                            ctx.textAlign = 'center';
                            ctx.fillText(`${distance.toFixed(1)}m`, midX, midY - 5);
                        });
                    });
                    
                    // Reset line dash
                    ctx.setLineDash([]);
                })
                .catch(error => {
                    console.log('Could not fetch device-node data for signal lines');
                });
        }

        function drawMovementTrails() {
            // Filter devices based on user filter
            const userFilter = document.querySelector('input[name="userFilter"]:checked')?.value || 'all';
            let devicesToShow = realTimeDevices;
            
            if (userFilter === 'users') {
                devicesToShow = realTimeDevices.filter(device => device.is_assigned);
            }
            
            devicesToShow.forEach(device => {
                if (!device.trail || device.trail.length < 2) return;
                
                const color = device.distance_color || '#666';
                ctx.strokeStyle = color + '80'; // Add transparency
                ctx.lineWidth = 2;
                
                ctx.beginPath();
                device.trail.forEach((point, index) => {
                    if (index === 0) {
                        ctx.moveTo(point.x, point.y);
                    } else {
                        ctx.lineTo(point.x, point.y);
                    }
                });
                ctx.stroke();
            });
        }

        function drawPredictions() {
            // Filter devices based on user filter
            const userFilter = document.querySelector('input[name="userFilter"]:checked')?.value || 'all';
            let devicesToShow = realTimeDevices;
            
            if (userFilter === 'users') {
                devicesToShow = realTimeDevices.filter(device => device.is_assigned);
            }
            
            devicesToShow.forEach(device => {
                if (!device.prediction || !device.x || !device.y) return;
                
                const currentX = device.x;
                const currentY = device.y;
                const predX = device.prediction.x;
                const predY = device.prediction.y;
                
                // Draw prediction line
                ctx.strokeStyle = '#ff6b6b';
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 5]);
                
                ctx.beginPath();
                ctx.moveTo(currentX, currentY);
                ctx.lineTo(predX, predY);
                ctx.stroke();
                
                // Draw prediction point
                ctx.fillStyle = '#ff6b6b';
                ctx.beginPath();
                ctx.arc(predX, predY, 6, 0, 2 * Math.PI);
                ctx.fill();
                
                ctx.setLineDash([]); // Reset line dash
            });
        }

        function drawCoverageAreas() {
            esp32Nodes.forEach(node => {
                const x = node.x_position || 400;
                const y = node.y_position || 300;
                const maxDistance = node.max_distance || 16; // Default 16m
                
                // Convert meters to pixels (scale: 50 pixels per meter)
                const radiusPixels = maxDistance * 50;
                
                // Draw coverage circle
                ctx.strokeStyle = '#667eea40';
                ctx.fillStyle = '#667eea10';
                ctx.lineWidth = 2;
                ctx.setLineDash([10, 5]);
                
                ctx.beginPath();
                ctx.arc(x, y, radiusPixels, 0, 2 * Math.PI);
                ctx.fill();
                ctx.stroke();
                
                // Draw distance label
                ctx.fillStyle = '#667eea';
                ctx.font = '12px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(`${maxDistance}m`, x, y - radiusPixels - 10);
                
                ctx.setLineDash([]); // Reset line dash
            });
        }

        // Floor plan drawing and interaction functions
        let selectedObject = null;
        let isFloorPlanDragging = false;
        let floorPlanDragOffset = { x: 0, y: 0 };

        function onFloorPlanMouseDown(e) {
            const rect = floorPlanCanvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            if (currentTool === 'select') {
                // Check if clicking on an existing object
                selectedObject = getObjectAtPosition(mouseX, mouseY);
                if (selectedObject) {
                    isFloorPlanDragging = true;
                    floorPlanDragOffset.x = mouseX - selectedObject.x;
                    floorPlanDragOffset.y = mouseY - selectedObject.y;
                    floorPlanCanvas.style.cursor = 'grabbing';
                    console.log('🎯 Selected object for dragging:', selectedObject.type);
                    return;
                }
            }
            
            // Drawing mode
            if (currentTool !== 'select') {
                startX = mouseX;
                startY = mouseY;
                isDrawing = true;
                console.log('🎨 Started drawing:', currentTool);
            }
        }

        function onFloorPlanMouseMove(e) {
            const rect = floorPlanCanvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            // Handle dragging
            if (isFloorPlanDragging && selectedObject) {
                selectedObject.x = mouseX - floorPlanDragOffset.x;
                selectedObject.y = mouseY - floorPlanDragOffset.y;
                drawFloorPlan();
                console.log('🔄 Dragging object to:', selectedObject.x, selectedObject.y);
                return;
            }
            
            // Handle drawing
            if (!isDrawing) {
                // Show cursor feedback for hovering over objects
                if (currentTool === 'select') {
                    const hoverObject = getObjectAtPosition(mouseX, mouseY);
                    floorPlanCanvas.style.cursor = hoverObject ? 'grab' : 'default';
                }
                return;
            }
            
            const currentX = mouseX;
            const currentY = mouseY;
            
            // Redraw canvas with current shape
            drawFloorPlan();
            
            // Draw current shape being drawn
            floorPlanCtx.strokeStyle = '#667eea';
            floorPlanCtx.lineWidth = 2;
            
            switch(currentTool) {
                case 'line':
                    floorPlanCtx.beginPath();
                    floorPlanCtx.moveTo(startX, startY);
                    floorPlanCtx.lineTo(currentX, currentY);
                    floorPlanCtx.stroke();
                    break;
                case 'rectangle':
                    floorPlanCtx.strokeRect(startX, startY, currentX - startX, currentY - startY);
                    break;
                case 'circle':
                    const radius = Math.sqrt((currentX - startX) ** 2 + (currentY - startY) ** 2);
                    floorPlanCtx.beginPath();
                    floorPlanCtx.arc(startX, startY, radius, 0, 2 * Math.PI);
                    floorPlanCtx.stroke();
                    break;
            }
        }

        function onFloorPlanMouseUp(e) {
            const rect = floorPlanCanvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            // Handle drag end
            if (isFloorPlanDragging) {
                isFloorPlanDragging = false;
                floorPlanCanvas.style.cursor = 'grab';
                console.log('✅ Finished dragging object');
                return;
            }
            
            // Handle drawing end
            if (!isDrawing) return;
            
            const endX = mouseX;
            const endY = mouseY;
            
            // Add shape to objects array
            const shape = {
                type: currentTool,
                style: {
                    strokeStyle: '#667eea',
                    lineWidth: 2,
                    fillStyle: 'rgba(102, 126, 234, 0.1)'
                }
            };
            
            switch(currentTool) {
                case 'line':
                    shape.x1 = startX;
                    shape.y1 = startY;
                    shape.x2 = endX;
                    shape.y2 = endY;
                    break;
                case 'rectangle':
                    shape.x = startX;
                    shape.y = startY;
                    shape.width = endX - startX;
                    shape.height = endY - startY;
                    break;
                case 'circle':
                    shape.x = startX;
                    shape.y = startY;
                    shape.radius = Math.sqrt((endX - startX) ** 2 + (endY - startY) ** 2);
                    break;
            }
            
            floorPlanObjects.push(shape);
            isDrawing = false;
            drawFloorPlan();
            console.log('✅ Added new object:', currentTool);
        }

        // Helper function to find object at mouse position
        function getObjectAtPosition(x, y) {
            // Check objects in reverse order (top to bottom)
            for (let i = floorPlanObjects.length - 1; i >= 0; i--) {
                const obj = floorPlanObjects[i];
                
                switch(obj.type) {
                    case 'rectangle':
                        if (x >= obj.x && x <= obj.x + obj.width && 
                            y >= obj.y && y <= obj.y + obj.height) {
                            return obj;
                        }
                        break;
                    case 'circle':
                        const distance = Math.sqrt((x - obj.x) ** 2 + (y - obj.y) ** 2);
                        if (distance <= obj.radius) {
                            return obj;
                        }
                        break;
                    case 'line':
                        // Check if point is near the line (within 5 pixels)
                        const lineDistance = distanceToLine(x, y, obj.x1, obj.y1, obj.x2, obj.y2);
                        if (lineDistance <= 5) {
                            return obj;
                        }
                        break;
                    case 'furniture':
                        if (x >= obj.x && x <= obj.x + obj.width && 
                            y >= obj.y && y <= obj.y + obj.height) {
                            return obj;
                        }
                        break;
                    case 'esp32_node':
                        const nodeDistance = Math.sqrt((x - obj.x) ** 2 + (y - obj.y) ** 2);
                        if (nodeDistance <= 15) { // 15px radius for ESP32 nodes
                            return obj;
                        }
                        break;
                }
            }
            return null;
        }

        // Helper function to calculate distance from point to line
        function distanceToLine(px, py, x1, y1, x2, y2) {
            const A = px - x1;
            const B = py - y1;
            const C = x2 - x1;
            const D = y2 - y1;
            
            const dot = A * C + B * D;
            const lenSq = C * C + D * D;
            
            if (lenSq === 0) return Math.sqrt(A * A + B * B);
            
            let param = dot / lenSq;
            
            if (param < 0) {
                return Math.sqrt(A * A + B * B);
            } else if (param > 1) {
                const E = px - x2;
                const F = py - y2;
                return Math.sqrt(E * E + F * F);
            } else {
                const projX = x1 + param * C;
                const projY = y1 + param * D;
                const dx = px - projX;
                const dy = py - projY;
                return Math.sqrt(dx * dx + dy * dy);
            }
        }

        function drawFloorPlan() {
            if (!floorPlanCtx || !floorPlanCanvas) return;
            
            // Clear canvas
            floorPlanCtx.clearRect(0, 0, floorPlanCanvas.width, floorPlanCanvas.height);
            
            // Draw background image if loaded
            if (backgroundImage) {
                floorPlanCtx.drawImage(backgroundImage, 0, 0, floorPlanCanvas.width, floorPlanCanvas.height);
            }
            
            // Draw grid
            drawFloorPlanGrid();
            
            // Draw all objects
            floorPlanObjects.forEach(obj => {
                drawFloorPlanObject(obj);
            });
        }

        function drawFloorPlanGrid() {
            floorPlanCtx.strokeStyle = '#e0e0e0';
            floorPlanCtx.lineWidth = 1;
            
            // Draw vertical lines
            for (let x = 0; x < floorPlanCanvas.width; x += scale) {
                floorPlanCtx.beginPath();
                floorPlanCtx.moveTo(x, 0);
                floorPlanCtx.lineTo(x, floorPlanCanvas.height);
                floorPlanCtx.stroke();
            }
            
            // Draw horizontal lines
            for (let y = 0; y < floorPlanCanvas.height; y += scale) {
                floorPlanCtx.beginPath();
                floorPlanCtx.moveTo(0, y);
                floorPlanCtx.lineTo(floorPlanCanvas.width, y);
                floorPlanCtx.stroke();
            }
        }

        function drawFloorPlanObject(obj) {
            floorPlanCtx.save();
            
            switch(obj.type) {
                case 'line':
                    floorPlanCtx.strokeStyle = obj.style.strokeStyle;
                    floorPlanCtx.lineWidth = obj.style.lineWidth;
                    floorPlanCtx.beginPath();
                    floorPlanCtx.moveTo(obj.x1, obj.y1);
                    floorPlanCtx.lineTo(obj.x2, obj.y2);
                    floorPlanCtx.stroke();
                    break;
                    
                case 'rectangle':
                    floorPlanCtx.strokeStyle = obj.style.strokeStyle;
                    floorPlanCtx.lineWidth = obj.style.lineWidth;
                    floorPlanCtx.fillStyle = obj.style.fillStyle;
                    floorPlanCtx.fillRect(obj.x, obj.y, obj.width, obj.height);
                    floorPlanCtx.strokeRect(obj.x, obj.y, obj.width, obj.height);
                    break;
                    
                case 'circle':
                    floorPlanCtx.strokeStyle = obj.style.strokeStyle;
                    floorPlanCtx.lineWidth = obj.style.lineWidth;
                    floorPlanCtx.fillStyle = obj.style.fillStyle;
                    floorPlanCtx.beginPath();
                    floorPlanCtx.arc(obj.x, obj.y, obj.radius, 0, 2 * Math.PI);
                    floorPlanCtx.fill();
                    floorPlanCtx.stroke();
                    break;
                    
                case 'furniture':
                    floorPlanCtx.font = `${obj.width}px Arial`;
                    floorPlanCtx.textAlign = 'center';
                    floorPlanCtx.fillText(obj.emoji, obj.x, obj.y + obj.height/2);
                    break;
            }
            
            floorPlanCtx.restore();
        }

        // Tool selection functions
        function selectTool(tool) {
            currentTool = tool;
            document.querySelectorAll('.tool-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelector(`[data-tool="${tool}"]`).classList.add('active');
        }

        function selectFurniture(type) {
            // Add furniture to floor plan at a default position
            const furniture = getFurnitureData(type);
            const obj = {
                type: 'furniture',
                x: 100 + Math.random() * 50, // Add some randomness to avoid overlap
                y: 100 + Math.random() * 50,
                width: furniture.width,
                height: furniture.height,
                emoji: furniture.emoji,
                furnitureType: type
            };
            floorPlanObjects.push(obj);
            drawFloorPlan();
            console.log('🪑 Added furniture:', type, 'at position:', obj.x, obj.y);
            
            // Automatically switch to select tool for immediate dragging
            selectTool('select');
        }

        function getFurnitureData(type) {
            const furniture = {
                desk: { width: 60, height: 30, emoji: '🖥️' },
                chair: { width: 20, height: 20, emoji: '🪑' },
                table: { width: 40, height: 40, emoji: '🪑' },
                printer: { width: 30, height: 25, emoji: '🖨️' },
                door: { width: 30, height: 10, emoji: '🚪' },
                window: { width: 40, height: 10, emoji: '🪟' },
                plant: { width: 15, height: 15, emoji: '🪴' },
                sofa: { width: 80, height: 35, emoji: '🛋️' }
            };
            return furniture[type] || { width: 20, height: 20, emoji: '📦' };
        }

        // File operations
        function importFloorPlan() {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.svg,.png,.jpg,.jpeg';
            input.onchange = function(e) {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(event) {
                        const img = new Image();
                        img.onload = function() {
                            backgroundImage = img;
                            drawFloorPlan();
                        };
                        img.src = event.target.result;
                    };
                    reader.readAsDataURL(file);
                }
            };
            input.click();
        }

        function saveFloorPlan() {
            const dataURL = floorPlanCanvas.toDataURL();
            const link = document.createElement('a');
            link.download = 'floor_plan.png';
            link.href = dataURL;
            link.click();
        }

        function loadFloorPlan() {
            // Implementation for loading saved floor plans
            alert('Load floor plan functionality would be implemented here');
        }

        function clearCanvas() {
            floorPlanObjects = [];
            backgroundImage = null;
            drawFloorPlan();
        }

        function setReferenceScale() {
            const newScale = parseInt(document.getElementById('scaleInput').value);
            scale = newScale;
            document.getElementById('scaleInfo').textContent = `Current: 1 pixel = ${(1/scale).toFixed(3)}m`;
            drawFloorPlan();
        }

        // Drag and drop handlers
        function handleDragOver(e) {
            e.preventDefault();
            e.currentTarget.classList.add('dragover');
        }

        function handleDrop(e) {
            e.preventDefault();
            e.currentTarget.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('image/')) {
                    const reader = new FileReader();
                    reader.onload = function(event) {
                        const img = new Image();
                        img.onload = function() {
                            backgroundImage = img;
                            drawFloorPlan();
                        };
                        img.src = event.target.result;
                    };
                    reader.readAsDataURL(file);
                }
            }
        }

        function updateDisplay() {
            drawCanvas();
        }

        function refreshMapping() {
            loadESP32Nodes();
            loadRealTimeDevices();
        }

        // Tab switching
        function switchTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.nav-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
            
            // Initialize canvas if visual mapping tab is selected
            if (tabName === 'visual-mapping') {
                setTimeout(initCanvas, 100);
                refreshMapping();
            } else if (tabName === 'floorplan') {
                setTimeout(initFloorPlanCanvas, 100);
            } else if (tabName === 'nodes') {
                loadESP32NodesManagement();
            }
        }

        // Data loading functions
        function loadStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('totalDevices').textContent = data.stats.total_devices;
                        document.getElementById('totalUsers').textContent = data.stats.total_users;
                        document.getElementById('esp32Nodes').textContent = data.stats.esp32_nodes;
                        document.getElementById('totalRooms').textContent = data.stats.total_rooms;
                    }
                });
        }

        function loadRealTimeDevices() {
            fetch('/api/devices/real-time')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        realTimeDevices = data.devices;
                        updateDevicesList();
                        if (canvas) drawCanvas();
                    }
                });
        }

        function loadESP32Nodes() {
            fetch('/api/esp32-nodes')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        esp32Nodes = data.esp32_nodes;
                        updateNodesList();
                        if (canvas) drawCanvas();
                    }
                });
        }

        function loadESP32NodesManagement() {
            fetch('/api/esp32-nodes')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateESP32NodesManagement(data.esp32_nodes);
                    }
                });
        }

        function updateDevicesList() {
            const container = document.getElementById('devicesList');
            if (!container) return;
            
            container.innerHTML = realTimeDevices.map(device => `
                <div class="device-item active">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>${device.device_emoji} ${device.device_name}</strong>
                            ${device.is_assigned ? '<span style="color: #2ecc71; font-size: 0.8rem;">👤 Assigned User</span>' : ''}
                            <div style="font-size: 0.8rem; color: #666;">
                                ${device.device_id} • ESP32: ${device.esp32_node}
                            </div>
                            <div style="font-size: 0.8rem;">
                                ${device.distance_symbol} ${device.distance?.toFixed(1)}m • 
                                Position: (${device.x?.toFixed(0) || 'N/A'}, ${device.y?.toFixed(0) || 'N/A'})
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.8rem; color: #666;">
                                Confidence: ${device.confidence}%
                            </div>
                            <div style="font-size: 0.7rem; color: #999;">
                                ${new Date(device.last_seen).toLocaleTimeString()}
                            </div>
                        </div>
                    </div>
                </div>
            `).join('');
        }

        function updateNodesList() {
            const container = document.getElementById('nodesList');
            if (!container) return;
            
            container.innerHTML = esp32Nodes.map(node => `
                <div class="node-item">
                    <div><strong>📡 ${node.name}</strong></div>
                    <div style="font-size: 0.8rem; color: #666;">
                        ID: ${node.node_id} • Room: ${node.room_name}
                    </div>
                    <div style="font-size: 0.8rem; color: #666;">
                        Position: (${node.x_position?.toFixed(0) || 0}, ${node.y_position?.toFixed(0) || 0})
                    </div>
                </div>
            `).join('');
        }

        function updateESP32NodesManagement(nodes) {
            const container = document.getElementById('esp32NodesManagement');
            if (!container) return;
            
            if (!nodes || nodes.length === 0) {
                container.innerHTML = '<div style="text-align: center; color: #666; padding: 20px;">No ESP32 nodes found</div>';
                return;
            }
            
            container.innerHTML = nodes.map(node => `
                <div class="node-item">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div><strong>📡 ${node.name}</strong></div>
                            <div style="font-size: 0.8rem; color: #666;">
                                ID: ${node.node_id} • Room: ${node.room_name}
                            </div>
                            <div style="font-size: 0.8rem; color: #666;">
                                Position: (${node.x_position?.toFixed(0) || 0}, ${node.y_position?.toFixed(0) || 0})
                            </div>
                            <div style="font-size: 0.8rem; color: #2ecc71;">
                                📶 Max Distance: 16m (Hardware Limit)
                            </div>
                            <div style="font-size: 0.8rem; color: #e67e22;">
                                🎯 Distance Set: ${node.max_distance || 16}m
                            </div>
                            <div style="font-size: 0.7rem; color: #999;">
                                Created: ${new Date(node.created_at).toLocaleDateString()}
                            </div>
                        </div>
                        <div>
                            <button class="btn" onclick="editNode(${node.id})" style="background: #3498db;">Edit</button>
                        </div>
                    </div>
                </div>
            `).join('');
        }

        function editNode(nodeId) {
            // Find node from the most recent data
            let node = null;
            
            // Try to get fresh data first
            fetch('/api/esp32-nodes')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        node = data.esp32_nodes.find(n => n.id === nodeId);
                        if (!node) {
                            alert('Node not found');
                            return;
                        }
                        
                        showEditDialog(node, nodeId);
                    }
                })
                .catch(error => {
                    // Fallback to global variable if fetch fails
                    node = esp32Nodes.find(n => n.id === nodeId);
                    if (!node) {
                        alert('Node not found');
                        return;
                    }
                    showEditDialog(node, nodeId);
                });
        }
        
        function showEditDialog(node, nodeId) {
            
            // Create a custom dialog with all fields
            const dialogHtml = `
                <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); max-width: 400px;">
                    <h3>Edit ESP32 Node</h3>
                    <div style="margin-bottom: 15px;">
                        <label style="display: block; margin-bottom: 5px;">Node Name:</label>
                        <input type="text" id="editNodeName" value="${node.name}" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
                    </div>
                    <div style="margin-bottom: 15px;">
                        <label style="display: block; margin-bottom: 5px;">Room:</label>
                        <input type="text" id="editNodeRoom" value="${node.room_name}" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
                    </div>
                    <div style="margin-bottom: 20px;">
                        <label style="display: block; margin-bottom: 5px;">Detection Distance (meters):</label>
                        <input type="number" id="editNodeDistance" value="${node.max_distance || 16}" min="0" max="16" step="0.1" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
                        <small style="color: #666;">Range: 0-16m (0 = no limit, 16 = hardware max)</small>
                    </div>
                    <div style="text-align: right;">
                        <button onclick="closeEditDialog()" style="margin-right: 10px; padding: 8px 16px; background: #ccc; border: none; border-radius: 4px; cursor: pointer;">Cancel</button>
                        <button onclick="saveNodeEdit(${nodeId})" style="padding: 8px 16px; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer;">Save</button>
                    </div>
                </div>
            `;
            
            // Create overlay
            const overlay = document.createElement('div');
            overlay.id = 'editNodeOverlay';
            overlay.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); display: flex; justify-content: center; align-items: center; z-index: 1000;';
            overlay.innerHTML = dialogHtml;
            
            document.body.appendChild(overlay);
        }
        
        function closeEditDialog() {
            const overlay = document.getElementById('editNodeOverlay');
            if (overlay) {
                overlay.remove();
            }
        }
        
        function saveNodeEdit(nodeId) {
            const name = document.getElementById('editNodeName').value.trim();
            const room = document.getElementById('editNodeRoom').value.trim();
            const distance = parseFloat(document.getElementById('editNodeDistance').value);
            
            if (!name) {
                alert('Please enter a node name');
                return;
            }
            
            if (!room) {
                alert('Please enter a room name');
                return;
            }
            
            if (isNaN(distance) || distance < 0 || distance > 16) {
                alert('Please enter a valid distance (0-16 meters). Hardware limit is 16m.');
                return;
            }
            
            const updateData = {
                name: name,
                room_name: room,
                max_distance: distance
            };
            
            fetch(`/api/esp32-nodes/${nodeId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(updateData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    closeEditDialog();
                    loadESP32NodesManagement();
                    loadESP32Nodes();
                    alert(`ESP32 node updated! Detection distance set to ${distance}m (within 16m hardware limit)`);
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(error => {
                alert('Error: ' + error.message);
            });
        }

        // User management functions
        function loadUsers() {
            fetch('/api/users')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateUsersList(data.users);
                    }
                });
        }

        function loadDevicesList() {
            fetch('/api/devices/list')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const select = document.getElementById('deviceMacSelect');
                        if (select) {
                            select.innerHTML = '<option value="">Select a detected device...</option>' +
                                data.devices.map(device => 
                                    `<option value="${device.device_mac}">${device.device_emoji} ${device.device_name} (${device.device_mac})</option>`
                                ).join('');
                        }
                    }
                });
        }

        function updateUsersList(users) {
            const container = document.getElementById('usersList');
            if (!container) return;
            
            container.innerHTML = users.map(user => `
                <div class="device-item">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="display: flex; align-items: center;">
                                <div style="width: 20px; height: 20px; background: ${user.avatar_color}; border-radius: 50%; margin-right: 0.5rem;"></div>
                                <strong>${user.name}</strong>
                            </div>
                            <div style="font-size: 0.8rem; color: #666;">
                                Device: ${user.device_mac}
                            </div>
                        </div>
                        <button class="btn" onclick="deleteUser(${user.id})" style="background: #e74c3c;">Delete</button>
                    </div>
                </div>
            `).join('');
        }

        function addUser() {
            const deviceMac = document.getElementById('deviceMacSelect').value;
            const userName = document.getElementById('userName').value;
            const avatarColor = document.getElementById('avatarColor').value;
            
            if (!deviceMac || !userName) {
                alert('Please select a device and enter a user name');
                return;
            }
            
            fetch('/api/users', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    device_mac: deviceMac,
                    name: userName,
                    avatar_color: avatarColor
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('deviceMacSelect').value = '';
                    document.getElementById('userName').value = '';
                    loadUsers();
                } else {
                    alert('Error: ' + data.error);
                }
            });
        }

        function deleteUser(userId) {
            if (!confirm('Are you sure you want to delete this user?')) return;
            
            fetch(`/api/users/${userId}`, { method: 'DELETE' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        loadUsers();
                    } else {
                        alert('Error: ' + data.error);
                    }
                });
        }

        // Settings functions
        function loadSettings() {
            fetch('/api/settings')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('mqttBrokerIp').value = data.settings.mqtt_broker_ip;
                        document.getElementById('mqttBrokerPort').value = data.settings.mqtt_broker_port;
                        document.getElementById('deviceTimeout').value = data.settings.device_timeout_seconds;
                    }
                });
        }

        function updateMqttSettings() {
            const ip = document.getElementById('mqttBrokerIp').value;
            const port = document.getElementById('mqttBrokerPort').value;
            
            fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    mqtt_broker_ip: ip,
                    mqtt_broker_port: port
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('MQTT settings updated successfully!');
                } else {
                    alert('Error: ' + data.error);
                }
            });
        }

        function updateDeviceTimeout() {
            const timeout = document.getElementById('deviceTimeout').value;
            
            fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    device_timeout_seconds: timeout
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Device timeout updated successfully!');
                } else {
                    alert('Error: ' + data.error);
                }
            });
        }

        // Initialize page
        document.addEventListener('DOMContentLoaded', function() {
            loadStats();
            loadUsers();
            loadDevicesList();
            loadSettings();
            loadESP32Nodes();
            loadESP32NodesManagement(); // Add this line
            
            // Auto-refresh data
            setInterval(() => {
                loadStats();
                loadRealTimeDevices();
            }, 2000);
            
            setInterval(() => {
                loadDevicesList();
                loadESP32NodesManagement(); // Add periodic refresh
            }, 10000);
        });
    </script>
</body>
</html>
'''

def cleanup():
    """Cleanup function"""
    global mqtt_client
    
    print(f"\n{TerminalColors.YELLOW}🔄 SHUTTING DOWN SYSTEM{TerminalColors.END}")
    print(f"   {TerminalColors.CYAN}Disconnecting MQTT client...{TerminalColors.END}")
    
    if mqtt_client:
        mqtt_client.disconnect()
        print(f"   {TerminalColors.GREEN}✅ MQTT client disconnected{TerminalColors.END}")
    
    print(f"   {TerminalColors.GREEN}✅ Cleanup completed{TerminalColors.END}")
    print(f"{TerminalColors.CYAN}{'='*50}{TerminalColors.END}")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    cleanup()
    sys.exit(0)

def main():
    """Main application entry point"""
    global app
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup)
    
    print_banner()
    
    app.config['SECRET_KEY'] = str(uuid.uuid4())
    CORS(app)
    
    print(f"\n{TerminalColors.YELLOW}🔄 STARTING MQTT CLIENT{TerminalColors.END}")
    if start_mqtt_client():
        print(f"{TerminalColors.GREEN}✅ MQTT client started successfully{TerminalColors.END}")
    else:
        print(f"{TerminalColors.RED}❌ MQTT client failed to start{TerminalColors.END}")
    
    network_info = get_network_info()
    broker_ip = db.get_setting('mqtt_broker_ip', network_info['local_ip'])
    
    print(f"\n{TerminalColors.YELLOW}🌐 STARTING WEB SERVER{TerminalColors.END}")
    print(f"   {TerminalColors.CYAN}Local:{TerminalColors.END} http://localhost:5000")
    print(f"   {TerminalColors.CYAN}Network:{TerminalColors.END} http://{broker_ip}:5000")
    print(f"{TerminalColors.CYAN}{'='*50}{TerminalColors.END}")
    
    print(f"\n{TerminalColors.BOLD}{TerminalColors.GREEN}🎯 PHASE 2 FIXED: ALL ISSUES RESOLVED{TerminalColors.END}")
    print(f"{TerminalColors.GREEN}✅ ESP32 node management, visual mapping, interactive floor plans, user filtering!{TerminalColors.END}")
    print(f"{TerminalColors.CYAN}{'='*50}{TerminalColors.END}\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print(f"\n{TerminalColors.YELLOW}Received shutdown signal{TerminalColors.END}")
    except Exception as e:
        print(f"\n{TerminalColors.RED}Error running web server: {e}{TerminalColors.END}")
    finally:
        cleanup()

if __name__ == '__main__':
    main()

