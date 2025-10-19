"""
BLE Enhanced Terminal Visual Mapping System - PHASE 2: REAL-TIME LOCATION TRACKING
Features: Interactive Floor Plans with Real-Time Device Tracking, ESP32 Node Positioning, Movement Trails
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
    print(f"{TerminalColors.BOLD}{TerminalColors.BLUE}🗺️  BLE ENHANCED TERMINAL VISUAL MAPPING SYSTEM - PHASE 2{TerminalColors.END}")
    print(f"{TerminalColors.CYAN}{'='*70}{TerminalColors.END}")
    print(f"{TerminalColors.GREEN}✨ REAL-TIME LOCATION TRACKING WITH VISUAL MAPPING{TerminalColors.END}")
    print(f"{TerminalColors.YELLOW}📍 ESP32 Node Positioning on Interactive Floor Plans{TerminalColors.END}")
    print(f"{TerminalColors.MAGENTA}🔴 Color-coded BLE Device Tracking{TerminalColors.END}")
    print(f"{TerminalColors.BLUE}📈 Movement Trails and Predictive Paths{TerminalColors.END}")
    print(f"{TerminalColors.WHITE}🏢 Multi-floor Support with Layer Controls{TerminalColors.END}")
    print(f"{TerminalColors.CYAN}{'='*70}{TerminalColors.END}")

# Global variables
app = Flask(__name__)
mqtt_client = None
devices_data = {}
esp32_nodes = {}
device_positions = {}  # Store calculated device positions
movement_trails = {}   # Store movement history for trails
prediction_data = {}   # Store data for predictive paths

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
                floor_plan_id INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
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
        cursor.execute('SELECT id, node_id, name, room_name, x_position, y_position, floor_plan_id, created_at FROM esp32_nodes ORDER BY name')
        nodes = []
        for row in cursor.fetchall():
            nodes.append({
                'id': row[0],
                'node_id': row[1],
                'name': row[2],
                'room_name': row[3],
                'x_position': row[4],
                'y_position': row[5],
                'floor_plan_id': row[6],
                'created_at': row[7]
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
            if key in ['name', 'room_name', 'x_position', 'y_position', 'floor_plan_id']:
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
    
    def add_or_update_esp32_node(self, node_id, name=None, room_name=None, x_position=0, y_position=0):
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
                    y_position = COALESCE(?, y_position)
                WHERE node_id = ?
            ''', (name, room_name, x_position, y_position, node_id))
        else:
            # Add new node
            cursor.execute('''
                INSERT INTO esp32_nodes (node_id, name, room_name, x_position, y_position) 
                VALUES (?, ?, ?, ?, ?)
            ''', (node_id, name or f'ESP32-{node_id}', room_name or 'Unassigned', x_position, y_position))
        
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
    """Calculate device position based on ESP32 node positions and distances"""
    try:
        esp32_node = device_data.get('esp32_node')
        distance = float(device_data.get('distance', 0))
        confidence = float(device_data.get('confidence', 0))
        
        # Get ESP32 node position from database
        nodes = db.get_esp32_nodes()
        node_position = None
        
        for node in nodes:
            if node['node_id'] == esp32_node:
                node_position = (node['x_position'], node['y_position'])
                break
        
        if not node_position:
            # Default position if node not found
            node_position = (400, 300)
        
        # Simple positioning: place device at distance from node
        # In a real implementation, you'd use trilateration with multiple nodes
        angle = hash(device_data.get('device_id', '')) % 360  # Pseudo-random angle
        angle_rad = math.radians(angle)
        
        # Convert distance to pixels (assuming 50 pixels per meter)
        distance_pixels = distance * 50
        
        x = node_position[0] + distance_pixels * math.cos(angle_rad)
        y = node_position[1] + distance_pixels * math.sin(angle_rad)
        
        # Keep within canvas bounds
        x = max(50, min(750, x))
        y = max(50, min(550, y))
        
        return x, y, confidence
        
    except Exception as e:
        print(f"Error calculating position: {e}")
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
        client.subscribe("espresense/+/+/+")
        print(f"{TerminalColors.CYAN}📡 Subscribed to espresense topics{TerminalColors.END}")
    else:
        print(f"{TerminalColors.RED}❌ MQTT CONNECTION FAILED: {rc}{TerminalColors.END}")

def on_message(client, userdata, msg):
    """MQTT message callback with enhanced terminal logging"""
    try:
        topic_parts = msg.topic.split('/')
        if len(topic_parts) >= 4:
            esp32_node = topic_parts[1]
            device_id = topic_parts[2]
            
            # Parse message
            try:
                data = json.loads(msg.payload.decode())
                distance = data.get('distance', 0)
                confidence = data.get('confidence', 0)
            except:
                distance = float(msg.payload.decode()) if msg.payload.decode().replace('.', '').isdigit() else 0
                confidence = 85  # Default confidence
            
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
                    
                    active_devices.append(device_with_position)
            except:
                pass
    
    return jsonify({
        'success': True, 
        'devices': active_devices, 
        'timestamp': datetime.now().isoformat()
    })

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
        if 'floor_plan_id' in data:
            update_data['floor_plan_id'] = data['floor_plan_id']
        
        success = db.update_esp32_node_by_id(node_db_id, **update_data)
        
        if success:
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

# Network settings routes (from original)
@app.route('/api/network/status')
def get_network_status():
    """Get current network status"""
    network_info = get_network_info()
    
    # Get saved settings
    mqtt_ip = db.get_setting('mqtt_broker_ip', network_info['local_ip'])
    mqtt_port = db.get_setting('mqtt_broker_port', '1883')
    
    return jsonify({
        'success': True,
        'network': {
            'current_ip': network_info['local_ip'],
            'subnet_mask': network_info['subnet_mask'],
            'gateway': network_info['gateway'],
            'dns_servers': network_info['dns_servers'],
            'mqtt_broker_ip': mqtt_ip,
            'mqtt_broker_port': mqtt_port
        }
    })

@app.route('/api/network/apply', methods=['POST'])
def apply_network_settings():
    """Apply network settings"""
    try:
        data = request.get_json()
        
        # Save MQTT settings
        if 'mqtt_broker_ip' in data:
            db.set_setting('mqtt_broker_ip', data['mqtt_broker_ip'])
        if 'mqtt_broker_port' in data:
            db.set_setting('mqtt_broker_port', str(data['mqtt_broker_port']))
        
        return jsonify({'success': True, 'message': 'Network settings applied successfully!'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

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

# HTML Template with Phase 2 Features
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BLE Enhanced Terminal Visual Mapping System - Phase 2</title>
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
    </style>
</head>
<body>
    <div class="header">
        <h1>🗺️ BLE Enhanced Terminal Visual Mapping System - Phase 2</h1>
        <p>REAL-TIME LOCATION TRACKING • ESP32 Node Positioning • Movement Trails • Predictive Paths</p>
        <div class="status-badge">✅ PHASE 2: VISUAL MAPPING • PYTHON 3.13 COMPATIBLE • v2.0.0</div>
    </div>

    <div class="container">
        <div class="nav-tabs">
            <div class="nav-tab active" onclick="switchTab('overview')">📊 Overview</div>
            <div class="nav-tab" onclick="switchTab('visual-mapping')">🗺️ Visual Mapping</div>
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
                <h2>🎯 Phase 2: Real-Time Location Tracking Features!</h2>
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
                <p><strong>✨ New Features: Real-time device positioning, movement trails, and predictive paths!</strong></p>
            </div>
        </div>

        <!-- Visual Mapping Tab -->
        <div id="visual-mapping-tab" class="tab-content">
            <div class="mapping-container">
                <!-- Sidebar -->
                <div class="mapping-sidebar">
                    <h3>🎛️ Layer Controls</h3>
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
            <div id="esp32NodesList"></div>
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
        let esp32Nodes = [];
        let realTimeDevices = [];
        let isDragging = false;
        let dragNode = null;
        let dragOffset = { x: 0, y: 0 };

        // Initialize canvas
        function initCanvas() {
            canvas = document.getElementById('visualMappingCanvas');
            ctx = canvas.getContext('2d');
            
            // Add mouse event listeners
            canvas.addEventListener('mousedown', onMouseDown);
            canvas.addEventListener('mousemove', onMouseMove);
            canvas.addEventListener('mouseup', onMouseUp);
            canvas.addEventListener('mouseleave', onMouseUp);
            
            drawCanvas();
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
            realTimeDevices.forEach(device => {
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
            });
        }

        function drawMovementTrails() {
            realTimeDevices.forEach(device => {
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
            realTimeDevices.forEach(device => {
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
                
                // Draw coverage circle
                ctx.strokeStyle = '#667eea40';
                ctx.lineWidth = 2;
                ctx.setLineDash([10, 5]);
                
                ctx.beginPath();
                ctx.arc(x, y, 100, 0, 2 * Math.PI); // 2m radius at 50px/m
                ctx.stroke();
                
                ctx.setLineDash([]); // Reset line dash
            });
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

        function updateDevicesList() {
            const container = document.getElementById('devicesList');
            if (!container) return;
            
            container.innerHTML = realTimeDevices.map(device => `
                <div class="device-item active">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>${device.device_emoji} ${device.device_name}</strong>
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
            
            // Auto-refresh data
            setInterval(() => {
                loadStats();
                loadRealTimeDevices();
            }, 2000);
            
            setInterval(() => {
                loadDevicesList();
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
    
    print(f"\n{TerminalColors.BOLD}{TerminalColors.GREEN}🎯 PHASE 2: REAL-TIME VISUAL MAPPING ACTIVE{TerminalColors.END}")
    print(f"{TerminalColors.GREEN}✨ ESP32 node positioning, device tracking, movement trails, and predictions!{TerminalColors.END}")
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

