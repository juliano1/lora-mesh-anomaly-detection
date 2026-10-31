"""
Script avulso (sqlite3): cria a tabela de vizinhos (relato de múltiplos enlaces).

Mudanças:
1. Adiciona a coluna lora_address à tabela node
2. Adiciona as colunas uptime e sequence à tabela telemetry
3. Cria a tabela neighbors com todos os vizinhos diretos
4. Cria os índices
"""

import sqlite3
from datetime import datetime
import os
import shutil

def backup_database(db_path):
    """Copia o banco antes da migração."""
    backup_path = f"{db_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if os.path.exists(db_path):
        shutil.copy2(db_path, backup_path)
        print(f"✅ Backup created: {backup_path}")
        return backup_path
    return None

def migrate():
    """Executa a migração."""
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lora_mesh.db')
    
    print("="*60)
    print("MIGRATION: Add Neighbors Table and Schema Updates")
    print("="*60)
    
    # Cópia de segurança
    backup_path = backup_database(db_path)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("\n📋 Applying schema changes...")
        
        # Verifica se as colunas já existem
        cursor.execute("PRAGMA table_info(node)")
        node_columns = [col[1] for col in cursor.fetchall()]
        
        cursor.execute("PRAGMA table_info(telemetry)")
        telemetry_columns = [col[1] for col in cursor.fetchall()]
        
        # 1. Coluna lora_address em node, se não existir
        if 'lora_address' not in node_columns:
            print("  → Adding lora_address column to node table...")
            cursor.execute('ALTER TABLE node ADD COLUMN lora_address VARCHAR(10)')
            cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_node_lora_addr ON node(lora_address)')
            print("  ✅ lora_address column added")
        else:
            print("  ℹ️  lora_address column already exists")
        
        # 2. Coluna uptime em telemetry, se não existir
        if 'uptime' not in telemetry_columns:
            print("  → Adding uptime column to telemetry table...")
            cursor.execute('ALTER TABLE telemetry ADD COLUMN uptime INTEGER')
            print("  ✅ uptime column added")
        else:
            print("  ℹ️  uptime column already exists")
        
        # 3. Coluna sequence em telemetry, se não existir
        if 'sequence' not in telemetry_columns:
            print("  → Adding sequence column to telemetry table...")
            cursor.execute('ALTER TABLE telemetry ADD COLUMN sequence INTEGER')
            print("  ✅ sequence column added")
        else:
            print("  ℹ️  sequence column already exists")
        
        # 4. Verifica se a tabela neighbors existe
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='neighbors'")
        neighbors_exists = cursor.fetchone() is not None
        
        if not neighbors_exists:
            print("  → Creating neighbors table...")
            cursor.execute('''
                CREATE TABLE neighbors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id INTEGER NOT NULL,
                    neighbor_address VARCHAR(10) NOT NULL,
                    rssi INTEGER NOT NULL,
                    snr REAL NOT NULL,
                    is_parent BOOLEAN DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (node_id) REFERENCES node(node_id)
                )
            ''')
            print("  ✅ neighbors table created")
            
            # Índices da tabela neighbors
            print("  → Creating indexes for neighbors table...")
            cursor.execute('CREATE INDEX idx_neighbor_node_addr ON neighbors(node_id, neighbor_address)')
            cursor.execute('CREATE INDEX idx_neighbor_timestamp ON neighbors(timestamp)')
            print("  ✅ Indexes created")
        else:
            print("  ℹ️  neighbors table already exists")
        
        # Grava as mudanças
        conn.commit()
        print("\n✅ Migration completed successfully!")
        
        # Mostra o esquema final
        print("\n📊 Final Schema:")
        print("\n  Node table columns:")
        cursor.execute("PRAGMA table_info(node)")
        for col in cursor.fetchall():
            print(f"    - {col[1]} ({col[2]})")
        
        print("\n  Telemetry table columns:")
        cursor.execute("PRAGMA table_info(telemetry)")
        for col in cursor.fetchall():
            print(f"    - {col[1]} ({col[2]})")
        
        print("\n  Neighbors table columns:")
        cursor.execute("PRAGMA table_info(neighbors)")
        for col in cursor.fetchall():
            print(f"    - {col[1]} ({col[2]})")
        
        conn.close()
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        if backup_path and os.path.exists(backup_path):
            print(f"💾 Restoring backup from: {backup_path}")
            shutil.copy2(backup_path, db_path)
            print("✅ Database restored")
        raise
    
    print("\n" + "="*60)
    print("Migration script finished")
    print("="*60)

def rollback(backup_path):
    """Desfaz a migração restaurando a cópia de segurança."""
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lora_mesh.db')
    if backup_path and os.path.exists(backup_path):
        shutil.copy2(backup_path, db_path)
        print(f"✅ Rollback completed. Database restored from: {backup_path}")
    else:
        print("❌ No backup found for rollback")

if __name__ == "__main__":
    migrate()
