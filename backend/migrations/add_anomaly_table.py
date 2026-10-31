"""
Script avulso (sqlite3): cria a tabela de anomalias da rede mesh LoRa.

Mudanças:
1. Cria a tabela 'anomaly' para as anomalias detectadas
2. Cria índices de desempenho (node_id, timestamp, anomaly_type, category, severity, resolved)
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
    """Executa a migração que cria a tabela anomaly."""
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'lora_mesh.db'
    )
    
    print("=" * 60)
    print("MIGRATION: Add Anomaly Table")
    print("=" * 60)
    
    # Cópia de segurança
    backup_path = backup_database(db_path)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Verifica se a tabela já existe
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='anomaly'
        """)
        if cursor.fetchone():
            print("ℹ️  Table 'anomaly' already exists. Skipping creation.")
            conn.close()
            return True
        
        # Cria a tabela anomaly
        print("\n1. Creating 'anomaly' table...")
        cursor.execute("""
            CREATE TABLE anomaly (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id INTEGER NOT NULL REFERENCES node(node_id),
                neighbor_address VARCHAR,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                anomaly_type VARCHAR NOT NULL,
                category VARCHAR NOT NULL,
                severity VARCHAR DEFAULT 'medium',
                description TEXT NOT NULL,
                suggested_action TEXT,
                evidence TEXT,
                detection_method VARCHAR DEFAULT 'rule',
                confidence REAL,
                resolved BOOLEAN DEFAULT 0,
                resolved_at DATETIME,
                notes TEXT
            )
        """)
        print("   ✅ Table 'anomaly' created")
        
        # Cria os índices
        print("\n2. Creating indexes...")
        
        indexes = [
            ("ix_anomaly_node_id", "anomaly(node_id)"),
            ("ix_anomaly_timestamp", "anomaly(timestamp)"),
            ("ix_anomaly_type", "anomaly(anomaly_type)"),
            ("ix_anomaly_category", "anomaly(category)"),
            ("ix_anomaly_severity", "anomaly(severity)"),
            ("ix_anomaly_resolved", "anomaly(resolved)"),
            ("ix_anomaly_node_type", "anomaly(node_id, anomaly_type)"),
            ("ix_anomaly_node_time", "anomaly(node_id, timestamp DESC)"),
        ]
        
        for idx_name, idx_def in indexes:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {idx_def}")
            print(f"   ✅ Index '{idx_name}' created")
        
        conn.commit()
        
        # Confere o resultado
        print("\n3. Verifying...")
        cursor.execute("PRAGMA table_info(anomaly)")
        columns = cursor.fetchall()
        print(f"   ✅ Table 'anomaly' has {len(columns)} columns:")
        for col in columns:
            print(f"      - {col[1]} ({col[2]})")
        
        conn.close()
        print("\n" + "=" * 60)
        print("✅ MIGRATION COMPLETE: Anomaly table ready")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ MIGRATION FAILED: {e}")
        if backup_path and os.path.exists(backup_path):
            print(f"   Restoring from backup: {backup_path}")
            shutil.copy2(backup_path, db_path)
            print("   ✅ Database restored")
        raise


if __name__ == "__main__":
    migrate()
