"""
Script avulso (sqlite3): adiciona o nível de bateria à telemetria.

Mudanças:
1. Adiciona a coluna battery_level à tabela telemetry (INTEGER, anulável)
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

    print("=" * 60)
    print("MIGRATION: Add battery_level to Telemetry")
    print("=" * 60)

    # Cópia de segurança
    backup_path = backup_database(db_path)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Verifica se a coluna já existe
        cursor.execute("PRAGMA table_info(telemetry)")
        columns = [col[1] for col in cursor.fetchall()]

        if "battery_level" in columns:
            print("ℹ️  Column 'battery_level' already exists in telemetry table. Skipping.")
        else:
            print("➕ Adding 'battery_level' column to telemetry table...")
            cursor.execute("ALTER TABLE telemetry ADD COLUMN battery_level INTEGER DEFAULT NULL")
            conn.commit()
            print("✅ Column 'battery_level' added successfully.")

        # Confere o resultado
        cursor.execute("PRAGMA table_info(telemetry)")
        columns = [col[1] for col in cursor.fetchall()]
        print(f"\n📋 Telemetry columns: {', '.join(columns)}")

        assert "battery_level" in columns, "Migration failed: battery_level not found!"
        print("\n✅ Migration battery_level completed successfully!")

        conn.close()

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        if backup_path and os.path.exists(backup_path):
            print(f"⚠️  Restore backup from: {backup_path}")
        raise


if __name__ == "__main__":
    migrate()
