"""
Script avulso (sqlite3): adiciona o estado persistente do LED ao nó.

Mudanças:
1. Adiciona a coluna led_state à tabela node (TEXT, padrão 'off')
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
    print("MIGRATION: Add led_state to Node")
    print("=" * 60)

    # Cópia de segurança
    backup_path = backup_database(db_path)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Verifica se a coluna já existe (idempotente)
        cursor.execute("PRAGMA table_info(node)")
        columns = [col[1] for col in cursor.fetchall()]

        if "led_state" in columns:
            print("ℹ️  Column 'led_state' already exists in node table. Skipping.")
        else:
            print("➕ Adding 'led_state' column to node table...")
            cursor.execute("ALTER TABLE node ADD COLUMN led_state TEXT DEFAULT 'off'")
            conn.commit()
            print("✅ Column 'led_state' added successfully.")

        # Confere o resultado
        cursor.execute("PRAGMA table_info(node)")
        columns = [col[1] for col in cursor.fetchall()]
        print(f"\n📋 Node columns: {', '.join(columns)}")

        assert "led_state" in columns, "Migration failed: led_state not found!"
        print("\n✅ Migration led_state completed successfully!")

        conn.close()

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        if backup_path and os.path.exists(backup_path):
            print(f"⚠️  Restore backup from: {backup_path}")
        raise


if __name__ == "__main__":
    migrate()
