"""
Script avulso (sqlite3): adiciona as colunas distance_m e hops à tabela neighbor,
usadas no vizinho mais próximo e na estimativa de distância.

Execução: python migrations/add_distance_fields.py
"""
import sqlite3
import os
import sys

# Inclui o diretório backend/ no path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lora_mesh.db")  # backend/lora_mesh.db

def migrate():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}")
        print("   The columns will be created automatically when the app starts.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Colunas existentes
    cursor.execute("PRAGMA table_info(neighbor)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    
    migrations_applied = 0
    
    if "hops" not in existing_columns:
        cursor.execute("ALTER TABLE neighbor ADD COLUMN hops INTEGER")
        print("✅ Added column: neighbor.hops (INTEGER)")
        migrations_applied += 1
    else:
        print("⏭️  Column neighbor.hops already exists")
    
    if "distance_m" not in existing_columns:
        cursor.execute("ALTER TABLE neighbor ADD COLUMN distance_m REAL")
        print("✅ Added column: neighbor.distance_m (REAL)")
        migrations_applied += 1
    else:
        print("⏭️  Column neighbor.distance_m already exists")
    
    conn.commit()
    conn.close()
    
    if migrations_applied > 0:
        print(f"\n✅ Migration complete: {migrations_applied} columns added")
    else:
        print("\n⏭️  Nothing to migrate — all columns already exist")

if __name__ == "__main__":
    migrate()
