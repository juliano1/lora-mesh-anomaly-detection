"""
Script avulso (sqlite3): adiciona a coluna link_quality à tabela neighbor.
Índice de qualidade 0–10 calculado pelo firmware a partir de RSSI/SNR (não é PDR medido).
"""
import sqlite3
import os

def migrate(db_path=None):
    if db_path is None:
        # Default: mesmo diretório do backend
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lora_mesh.db")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Verificar se coluna já existe
    cursor.execute("PRAGMA table_info(neighbor)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if "link_quality" not in columns:
        cursor.execute(
            "ALTER TABLE neighbor ADD COLUMN link_quality INTEGER DEFAULT NULL"
        )
        print("✅ Coluna 'link_quality' adicionada à tabela Neighbor")
    else:
        print("ℹ️  Coluna 'link_quality' já existe")
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    migrate()
