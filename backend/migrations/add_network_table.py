"""
Script avulso (sqlite3): cria a tabela network e as colunas de rede em node.

Mudanças:
1. Criar tabela 'network' com id, name, description, active, timestamps
2. Adicionar colunas network_id, is_gateway, device_model à tabela 'node'
3. Criar registro "Rede Padrão" (id=1)
4. Associar todos os nós existentes à Rede 1
5. Marcar node_id=1 (0x0001) como is_gateway=True
"""

import sqlite3
from datetime import datetime, timezone
import os
import shutil


def backup_database(db_path):
    backup_path = f"{db_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if os.path.exists(db_path):
        shutil.copy2(db_path, backup_path)
        print(f"✅ Backup criado: {backup_path}")
        return backup_path
    return None


def migrate():
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lora_mesh.db')

    print("=" * 60)
    print("MIGRATION: Criar Network + Atualizar Node ")
    print("=" * 60)

    backup_path = backup_database(db_path)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()

        # 1. Criar tabela network (se não existir)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='network'")
        if cursor.fetchone():
            print("ℹ️  Tabela 'network' já existe. Pulando criação.")
        else:
            print("➕ Criando tabela 'network'...")
            cursor.execute("""
                CREATE TABLE network (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name VARCHAR(100) NOT NULL UNIQUE,
                    description VARCHAR(500),
                    active BOOLEAN NOT NULL DEFAULT 1,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX ix_network_active ON network (active)")
            print("✅ Tabela 'network' criada.")

        # 2. Adicionar colunas ao node (se não existirem)
        cursor.execute("PRAGMA table_info(node)")
        node_columns = [col[1] for col in cursor.fetchall()]

        if "network_id" not in node_columns:
            print("➕ Adicionando coluna 'network_id' ao node...")
            cursor.execute("ALTER TABLE node ADD COLUMN network_id INTEGER REFERENCES network(id)")
            cursor.execute("CREATE INDEX ix_node_network_id ON node (network_id)")
            print("✅ Coluna 'network_id' adicionada.")
        else:
            print("ℹ️  Coluna 'network_id' já existe.")

        if "is_gateway" not in node_columns:
            print("➕ Adicionando coluna 'is_gateway' ao node...")
            cursor.execute("ALTER TABLE node ADD COLUMN is_gateway BOOLEAN NOT NULL DEFAULT 0")
            print("✅ Coluna 'is_gateway' adicionada.")
        else:
            print("ℹ️  Coluna 'is_gateway' já existe.")

        if "device_model" not in node_columns:
            print("➕ Adicionando coluna 'device_model' ao node...")
            cursor.execute("ALTER TABLE node ADD COLUMN device_model VARCHAR(100)")
            print("✅ Coluna 'device_model' adicionada.")
        else:
            print("ℹ️  Coluna 'device_model' já existe.")

        conn.commit()

        # 3. Criar "Rede Padrão" se não existir
        cursor.execute("SELECT id FROM network WHERE id=1")
        if cursor.fetchone():
            print("ℹ️  Rede Padrão (id=1) já existe.")
        else:
            print("➕ Criando Rede Padrão (id=1)...")
            cursor.execute(
                "INSERT INTO network (name, description, active, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                ("Rede Padrão", "Rede migrada automaticamente com os dispositivos existentes", 1, now, now)
            )
            print("✅ Rede Padrão criada.")

        # 4. Associar nós existentes à Rede 1 (apenas os que ainda não têm rede)
        cursor.execute("UPDATE node SET network_id=1 WHERE network_id IS NULL")
        updated = cursor.rowcount
        print(f"✅ {updated} nó(s) associado(s) à Rede Padrão.")

        # 5. Marcar node_id=1 como gateway
        cursor.execute("UPDATE node SET is_gateway=1 WHERE node_id=1")
        cursor.execute("UPDATE node SET is_gateway=0 WHERE node_id != 1 AND is_gateway IS NULL")
        print("✅ node_id=1 marcado como gateway.")

        conn.commit()

        # Verificação final
        cursor.execute("SELECT id, name, active FROM network")
        networks = cursor.fetchall()
        print(f"\n📋 Redes cadastradas: {networks}")

        cursor.execute("PRAGMA table_info(node)")
        columns = [col[1] for col in cursor.fetchall()]
        print(f"📋 Colunas de node: {', '.join(columns)}")

        print("\n✅ Migração de redes concluída com sucesso!")
        conn.close()

    except Exception as e:
        print(f"\n❌ Migration falhou: {e}")
        if backup_path and os.path.exists(backup_path):
            print(f"⚠️  Restaure o backup de: {backup_path}")
        raise


if __name__ == "__main__":
    migrate()
