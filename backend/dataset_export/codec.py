"""
Valores de CSV por tipo de coluna.

- Nulo: célula vazia SEM aspas; todo valor não nulo vai entre aspas
  (convenção `csv.QUOTE_NOTNULL`, com escritor compatível em Python 3.11).
  Assim string vazia ("") e nulo são
  distintos. `read_csv` preserva essa diferença também no Python 3.12, cujo
  reader padrão ainda não implementa a leitura de QUOTE_NOTNULL corretamente.
- Booleano: 0/1. Datas: ISO 8601 UTC com Z, microssegundos preservados.
- Float: `repr` (ida e volta exata); NaN/Inf são rejeitados.
"""
from __future__ import annotations

import csv
import math
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer

QUOTING = getattr(csv, "QUOTE_NOTNULL", csv.QUOTE_ALL)


def csv_writer(fh):
    """Convenção QUOTE_NOTNULL também no Python 3.11 (QUOTE_NOTNULL nativo só a partir do 3.12)."""
    native = csv.writer(fh, quoting=csv.QUOTE_NOTNULL) if hasattr(csv, "QUOTE_NOTNULL") else None

    class NotNullWriter:
        def writerow(self, values):
            values = list(values)
            if native is not None and values != [None]:
                return native.writerow(values)
            return fh.write(','.join(
                '' if value is None else '"' + str(value).replace('"', '""') + '"'
                for value in values) + '\r\n')

        def writerows(self, rows):
            for row in rows:
                self.writerow(row)

    return NotNullWriter()


def kind(column) -> str:
    if isinstance(column.type, Boolean):
        return "boolean"
    if isinstance(column.type, DateTime):
        return "datetime"
    if isinstance(column.type, Float):
        return "float"
    if isinstance(column.type, Integer):
        return "integer"
    return "string"


def _name(column) -> str:
    return f"{column.table.name}.{column.name}"


def encode(value, column):
    if value is None:
        return None
    k = kind(column)
    if k == "boolean":
        return "1" if value else "0"
    if k == "datetime":
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value.isoformat() + "Z"
    if k == "float":
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{_name(column)}: valor não finito ({number})")
        return repr(number)
    if k == "integer":
        return str(int(value))
    return str(value)


def decode(text, column):
    if text is None:
        if not column.nullable:
            raise ValueError(f"{_name(column)}: nulo em coluna obrigatória")
        return None
    k = kind(column)
    if k == "boolean":
        if text not in ("0", "1"):
            raise ValueError(f"{_name(column)}: booleano inválido {text!r}")
        return text == "1"
    if k == "datetime":
        if not text.endswith("Z"):
            raise ValueError(f"{_name(column)}: data sem Z {text!r}")
        value = datetime.fromisoformat(text[:-1])
        if value.tzinfo is not None:
            raise ValueError(f"{_name(column)}: offset adicional antes de Z")
        return value
    if k == "float":
        value = float(text)
        if not math.isfinite(value):
            raise ValueError(f"{_name(column)}: valor não finito")
        return value
    if k == "integer":
        return int(text)
    return text


def read_csv(fh):
    """Reader RFC 4180 com nulos sem aspas, em streaming (Python 3.11 e 3.12+).

    Não depende do suporte do módulo csv a QUOTE_NOTNULL na leitura.

    O parser csv valida/decodifica os campos. A forma original de cada registro
    informa quais vazios eram nulos; só um registro é mantido em memória.
    """
    consumed = []

    def lines():
        for line in fh:
            consumed.append(line)
            yield line

    reader = csv.reader(lines(), strict=True)
    while True:
        consumed.clear()
        try:
            values = next(reader)
        except StopIteration:
            return
        raw = "".join(consumed)
        if not values:
            # Na convenção NOTNULL, uma única célula nula é uma linha vazia.
            # Os schemas do pacote sempre têm ao menos uma coluna.
            yield [None]
            continue
        empty, quoted, start, i = [], False, 0, 0
        while i < len(raw):
            char = raw[i]
            if quoted:
                if char == '"':
                    if i + 1 < len(raw) and raw[i + 1] == '"':
                        i += 2
                        continue
                    quoted = False
            elif char == '"' and i == start:
                quoted = True
            elif char == ',':
                empty.append(i == start)
                start = i + 1
            elif char in '\r\n':
                empty.append(i == start)
                break
            i += 1
        else:
            empty.append(len(raw) == start)
        if len(empty) != len(values):
            raise ValueError("CSV: número de campos inconsistente")
        yield [None if null else value for null, value in zip(empty, values)]


def encode_value(value):
    """Codificação por tipo Python (arquivos derivados sem coluna de banco)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value.isoformat() + "Z"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"valor não finito ({value})")
        return repr(value)
    if isinstance(value, int):
        return str(value)
    return str(value)
