"""Compatibilidade dos identificadores de versão de artefatos persistidos."""
import re


def version_matches(value, expected: str) -> bool:
    """Aceita a versão atual e seu antigo identificador com prefixo numerado.

    O sufixo precisa coincidir por inteiro: mudar a versão continua exigindo
    migração ou retreino. Esta comparação não autentica a origem do artefato.
    """
    if not isinstance(value, str):
        return False
    return value == expected or re.fullmatch(
        r"[a-z]+[0-9]{2,3}-" + re.escape(expected), value
    ) is not None
