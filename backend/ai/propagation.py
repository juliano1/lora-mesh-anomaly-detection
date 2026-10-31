"""
Modelo de Propagação Rádio para Redes LoRa Mesh.

Calcula distância estimada e RSSI esperado por contexto de ambiente,
usando o modelo log-distance com shadowing e perda por obstáculos.

Referências:
  [R1] ITU-R P.1238 — Propagation data and prediction methods for
       indoor radio communication systems.
  [R2] ITU-R P.1411 — Propagation data for short-range outdoor
       radiocommunication systems.
  [R3] ITU-R P.2109 — Prediction of building entry loss.
  [R4] COST 231 Multi-Wall / Motley-Keenan — indoor path loss model.
  [R5] Rappaport, T. S. — Wireless Communications: Principles and Practice,
       2ª ed., Prentice Hall, 2002. Cap. 3 (Log-Distance Path Loss Model).
  [R6] Augustin, A. et al. — A Study of LoRa: Long Range & Low Power
       Networks for the Internet of Things. Sensors, 2016.
  [R7] Adelantado, F. et al. — Understanding the Limits of LoRaWAN.
       IEEE Communications Magazine, 2017.
"""

import math
import logging

logger = logging.getLogger(__name__)

# ─── RSSI de referência a 1 m ───────────────────────────────────────────────
# RSSI medido a 1 metro em linha de visada (LOS), em ambiente fechado,
# com antenas dipolo 868/915 MHz (Heltec LoRa 32 V2).
# Deve ser idêntico ao RSSI_REF_1M do firmware (firmware/src/main.cpp).
RSSI_REF_1M = -52  # dBm (medido indoor @1m; firmware: RSSI_REF_1M=-52)

# ─── RSSI de referência @1m por ambiente ────────────────────────────────────
# A referência indoor (-52) foi medida em ambiente fechado: multipath e campo
# próximo já atenuam o sinal a 1 m. Em LOS outdoor o sinal a 1 m é muito mais
# forte, então -52 subestima grosseiramente distâncias externas (ex.: enlaces
# reais de ~40 m da rede outdoor 1 resultam em ~3-6 m).
#
# Estimativa outdoor a partir de uma única medição em campo (rede "outdoor 1"):
# enlaces do gateway a ~40 m mediam -69/-73 dBm; com n=2.1 (open_field) isso
# implica ref ≈ -37 dBm.
#   ref = rssi + 10·n·log10(d) = -71 + 21·log10(40) ≈ -37
# Valor provisório — refine com uma medição de ponto único a distância conhecida.
_RSSI_REF_1M_BY_ENV: dict[str, float] = {
    "indoor":  -52.0,   # medição indoor a 1 m (mesmo valor do firmware)
    "outdoor": -37.0,   # estimativa provisória de uma medição (rede outdoor 1, n=2.1, ~40 m)
    "hybrid":  -44.0,   # ponto médio indoor/outdoor
}


def _reference_rssi_1m(network) -> float:
    """RSSI @1m de referência conforme environment_type da rede.

    Depende SOMENTE de environment_type (característica física do ambiente),
    não dos 5 campos de has_context — availability_criticality não deve afetar
    a referência de propagação. Sem rede/ambiente, mantém o global (-52).
    """
    env = getattr(network, "environment_type", None) if network is not None else None
    return _RSSI_REF_1M_BY_ENV.get(env, RSSI_REF_1M)

# ─── Expoente de perda de percurso (n) por perfil de ambiente ────────────────
# Fonte: [R1] ITU-R P.1238 (indoor) / [R2] ITU-R P.1411 (outdoor).
# Valores = ponto médio das faixas documentadas em cada norma.
#
# Faixas de referência:
#   outdoor aberto:   n = 2.0–2.2  (Rappaport [R5] / free-space)
#   outdoor urbano:   n = 2.7–3.5  (ITU-R P.1411 [R2])
#   indoor escritório:n = 3.0–3.5  (ITU-R P.1238 [R1])
#   indoor industrial:n = 2.0–3.0  (ITU-R P.1238 [R1])
#   indoor residencial:n= 2.8–3.3  (ITU-R P.1238 [R1])
_PATH_LOSS_EXPONENT: dict[tuple, float] = {
    ("outdoor", "open_field"):    2.1,   # free-space / campo aberto [R5]
    ("outdoor", "urban_street"):  3.1,   # ITU-R P.1411 [R2]
    ("outdoor", "other"):         2.7,   # fallback outdoor genérico
    ("indoor",  "office"):        3.25,  # ITU-R P.1238 [R1]
    ("indoor",  "industrial"):    2.5,   # ITU-R P.1238 [R1]
    ("indoor",  "residential"):   3.05,  # ITU-R P.1238 [R1]
    ("indoor",  "warehouse"):     2.3,   # galpão — próximo a industrial [R1]
    ("indoor",  "other"):         3.0,   # fallback indoor genérico
    ("hybrid",  None):            2.85,  # média outdoor-urbano / indoor-escritório
}

# Fallback quando não há par (env, building) no dicionário.
# Mantém compatibilidade com o valor hardcoded atual do sistema.
_N_DEFAULT = 2.7

# ─── Perda por parede (dB/parede) ───────────────────────────────────────────
# Fonte: COST 231 Multi-Wall [R4] / ITU-R P.1238 [R1] / ITU-R P.2109 [R3].
# Valores = ponto médio das faixas documentadas.
#
# Faixas de referência:
#   concreto:  12–15 dB  (COST 231 / ITU-R P.1238)
#   tijolo:     8–12 dB  (COST 231)
#   drywall:    3–5 dB   (COST 231)
#   madeira:    3–6 dB   (ITU-R P.1238)
#   vidro:      2–3 dB   (ITU-R P.2109)
#   metal:     15–25 dB  (COST 231)
#   misto:      6–10 dB  (estimativa conservadora)
_WALL_LOSS_DB: dict[str, float] = {
    "concrete": 13.5,
    "brick":    10.0,
    "drywall":   4.0,
    "wood":      4.5,
    "glass":     2.5,
    "metal":    20.0,
    "mixed":     8.0,
}

# ─── PDR esperado por perfil de densidade e faixa de distância ───────────────
# Escala adimensional 0–10, compatível com link_quality; não representa percentual de entrega medido.
# Valores heurísticos do projeto, inspirados qualitativamente em Augustin et al. [R6]
# e Adelantado et al. [R7]; esses trabalhos não publicam tabela nesta escala.
#
# Leitura: ambiente com obstáculos altos (high) e enlaces longos (long)
# tem PDR esperado baixo (3.5/10); campo aberto e enlaces curtos têm PDR alto.
_PDR_BASELINE: dict[tuple, float] = {
    ("low",    "short"):  9.0,
    ("low",    "medium"): 8.0,
    ("low",    "long"):   7.0,
    ("medium", "short"):  8.0,
    ("medium", "medium"): 6.5,
    ("medium", "long"):   5.0,
    ("high",   "short"):  7.0,
    ("high",   "medium"): 5.0,
    ("high",   "long"):   3.5,
}

# Fallback sem contexto — mantém comportamento atual do sistema.
_PDR_DEFAULT = 5.0


def get_propagation_params(network) -> dict:
    """
    Retorna parâmetros de propagação para uma rede com base no contexto de ambiente.

    Combina expoente de perda (n) por perfil de ambiente [R1][R2],
    perda por obstáculo (wall_loss_db) por material [R4] e
    PDR esperado por densidade e distância (heurística inspirada em [R6][R7]).

    Args:
        network: objeto Network (SQLModel) com o contexto de ambiente.
                 Aceita None — retorna defaults seguros (compat retroativa).

    Returns:
        dict com:
            n                    — expoente de perda de percurso
            wall_loss_db_per_wall — perda por parede em dB
            wall_loss_db_total   — perda total por paredes no enlace médio (dB)
            pdr_expected         — PDR esperado (escala 0–10)
            has_context          — True se os 5 campos obrigatórios estão preenchidos
                                   (environment_type, building_type, obstacle_density,
                                    typical_link_distance_class, availability_criticality)
    """
    # has_context = True apenas quando os 5 campos obrigatórios estão preenchidos,
    # alinhado com NetworkCreate e environment_context_complete.
    # Uma rede parcialmente cadastrada usa parâmetros contextuais disponíveis
    # mas não é considerada "com contexto completo" para fins de classificação.
    _required = [
        getattr(network, "environment_type", None),
        getattr(network, "building_type", None),
        getattr(network, "obstacle_density", None),
        getattr(network, "typical_link_distance_class", None),
        getattr(network, "availability_criticality", None),
    ] if network is not None else []
    has_context = all(f is not None for f in _required)

    if not has_context:
        return {
            "n":                    _N_DEFAULT,
            "wall_loss_db_per_wall": 0.0,
            "wall_loss_db_total":    0.0,
            "pdr_expected":          _PDR_DEFAULT,
            "reference_rssi_1m":     _reference_rssi_1m(network),
            "has_context":           False,
        }

    env   = network.environment_type
    btype = network.building_type

    # Lookup expoente n: tenta (env, btype) primeiro, fallback (env, "other")
    n = _PATH_LOSS_EXPONENT.get((env, btype))
    if n is None:
        if env == "hybrid":
            n = _PATH_LOSS_EXPONENT.get(("hybrid", None), _N_DEFAULT)
        else:
            n = _PATH_LOSS_EXPONENT.get((env, "other"), _N_DEFAULT)

    # Perda total por paredes
    wall_loss_per   = _WALL_LOSS_DB.get(network.predominant_obstacle_material or "", 0.0)
    avg_walls       = network.avg_walls_per_link or 0
    wall_loss_total = wall_loss_per * avg_walls

    # PDR esperado
    pdr_exp = _PDR_BASELINE.get(
        (network.obstacle_density, network.typical_link_distance_class),
        _PDR_DEFAULT,
    )

    logger.debug(
        f"PropagationProfile: env={env}/{btype} n={n:.2f} "
        f"wall={wall_loss_per:.1f}dB×{avg_walls}={wall_loss_total:.1f}dB "
        f"pdr_exp={pdr_exp:.1f}"
    )

    return {
        "n":                    n,
        "wall_loss_db_per_wall": wall_loss_per,
        "wall_loss_db_total":    wall_loss_total,
        "pdr_expected":          pdr_exp,
        "reference_rssi_1m":     _reference_rssi_1m(network),
        "has_context":           True,
    }


def rssi_expected(
    distance_m: float,
    n: float,
    wall_loss_db_total: float = 0.0,
    reference_rssi_1m: float = RSSI_REF_1M,
) -> float:
    """
    RSSI esperado a uma distância dada, com expoente n e perda por obstáculos.

    Modelo log-distance com perda adicional por paredes (Motley-Keenan / COST 231):
        RSSI(d) = reference_rssi_1m - 10·n·log10(d) - wall_loss_db_total

    Fontes: [R5] Rappaport Cap. 3 (log-distance); [R4] COST 231 (wall_loss).

    Args:
        distance_m:          Distância em metros (deve ser > 0).
        n:                   Expoente de perda de percurso.
        wall_loss_db_total:  Perda adicional total por obstáculos (dB).
        reference_rssi_1m:   RSSI @1m de referência (por ambiente; default indoor).

    Returns:
        RSSI esperado em dBm.
    """
    if distance_m <= 0:
        return float(reference_rssi_1m)
    return reference_rssi_1m - 10.0 * n * math.log10(distance_m) - wall_loss_db_total


def estimate_distance_contextual(
    rssi: int,
    n: float,
    wall_loss_db_total: float = 0.0,
    reference_rssi_1m: float = RSSI_REF_1M,
) -> float | None:
    """
    Estima distância a partir do RSSI usando o perfil de propagação da rede.

    Inversão do modelo log-distance:
        d = 10 ^ ((reference_rssi_1m - rssi - wall_loss_db_total) / (10·n))

    Fonte: [R5] Rappaport, Wireless Communications, Cap. 3.

    Args:
        rssi:                RSSI medido em dBm.
        n:                   Expoente de perda de percurso do perfil da rede.
        wall_loss_db_total:  Perda total por obstáculos no enlace (dB).
        reference_rssi_1m:   RSSI @1m de referência (por ambiente; default indoor).

    Returns:
        Distância estimada em metros, ou None se RSSI inválido.
    """
    if rssi is None or rssi >= 0 or rssi < -120:
        return None
    exponent = (reference_rssi_1m - rssi - wall_loss_db_total) / (10.0 * n)
    return round(math.pow(10.0, exponent), 1)
